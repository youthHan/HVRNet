import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from pytorch_metric_learning.losses import TripletNonLocalLoss
from pytorch_metric_learning.miners import BatchHardMiner
from mmdet.core import (bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)

from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head import BBoxHead
from ..losses import accuracy


@HEADS.register_module
class HNMBBBoxHead(BBoxHead):
    r"""A selsa style non-local bboxhead for feature aggregation.

    bbox_feats_fc_1 -> selsa_1 -> bbox_feats_fc_2 -> selsa_2
    Caution: the feat_dim and fc_feat_dim 
             feat_dim has different meaning from that in SELSA
        
    """  # noqa: W605

    def __init__(self, 
                 sampler_num,
                 t_dim,
                 imgs_per_video,
                 fc_feat_dim=1024,
                 non_cur_space=False,
                 dim=(1024, 1024, 1024),
                 output_cur_only=False,
                 conv_z=[True,True,True],
                 conv_g=[False,False,False],
                 *args,
                 **kwargs):
        super(HNMBBBoxHead, self).__init__(*args, **kwargs)
        self.feat_dim = self.in_channels * self.roi_feat_area
        # self.selsa_num = 2
        #Calculated by t_dim * RPN_POST_NUM
        self.imgs_per_video = imgs_per_video
        self.sampler_num = sampler_num
        
        self.nongt_dim = sampler_num * t_dim
        self.t_dim = t_dim
        self.fc_feat_dim = fc_feat_dim
        self.non_cur_space = non_cur_space
        self.dim = dim
        self.output_cur_only=False
        self.conv_z = conv_z
        self.conv_g = conv_g

        self.selsa_1, self.selsa_2, self.selsa_3 = \
                self._add_selsa_with_fc(self.feat_dim, self.fc_feat_dim, 
                                        self.dim, self.conv_z, self.conv_g)

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.dim[2], self.num_classes)
            self.fc_cls_2 = nn.Linear(self.dim[2], self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.dim[2], out_dim_reg)
            self.fc_reg_2 = nn.Linear(self.dim[2], out_dim_reg)

    def _add_selsa_with_fc(self, 
                          feat_dim,
                          fc_feat_dim,
                          dim=(1024,1024,1024),
                          conv_z=[True,True],
                          conv_g=[False,False]):
        self.fc_new_1 = nn.Linear(feat_dim, fc_feat_dim)
        selsa_1 = OrderedDict(
            q_data_fc_1 = nn.Linear(fc_feat_dim, dim[0]),
            k_data_fc_1 = nn.Linear(fc_feat_dim, dim[1]),
            aff_softmax_1 = nn.Softmax(dim=2)
        )
        if conv_g[0]:
            selsa_1.update(v_data_fc_1 = nn.Linear(fc_feat_dim, dim[2]))
        if conv_z[0]:
            selsa_1.update(linear_out_1 = nn.Conv2d(dim[2], dim[2], 1))
        selsa_1 = nn.ModuleDict(selsa_1)

        self.fc_new_2 = nn.Linear(dim[2], fc_feat_dim)
        selsa_2 = OrderedDict(
            q_data_fc_2 = nn.Linear(fc_feat_dim, dim[0]),
            k_data_fc_2= nn.Linear(fc_feat_dim, dim[1]),
            aff_softmax_2 = nn.Softmax(dim=2)
        )
        if conv_g[1]:
            selsa_2.update(v_data_fc_2 = nn.Linear(fc_feat_dim, dim[2]))
        if conv_z[1]:
            selsa_2.update(linear_out_2 = nn.Conv2d(dim[2], dim[2], 1))
        selsa_2 = nn.ModuleDict(selsa_2)

        self.fc_new_3 = nn.Linear(dim[2], fc_feat_dim)
        selsa_3 = OrderedDict(
            q_data_fc_3 = nn.Linear(fc_feat_dim, dim[0]),
            k_data_fc_3= nn.Linear(fc_feat_dim, dim[1]),
            aff_softmax_3 = nn.Softmax(dim=2)
        )
        if conv_g[1]:
            selsa_3.update(v_data_fc_3 = nn.Linear(fc_feat_dim, dim[2]))
        if conv_z[1]:
            selsa_3.update(linear_out_3 = nn.Conv2d(dim[2], dim[2], 1))
        selsa_3 = nn.ModuleDict(selsa_3)
        
        return selsa_1, selsa_2, selsa_3

    #TODO: Rewrite this function
    def init_weights(self):
        super(HNMBBBoxHead, self).init_weights()
        # print("Param Init in SelsaBBoxhead for fc has changed to uniform, rather than xavier_uniform")
        for module_list in [self.fc_new_1, self.selsa_1,
                            self.fc_new_2, self.selsa_2,
                            self.fc_new_3, self.selsa_3]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    # nn.init.xavier_uniform_(m.weight)
                    nn.init.normal(m.weight, 0.0, 0.01)
                    nn.init.constant_(m.bias, 0)
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
            nn.init.normal_(self.fc_cls_2.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls_2.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.01)
            nn.init.constant_(self.fc_reg.bias, 0)
            nn.init.normal_(self.fc_reg_2.weight, 0, 0.01)
            nn.init.constant_(self.fc_reg_2.bias, 0)

    def hardest_proposal_mining(self,
                                labels, 
                                all_labels, 
                                aff_scale,
                                metric_loss=None):
        # Get class mask
        # inds = aff_softmax_reshape.new_zeros(aff_scale.shape)
        all_labels_n = all_labels.unsqueeze(dim=0).repeat_interleave(labels.shape[0], dim=0)
        label_mask = all_labels_n.permute(1,0).ne(labels).permute(1,0)
        # ! for cls_id == 0
        bg_mask = label_mask.clone().detach()
        ind_cls_not_bg = torch.where(labels != 0)[0]
        bg_mask[ind_cls_not_bg] = False
        bg_aff_scale = aff_scale.mul(bg_mask.to(torch.float))
        bg_aff_scale.masked_fill_(bg_mask.logical_not(), float('-inf'))
        inds_for_bg = bg_aff_scale.topk(2, dim=2).indices
        bg_collect = all_labels_n.new_zeros(all_labels_n.shape).to(torch.float)
        bg_collect.scatter_(1, inds_for_bg.squeeze_(dim=0), inds_for_bg.new_ones(inds_for_bg.shape).to(torch.float))

        # ! for cls_id != 0, pos_sm for positive same class
        pos_sm_mask = label_mask.clone().detach()
        ind_cls_bg = torch.where(labels == 0)[0]
        pos_sm_mask[ind_cls_bg] = False
        pos_sm_aff_scale = aff_scale.mul(pos_sm_mask.to(torch.float))
        pos_sm_aff_scale.masked_fill_(pos_sm_mask.logical_not(), float('-inf'))
        inds_for_pos_sm = pos_sm_aff_scale.topk(1, dim=2).indices
        pos_sm_collect = all_labels_n.new_zeros(all_labels_n.shape).to(torch.float)
        pos_sm_collect.scatter_(1, inds_for_pos_sm.squeeze_(dim=0), inds_for_pos_sm.new_ones(inds_for_pos_sm.shape).to(torch.float))

        # ! for cls_id != 0, pos_nsm for positive not same class
        pos_nsm_mask = label_mask.logical_not().clone().detach()
        ind_cls_bg = torch.where(labels == 0)[0]
        pos_nsm_mask[ind_cls_bg] = False
        pos_nsm_aff_scale = aff_scale.mul(pos_nsm_mask.to(torch.float))
        pos_nsm_aff_scale.masked_fill_(pos_nsm_mask.logical_not(), float('inf'))
        inds_for_pos_nsm = pos_nsm_aff_scale.topk(1, dim=2, largest=False).indices
        pos_nsm_collect = all_labels_n.new_zeros(all_labels_n.shape).to(torch.float)
        pos_nsm_collect.scatter_(1, inds_for_pos_nsm.squeeze_(dim=0), inds_for_pos_nsm.new_ones(inds_for_pos_nsm.shape).to(torch.float))

        bg_collect[ind_cls_not_bg]=0.
        pos_sm_collect[ind_cls_bg]=0.
        pos_nsm_collect[ind_cls_bg]=0.
        inds_collect = bg_collect + pos_sm_collect + pos_nsm_collect

        # if metric_loss is not None:
        #     q_data_metric = q_data_batch.squeeze()
        #     k_data_metric = k_data_batch.permute(0, 2, 1).squeeze()
        #     not_bg_inds = torch.where(labels!=0)
        #     pos_indices = torch.gather(inds_for_pos_sm.squeeze(), 0, ind_cls_not_bg)
        #     neg_indices = torch.gather(inds_for_pos_nsm.squeeze(), 0, ind_cls_not_bg)
        #     m_loss = metric_loss.compute_loss(q_data_metric, k_data_metric, labels, 
        #                             [ind_cls_not_bg, pos_indices, neg_indices])
        #     return [ind_cls_not_bg, pos_indices, neg_indices], m_loss
        # else:
        # ! Bug exists in above code: pos_sm and pos_nsm are in wrong positions
        pos_indices = torch.gather(inds_for_pos_sm.squeeze(), 0, ind_cls_not_bg)
        neg_indices = torch.gather(inds_for_pos_nsm.squeeze(), 0, ind_cls_not_bg)
        # ! please change the order of following hardest_indices: neg_indices is actually the positive ones
        return [ind_cls_not_bg, pos_indices, neg_indices]

    def forward_single_selsa(self, 
                            roi_feat, 
                            key_dim,
                            nongt_dim, 
                            index, 
                            cur_range_s=None,
                            non_cur_space=False, 
                            idx_output_cur_only=False,
                            mining=False,
                            labels=None,
                            all_labels=None,
                            metric_loss=None):
        '''
        args:
        '''
        # assert not non_cur_space and not idx_output_cur_only, "Under HNMBBBoxHead, these two options haven't been carefully reviewed"
        assert not non_cur_space, "Under HNMBBBoxHead, `non_cur_space` hasn't been carefully reviewed"
        if non_cur_space or idx_output_cur_only:
            assert cur_range_s is not None, "Fearture range of current frame need specified"
            # feat_select_tensor = torch.tensor(range(cur_range['start'], cur_range['start']+cur_range['length']))
            # feat_exclude_tensor = torch.tensor([i for i in range(roi_feat.shape[0]) if i not in feat_select_tensor])

            feat_strt_dims = []
            feat_lens = []
            for cur_range in cur_range_s:
                feat_strt_dim = cur_range['start']
                feat_len = cur_range['length']
                feat_strt_dims.append(feat_strt_dim)
                feat_lens.append(feat_len)

        dim=self.dim
        conv_z=self.conv_z
        conv_g=self.conv_g
        nongt_roi_feat = roi_feat[0:nongt_dim]

        # TODO: Fix this function under HNMBBBoxHead
        if non_cur_space:
            # assert nongt_roi_feat.shape[0]%t_dim == 0, "The len of roi_feat should be equal among different frames"
            # nongt_roi_feat_split = torch.split(nongt_roi_feat, 1, dim=0)
            if feat_strt_dim != 0:
                cur_nongt_roi_feat = torch.cat([nongt_roi_feat[0:feat_strt_dim], 
                                                nongt_roi_feat[feat_strt_dim+feat_len:]],
                                                dim=0)
                # cur_nongt_roi_feat = torch.cat([nongt_roi_feat.narrow_copy(0, 0, key_dim), 
                #                     nongt_roi_feat.narrow_copy(0, key_dim+cur_range['length'], nongt_roi_feat.shape[0]-cur_range['length']-key_dim)],dim=0)
            else:
                cur_nongt_roi_feat = nongt_roi_feat[feat_len:]
                # cur_nongt_roi_feat = nongt_roi_feat.narrow_copy(0, key_dim + cur_range['length'], nongt_roi_feat.shape[0]-cur_range['length'])
            # cur_nongt_roi_feat = torch.index_select(nongt_roi_feat, dim=0, index=feat_exclude_tensor)
            nongt_roi_feat = cur_nongt_roi_feat
        
        #########
        if idx_output_cur_only:
            # assert roi_feat.shape[0]%t_dim == 0, "The len of roi_feat should be equal among different frames"
            # roi_feat = torch.index_select(roi_feat, dim=0, index=feat_select_tensor)
            roi_feat_s = []
            for i in range(len(cur_range_s)):
                feat_strt_dim = feat_strt_dims[i]
                feat_len = feat_lens[i]
                roi_feat_s.append(roi_feat[feat_strt_dim:feat_strt_dim+feat_len])
            roi_feat = torch.cat(roi_feat_s, dim=0)
            del roi_feat_s
        ########

        # multi head
        assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
        q_data = getattr(self, 'selsa_%d'%index)['q_data_fc_%d'%index](roi_feat)
        q_data_batch = torch.reshape(q_data, shape=(-1,1,dim[0]))
        q_data_batch = q_data_batch.permute(1, 0, 2)
        k_data = getattr(self, 'selsa_%d'%index)['k_data_fc_%d'%index](nongt_roi_feat)
        k_data_batch = torch.reshape(k_data, shape=(-1,1,dim[1])) 
        k_data_batch = k_data_batch.permute(1, 2, 0)   
        v_data = nongt_roi_feat
        #!!Caution: the shape of the batch dot result
        # print(q_data.shape)
        # print(v_data.shape)
        aff = torch.bmm(q_data_batch, k_data_batch)
        aff_scale = (1.0 / math.sqrt(float(dim[1]))) * aff
        aff_scale.permute(1, 0, 2)

        if mining:
            aff_scale_key_frames = []
            for i in range(len(cur_range_s)):
                feat_strt_dim = feat_strt_dims[i]
                feat_len = feat_lens[i]
                aff_scale_key_frames.append(aff_scale[:, :, feat_strt_dim:feat_strt_dim+feat_len])
            aff_scale_key_frames = torch.cat(aff_scale_key_frames, dim=2)
            anchor_idx, hardest_pos_idx, hardest_neg_idx = self.hardest_proposal_mining(labels, labels, aff_scale_key_frames, None)
            # miner = BatchHardMiner(use_similarity=True, squared_distances=False)
            q_data_metric = q_data_batch.squeeze()
            k_data_metric = k_data_batch.permute(0, 2, 1).squeeze()
            if idx_output_cur_only:
                k_data_s = []
                k_labels_s = []
                for i in range(len(cur_range_s)):
                    feat_strt_dim = feat_strt_dims[i]
                    feat_len = feat_lens[i]
                    k_data_s.append(k_data_metric[feat_strt_dim:feat_strt_dim+feat_len])
                    # k_labels_s.append(all_labels[feat_strt_dim:feat_strt_dim+feat_len])
                k_data_metric = torch.cat(k_data_s, dim=0)
            # k_labels_metric = torch.cat(k_labels_s, dim=0)
            # anchor_indices, hardest_positive_indices, hardest_negative_indices = \
            #                         miner(q_data_metric, labels, k_data_metric, k_labels_metric)
            # # anchor_indices, hardest_positive_indices, hardest_negative_indices = \
            # #                         miner(q_data_batch, labels, q_data_batch.permute(0,2,1), labels)
            # m_loss = metric_loss.compute_loss(q_data_metric, k_data_metric, labels, 
            #                             [anchor_indices[torch.where(labels!=0)], 
            #                             hardest_positive_indices[torch.where(labels!=0)], 
            #                             hardest_negative_indices[torch.where(labels!=0)]])
            m_loss = metric_loss.compute_loss(q_data_metric, k_data_metric, labels, 
                                        [anchor_idx, hardest_pos_idx, hardest_neg_idx])
        else:
            m_loss = None

        weighted_aff = aff_scale
        aff_softmax = getattr(self, 'selsa_%d'%index)['aff_softmax_%d'%index](weighted_aff)
        #!!Caution: the copy of the tensor, may cause problem 
        # similarity = torch.empty(aff_softmax.shape, 
        #                         dtype=aff_softmax.dtype, 
        #                         device=aff_softmax.device).copy_(aff_softmax)
        
        aff_softmax_reshape = aff_softmax.view(-1, aff_softmax.shape[2])
        
        if conv_g[index-1]:
            v_data = getattr(self, 'selsa_%d'%index)['v_data_fc_%d'%index](v_data)
        output_t = torch.mm(aff_softmax_reshape, v_data)
        output_t = torch.reshape(output_t, shape=(-1, self.fc_feat_dim, 1, 1))

        if conv_z[index-1]:
            linear_out = getattr(self, 'selsa_%d'%index)['linear_out_%d'%index](output_t)
        else:
            linear_out = output_t
        
        output = torch.reshape(linear_out, shape=linear_out.shape[:2])
        
        if metric_loss is not None:
            return output, m_loss, None
        else:
            return output, None

    #TODO: Rewrite this function, most of the functionality needs to be implemented here
    def forward(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False, others=None, all_labels=None, dynamic=None):
        """Main Selsa Functionality implementation

        Args:
            bbox_feat: Conved style features in shape [roi_nums, num_channel, roi_feat, roi_feat]

        Returns:

        """
        assert cur_range_s is not None, "Feature num range along axis needs specified, \
                                        as length of features for each frame could be different"
        self.key_dim = key_dim
        self.nongt_dim = self.sampler_num * self.t_dim
        
        if type(all_labels) is list:
            all_labels = torch.cat(all_labels, dim=0)

        num_videos = len(cur_range_s)
        video_feats = []
        key_frames_video_feats = []
        bbox_num_per_video = []
        cls_res_branches = []
        reg_res_branches = []
        imgs_per_video = self.imgs_per_video
        sampler_num = self.sampler_num
        loss_additional = dict()
        for i, bbox_feat in enumerate(bbox_feat_s):
            cur_range = cur_range_s[i]
            feat_strt_dim = cur_range['start']
            feat_len = cur_range['length']
            # print('enter aggregation module')
            bbox_num_per_video.append(bbox_feat.size(0))
            bbox_feat = bbox_feat.view(bbox_feat.size(0), -1)
            fc_new_feat_1 = self.fc_new_1(bbox_feat)
            attention_1, _ = self.forward_single_selsa(fc_new_feat_1, self.key_dim, 
                                    imgs_per_video*sampler_num, index=1, idx_output_cur_only=False, cur_range_s=[cur_range])
            fc_all_1 = fc_new_feat_1 + attention_1
            del fc_new_feat_1
            del attention_1
            fc_all_1_relu = self.relu(fc_all_1)
            del fc_all_1

            fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
            attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                    imgs_per_video*sampler_num, index=2, idx_output_cur_only=False, cur_range_s=[cur_range])
            # if self.output_cur_only:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
            fc_all_2 = fc_new_feat_2 + attention_2
            del fc_new_feat_2
            del attention_2

            # ! whether use all output of selsa_2 or not
            # ############
            # if not self.output_cur_only:
            #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            #     # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            #     fc_all_2 = fc_all_2[feat_strt_dim:feat_strt_dim+feat_len]
            # ############
            fc_all_2_relu = self.relu(fc_all_2)
            key_frames_video_feats.append(fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len])
            del fc_all_2
            video_feats.append(fc_all_2_relu)

            fc_all_2_branch_relu = fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len]
            cls_res_branches.append(self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None) 
            reg_res_branches.append(self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None)
            del fc_all_2_branch_relu

        all_frames = True
        if all_frames:
            cur_only_for_3 = True
            # ! please check the order of "hardest_pos_inds" and "hardest_neg_inds"
            mining_3 = True
            loss_metric = TripletNonLocalLoss(margin=10) 
            video_feats = torch.cat(video_feats, dim=0)
            fc_new_feat_3 = self.fc_new_3(video_feats)
            target_ranges_3 = []
            for i, cur_range in enumerate(cur_range_s):
                feat_strt_dim = cur_range['start'] + sum(bbox_num_per_video[:i])
                feat_len = cur_range['length']
                target_ranges_3.append(dict(start=feat_strt_dim, length=feat_len))
                # cur_labels.extend(all_labels[feat_strt_dim:feat_strt_dim+feat_len])            
            cur_labels = others if cur_only_for_3 else all_labels
            attention_3, m_loss, _ = self.forward_single_selsa(fc_new_feat_3, None, self.nongt_dim, index=3, 
                                    idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3, mining=mining_3, 
                                    labels=cur_labels, all_labels=all_labels, metric_loss=loss_metric)
            if cur_only_for_3:
                # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
                feats_key_frames = []
                for i, cur_range in enumerate(target_ranges_3):
                    feat_strt_dim = cur_range['start']
                    feat_len = cur_range['length']
                    feats_key_frames.append(fc_new_feat_3[feat_strt_dim : feat_strt_dim+feat_len])
                del fc_new_feat_3
                fc_new_feat_3 = torch.cat(feats_key_frames, dim=0)

            fc_all_3 = fc_new_feat_3 + attention_3
            del fc_new_feat_3
            del attention_3

            ############
            if not cur_only_for_3:
                # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
                # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
                feats_key_frames = []
                for i, cur_range in enumerate(target_ranges_3):
                    feat_strt_dim = cur_range['start']
                    feat_len = cur_range['length']
                    feats_key_frames.append(fc_all_3[feat_strt_dim : feat_strt_dim+feat_len])
                del fc_all_3
                fc_all_3 = torch.cat(feats_key_frames, dim=0)
        else:
            cur_only_for_3 = False
            key_frames_video_feats = torch.cat(key_frames_video_feats, dim=0)
            fc_new_feat_3 = self.fc_new_3(key_frames_video_feats)
            target_ranges_3 = []
            for i, cur_range in enumerate(cur_range_s):
                feat_strt_dim = cur_range['start'] + sum(bbox_num_per_video[:i])
                feat_len = cur_range['length']
                target_ranges_3.append(dict(start=feat_strt_dim, length=feat_len))
            attention_3, _ = self.forward_single_selsa(fc_new_feat_3, None, 
                                    self.nongt_dim, index=3, idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3)
            # if cur_only_for_3:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     feats_key_frames = []
            #     for i, cur_range in enumerate(target_ranges_3):
            #         feat_strt_dim = cur_range['start']
            #         feat_len = cur_range['length']
            #         feats_key_frames.append(fc_new_feat_3[feat_strt_dim : feat_strt_dim+feat_len])
            #     del fc_new_feat_3
            #     fc_new_feat_3 = torch.cat(feats_key_frames, dim=0)

            fc_all_3 = fc_new_feat_3 + attention_3
            del fc_new_feat_3
            del attention_3

            # ! below is unnecessary as the output of selsa_3 only contains key frames themselves
            ############
            # if not cur_only_for_3:
            #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            #     # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            #     feats_key_frames = []
            #     for i, cur_range in enumerate(target_ranges_3):
            #         feat_strt_dim = cur_range['start']
            #         feat_len = cur_range['length']
            #         feats_key_frames.append(fc_all_3[feat_strt_dim : feat_strt_dim+feat_len])
            #     del fc_all_3
            #     fc_all_3 = torch.cat(feats_key_frames, dim=0)
            ############
        fc_all_3_relu = self.relu(fc_all_3)

        # loss_metric = TripletNonLocalLoss(margin=0.2) 
        # labels = others
        # fc_all_3_relu_aff = torch.matmul(fc_all_3_relu, fc_all_3_relu.t()).unsqueeze(dim=0)
        # anchor_idx, hardest_pos_idx, hardest_neg_idx = self.hardest_proposal_mining(labels, 
        #                                                                             labels, 
        #                                                                             fc_all_3_relu_aff, 
        #                                                                             None)
        # m_loss = loss_metric.compute_loss(fc_all_3_relu, fc_all_3_relu, labels, 
        #                             [anchor_idx, hardest_pos_idx, hardest_neg_idx])
        loss_additional.update(dict(loss_trip=m_loss))

        cls_score = self.fc_cls_2(fc_all_3_relu) if self.with_cls else None
        bbox_pred = self.fc_reg_2(fc_all_3_relu) if self.with_reg else None

        cls_score_branch = torch.cat(cls_res_branches, dim=0)
        bbox_pred_branch = torch.cat(reg_res_branches, dim=0)
        # cls_res_branches.append(cls_score)
        # reg_res_branches.append(bbox_pred)
        assert cls_score_branch.shape[0] == bbox_pred_branch.shape[0]
        assert cls_score_branch.shape[0] == cls_score.shape[0]

        if len(loss_additional) == 0:
            return [cls_score_branch, cls_score], [bbox_pred_branch, bbox_pred], None
        else:
            return [cls_score_branch, cls_score], [bbox_pred_branch, bbox_pred], loss_additional
        # return [cls_score], [bbox_pred]

#TODO: Rewrite this function, most of the functionality needs to be implemented here
    def forward_test(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False):
        """Main Selsa Functionality implementation

        Args:
            bbox_feat: Conved style features in shape [roi_nums, num_channel, roi_feat, roi_feat]

        Returns:

        """
        assert cur_range_s is not None, "Feature num range along axis need specified, \
                                        as length of features for each frame could be different"
        self.key_dim = key_dim
        self.nongt_dim = self.sampler_num * self.t_dim
        
        # num_videos = len(cur_range_s)
        # video_feats = []
        # bbox_num_per_video = []
        # imgs_per_video = self.imgs_per_video
        # sampler_num = self.sampler_num
      
        cur_range = cur_range_s[0]
        bbox_feat = bbox_feat_s
        # cur_range = cur_range_s[i]
        feat_strt_dim = cur_range['start']
        feat_len = cur_range['length']
        # print('enter aggregation module')
        # bbox_num_per_video.append(bbox_feat.size(0))
        bbox_feat = bbox_feat.view(bbox_feat.size(0), -1)
        fc_new_feat_1 = self.fc_new_1(bbox_feat)
        attention_1, _ = self.forward_single_selsa(fc_new_feat_1, self.key_dim, 
                                self.nongt_dim, index=1, idx_output_cur_only=False, cur_range_s=[cur_range])
        fc_all_1 = fc_new_feat_1 + attention_1
        del fc_new_feat_1
        del attention_1
        fc_all_1_relu = self.relu(fc_all_1)
        del fc_all_1

        fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
        attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                self.nongt_dim, index=2, idx_output_cur_only=self.output_cur_only, cur_range_s=[cur_range])
        # if self.output_cur_only:
        #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
        #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
        fc_all_2 = fc_new_feat_2 + attention_2
        del fc_new_feat_2
        del attention_2
        # ############
        # if not self.output_cur_only:
        #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
        #     # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
        #     fc_all_2 = fc_all_2[feat_strt_dim:feat_strt_dim+feat_len]
        # ############
        fc_all_2_relu = self.relu(fc_all_2)
        del fc_all_2
        # video_feats.append(fc_all_2_relu)

        fc_all_2_branch_relu = fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len]
        cls_res_branch = self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None
        reg_res_branch = self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None

        # video_feats = torch.cat(video_feats, dim=0)
        # fc_new_feat_3 = self.fc_new_3(video_feats)
        fc_new_feat_3 = self.fc_new_3(fc_all_2_relu)
        attention_3, _ = self.forward_single_selsa(fc_new_feat_3, None, 
                                  self.nongt_dim, index=3, idx_output_cur_only=self.output_cur_only)
        
        
        if self.output_cur_only:
            # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            fc_new_feat_3 = fc_new_feat_3[feat_strt_dim:feat_strt_dim+feat_len]
        fc_all_3 = fc_new_feat_3 + attention_3

        ############
        if not self.output_cur_only:
            # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            fc_all_3 = fc_all_3[feat_strt_dim:feat_strt_dim+feat_len]
        ############
        fc_all_3_relu = self.relu(fc_all_3)

        del fc_new_feat_3
        del attention_3

        cls_score = self.fc_cls_2(fc_all_3_relu) if self.with_cls else None
        bbox_pred = self.fc_reg_2(fc_all_3_relu) if self.with_reg else None

        return [cls_res_branch, cls_score], [reg_res_branch, bbox_pred]

    def forward_test_multi_passes(self, bbox_feat_s, cur_range_multi, cur_range_s, key_dim=0, all_res=False):
        
        # num_videos = len(cur_range_s)
        # video_feats = []
        # bbox_num_per_video = []
        # imgs_per_video = self.imgs_per_video
        # sampler_num = self.sampler_num
        assert cur_range_s is not None, "Feature num range along axis need specified, \
                                        as length of features for each frame could be different"
        self.key_dim = key_dim
        self.nongt_dim = self.sampler_num * self.t_dim

        for i in range(len(cur_range_multi)-1,-1,-1):
            cur_range_multi[i] = [sum(cur_range_multi[:i]), cur_range_multi[i]]

        video_feats = []
        for i, cur_range_m in enumerate(cur_range_multi):
            bbox_feat = bbox_feat_s[cur_range_m[0]:cur_range_m[0]+cur_range_m[1]]
            bbox_feat = bbox_feat.view(bbox_feat.size(0), -1)
            fc_new_feat_1 = self.fc_new_1(bbox_feat)
            attention_1, _ = self.forward_single_selsa(fc_new_feat_1, None, 
                                    self.nongt_dim, index=1, idx_output_cur_only=False)
            fc_all_1 = fc_new_feat_1 + attention_1
            del fc_new_feat_1
            del attention_1
            fc_all_1_relu = self.relu(fc_all_1)
            del fc_all_1

            fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
            attention_2, _ = self.forward_single_selsa(fc_new_feat_2, None, 
                                    self.nongt_dim, index=2, idx_output_cur_only=False)
            fc_all_2 = fc_new_feat_2 + attention_2
            del fc_new_feat_2
            del attention_2
            fc_all_2_relu = self.relu(fc_all_2)
            del fc_all_2
            video_feats.append(fc_all_2_relu)

        video_feats = torch.cat(video_feats, dim=0)
        fc_new_feat_3 = self.fc_new_3(video_feats)
        attention_3, _ = self.forward_single_selsa(fc_new_feat_3, None, 
                                  self.nongt_dim, index=3, idx_output_cur_only=True, cur_range_s=cur_range_s)
        feat_strt_dim = cur_range_s[0]['start']
        feat_len = cur_range_s[0]['length']
        fc_new_feat_3 = fc_new_feat_3[feat_strt_dim : feat_strt_dim+feat_len]

        fc_all_3 = fc_new_feat_3 + attention_3
        del fc_new_feat_3
        del attention_3

        ############
        fc_all_3_relu = self.relu(fc_all_3)

        cls_score = self.fc_cls_2(fc_all_3_relu) if self.with_cls else None
        bbox_pred = self.fc_reg_2(fc_all_3_relu) if self.with_reg else None

        return [cls_score], [bbox_pred]

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        assert isinstance(cls_score, list)
        assert cls_score[0].shape[0] == labels.shape[0]
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls_1'] = self.loss_cls(
                cls_score[0],
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['loss_cls_2'] = self.loss_cls(
                cls_score[1],
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc_1'] = accuracy(cls_score[0], labels)
            losses['acc_2'] = accuracy(cls_score[1], labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            pos_bbox_pred = []
            for b_p in bbox_pred:
                if self.reg_class_agnostic:
                    pos_bbox_pred.append(b_p.view(b_p.size(0), 4)[pos_inds])
                else:
                    pos_bbox_pred.append(b_p.view(b_p.size(0), -1,
                                                4)[pos_inds, labels[pos_inds]])
            losses['loss_bbox_1'] = self.loss_bbox(
                pos_bbox_pred[0],
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
            losses['loss_bbox_2'] = self.loss_bbox(
                pos_bbox_pred[1],
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        return losses

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_det_bboxes(self,
                       rois,
                       cls_scores,
                       bbox_preds,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        bboxes_collect = []
        scores_collect = []
        for cls_score,bbox_pred in zip(cls_scores,bbox_preds):
            if isinstance(cls_score, list):
                cls_score = sum(cls_score) / float(len(cls_score))
            scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

            if bbox_pred is not None:
                bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                    self.target_stds, img_shape)
            else:
                bboxes = rois[:, 1:].clone()
                if img_shape is not None:
                    bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                    bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

            if rescale:
                if isinstance(scale_factor, float):
                    bboxes /= scale_factor
                else:
                    scale_factor = torch.from_numpy(scale_factor).to(bboxes.device)
                    bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                            scale_factor).view(bboxes.size()[0], -1)

            if cfg is None:
                bboxes_collect.append(bboxes)
                scores_collect.append(scores)
            else:
                det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
                bboxes_collect.append(det_bboxes)
                scores_collect.append(det_labels)

        return bboxes_collect, scores_collect