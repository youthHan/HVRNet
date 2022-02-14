import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from pytorch_metric_learning.losses import TripletNonLocalLoss
from mmdet.core import (bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)

from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head import BBoxHead
from ..losses import accuracy

def masked_softmax(
    vector: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    memory_efficient: bool = False,
    mask_fill_value: float = -1e32,
) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result

@HEADS.register_module
class HRNMPBBoxHead(BBoxHead):
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
                 conv_z=[True,True,True,True,True,True,True,True],
                 conv_g=[False,False,False,False,False,False,False,False],
                 *args,
                 **kwargs):
        super(HRNMPBBoxHead, self).__init__(*args, **kwargs)
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

        # self.selsa_1, self.selsa_2, self.selsa_3, \
        # self.selsa_4, self.selsa_5, self.selsa_6, \
        # self.selsa_7, self.selsa_8 = \
        #         self._add_selsa_with_fc(self.feat_dim, self.fc_feat_dim, 
        #                                 self.dim, self.conv_z, self.conv_g)

        self.selsa_1, self.selsa_2, self.selsa_3, \
        self.selsa_4, self.selsa_5, self.selsa_6 = \
                  self._add_selsa_with_fc(self.feat_dim, self.fc_feat_dim, 
                                        self.dim, self.conv_z, self.conv_g)

        # self.selsa_1, self.selsa_2, self.selsa_3, \
        # self.selsa_4  = \
        #           self._add_selsa_with_fc(self.feat_dim, self.fc_feat_dim, 
        #                                 self.dim, self.conv_z, self.conv_g)

        # self.selsa_1, self.selsa_2, self.selsa_3, \
        # self.selsa_4, self.selsa_5  = \
        #           self._add_selsa_with_fc(self.feat_dim, self.fc_feat_dim, 
        #                                 self.dim, self.conv_z, self.conv_g)

        # self.selsa_1, self.selsa_2, self.selsa_3, self.selsa_4 = \
        #         self._add_selsa_with_fc(self.feat_dim, self.fc_feat_dim, 
        #                                 self.dim, self.conv_z, self.conv_g)

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.dim[2], self.num_classes)
            self.fc_cls_2 = nn.Linear(self.dim[2], self.num_classes)
            # self.fc_cls_3 = nn.Linear(self.dim[2], self.num_classes)
            # self.fc_cls_4 = nn.Linear(self.dim[2], self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.dim[2], out_dim_reg)
            self.fc_reg_2 = nn.Linear(self.dim[2], out_dim_reg)
            # self.fc_reg_3 = nn.Linear(self.dim[2], out_dim_reg)
            # self.fc_reg_4 = nn.Linear(self.dim[2], out_dim_reg)

    def _add_selsa_with_fc(self, 
                          feat_dim,
                          fc_feat_dim,
                          dim=(1024,1024),
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
        if conv_g[2]:
            selsa_3.update(v_data_fc_3 = nn.Linear(fc_feat_dim, dim[2]))
        if conv_z[2]:
            selsa_3.update(linear_out_3 = nn.Conv2d(dim[2], dim[2], 1))
        selsa_3 = nn.ModuleDict(selsa_3)

        self.fc_new_4 = nn.Linear(dim[2], fc_feat_dim)
        selsa_4 = OrderedDict(
            q_data_fc_4 = nn.Linear(fc_feat_dim, dim[0]),
            k_data_fc_4= nn.Linear(fc_feat_dim, dim[1]),
            aff_softmax_4 = nn.Softmax(dim=2)
        )
        if conv_g[3]:
            selsa_4.update(v_data_fc_4 = nn.Linear(fc_feat_dim, dim[2]))
        if conv_z[3]:
            selsa_4.update(linear_out_4 = nn.Conv2d(dim[2], dim[2], 1))
        selsa_4 = nn.ModuleDict(selsa_4)

        self.fc_new_5 = nn.Linear(dim[2], fc_feat_dim)
        selsa_5 = OrderedDict(
            q_data_fc_5 = nn.Linear(fc_feat_dim, dim[0]),
            k_data_fc_5= nn.Linear(fc_feat_dim, dim[1]),
            aff_softmax_5 = nn.Softmax(dim=2)
        )
        if conv_g[3]:
            selsa_5.update(v_data_fc_5 = nn.Linear(fc_feat_dim, dim[2]))
        if conv_z[3]:
            selsa_5.update(linear_out_5 = nn.Conv2d(dim[2], dim[2], 1))
        selsa_5 = nn.ModuleDict(selsa_5)
        
        self.fc_new_6 = nn.Linear(dim[2], fc_feat_dim)
        selsa_6 = OrderedDict(
            q_data_fc_6 = nn.Linear(fc_feat_dim, dim[0]),
            k_data_fc_6= nn.Linear(fc_feat_dim, dim[1]),
            aff_softmax_6 = nn.Softmax(dim=2)
        )
        if conv_g[3]:
            selsa_6.update(v_data_fc_6 = nn.Linear(fc_feat_dim, dim[2]))
        if conv_z[3]:
            selsa_6.update(linear_out_6 = nn.Conv2d(dim[2], dim[2], 1))
        selsa_6 = nn.ModuleDict(selsa_6)

        # self.fc_new_7 = nn.Linear(dim[2], fc_feat_dim)
        # selsa_7 = OrderedDict(
        #     q_data_fc_7 = nn.Linear(fc_feat_dim, dim[0]),
        #     k_data_fc_7= nn.Linear(fc_feat_dim, dim[1]),
        #     aff_softmax_7 = nn.Softmax(dim=2)
        # )
        # if conv_g[3]:
        #     selsa_7.update(v_data_fc_7 = nn.Linear(fc_feat_dim, dim[2]))
        # if conv_z[3]:
        #     selsa_7.update(linear_out_7 = nn.Conv2d(dim[2], dim[2], 1))
        # selsa_7 = nn.ModuleDict(selsa_7)

        # self.fc_new_8 = nn.Linear(dim[2], fc_feat_dim)
        # selsa_8 = OrderedDict(
        #     q_data_fc_8 = nn.Linear(fc_feat_dim, dim[0]),
        #     k_data_fc_8= nn.Linear(fc_feat_dim, dim[1]),
        #     aff_softmax_8 = nn.Softmax(dim=2)
        # )
        # if conv_g[3]:
        #     selsa_8.update(v_data_fc_8 = nn.Linear(fc_feat_dim, dim[2]))
        # if conv_z[3]:
        #     selsa_8.update(linear_out_8 = nn.Conv2d(dim[2], dim[2], 1))
        # selsa_8 = nn.ModuleDict(selsa_8)

        return selsa_1, selsa_2, selsa_3, selsa_4, selsa_5, selsa_6#, selsa_7, selsa_8

    #TODO: Rewrite this function
    def init_weights(self):
        super(HRNMPBBoxHead, self).init_weights()
        # print("Param Init in SelsaBBoxhead for fc has changed to uniform, rather than xavier_uniform")
        for module_list in [self.fc_new_1, self.selsa_1,
                            self.fc_new_2, self.selsa_2,
                            self.fc_new_3, self.selsa_3,
                            self.fc_new_4, self.selsa_4,
                            self.fc_new_5, self.selsa_5,
                            self.fc_new_6, self.selsa_6,
                            # self.fc_new_7, self.selsa_7,
                            # self.fc_new_8, self.selsa_8
                            ]:
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
            # nn.init.normal_(self.fc_cls_3.weight, 0, 0.01)
            # nn.init.constant_(self.fc_cls_3.bias, 0)
            # nn.init.normal_(self.fc_cls_4.weight, 0, 0.01)
            # nn.init.constant_(self.fc_cls_4.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.01)
            nn.init.constant_(self.fc_reg.bias, 0)
            nn.init.normal_(self.fc_reg_2.weight, 0, 0.01)
            nn.init.constant_(self.fc_reg_2.bias, 0)
            # nn.init.normal_(self.fc_reg_3.weight, 0, 0.01)
            # nn.init.constant_(self.fc_reg_3.bias, 0)
            # nn.init.normal_(self.fc_reg_4.weight, 0, 0.01)
            # nn.init.constant_(self.fc_reg_4.bias, 0)

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
        # assert not non_cur_space and not idx_output_cur_only, "Under HRNMPBBoxHead, these two options haven't been carefully reviewed"
        assert not non_cur_space, "Under HRNMPBBoxHead, `non_cur_space` hasn't been carefully reviewed"
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
        assert nongt_dim >= roi_feat.shape[0]
        nongt_roi_feat = roi_feat[0:nongt_dim]

        # TODO: Fix this function under HRNMPBBoxHead
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
        # ! Bug exists in above code: pos_sm and pos_nsm are in wrong (inversed) positions
        pos_indices = torch.gather(inds_for_pos_sm.squeeze(), 0, ind_cls_not_bg)
        neg_indices = torch.gather(inds_for_pos_nsm.squeeze(), 0, ind_cls_not_bg)
        return [ind_cls_not_bg, neg_indices, pos_indices]


    def forward_single_selsa_with_mining(self, 
                                        roi_feat, 
                                        key_dim,
                                        nongt_dim, 
                                        index, 
                                        cur_range_s=None,
                                        non_cur_space=False, 
                                        idx_output_cur_only=False,
                                        labels=None,
                                        inds_chosen=None):
        '''
        args:
        '''
        # assert not non_cur_space and not idx_output_cur_only, "Under HRNMPBBoxHead, these two options haven't been carefully reviewed"
        assert not non_cur_space and not idx_output_cur_only, "Under HRNMPBBoxHead, `non_cur_space` hasn't been carefully reviewed"
        assert labels is not None, "labels should be specified when mining within proposals"
        # assert inds_chosen is not None, "inds_chosen should be specified when perform third non-local module"
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

        # TODO: Fix this function under HRNMPBBoxHead
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

        with torch.no_grad():
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
            # aff_scale.permute(1, 0, 2)

            # for frame_range in cur_range_s:
            #     feat_strt_dim = frame_range['start']
            #     feat_len = frame_range['length']

            #     # Get frame mask
            #     # frame_aff = aff_softmax_copy[feat_strt_dim:feat_strt_dim+feat_len, :]
            #     # mask = frame_aff.new_ones(frame_aff.shape)
            #     # mask[:, feat_strt_dim:feat_strt_dim+feat_len] = 0
            #     # frame_aff = frame_aff.mul(mask)
            #     # del mask

            # Get class mask
            # inds = aff_softmax_reshape.new_zeros(aff_scale.shape)
            mask = torch.ones([1, aff_scale.shape[-1]], device=aff_scale.device)
            # mask[...] = float('inf')
            # m_proposal_aff = []
            inds_chosen = []
            for proposal_ind in range(nongt_roi_feat.shape[0]):
                cls_map = dict()
                cls_id = labels[proposal_ind]
                cls_map.setdefault(cls_id, [torch.where(labels==cls_id), torch.where(labels!=cls_id)])
                proposal_aff = aff_scale[0][proposal_ind:proposal_ind+1, :]
                inds = []
                if cls_id != 0:
                    # Get cur_class mask and affinity vector of cls_id
                    # mask = proposal_aff.new_ones(proposal_aff.shape)
                    mask[...] = 1
                    c_frame_aff = proposal_aff.mul(mask)
                    # ! check the shape and the value of mask and c_frame_aff
                    c_frame_aff[0][cls_map[cls_id][1]] = float('inf')
                    c_proposal_ind = torch.argmin(c_frame_aff)
                    
                    # Get not_cur_class mask and affinity vector 
                    mask[...] = 0
                    mask[0][cls_map[cls_id][1]] = 1
                    nc_frame_aff = proposal_aff.mul(mask)
                    nc_frame_aff[torch.where(mask==0)] = float('-inf')
                    nc_proposal_ind = torch.argmax(nc_frame_aff)
                    
                    assert c_proposal_ind != nc_proposal_ind

                    # # mask = proposal_aff.new_ones(proposal_aff.shape)
                    # s_mask = proposal_aff.new_ones(proposal_aff.shape)
                    # s_mask = s_mask.type(torch.bool)
                    # s_mask[0][c_proposal_ind] = 0
                    # s_mask[0][nc_proposal_ind] = 0
                    # # ! check if the mask could be well-broadcastable with `proposal_aff`
                    # # m_proposal_aff.append(proposal_aff.masked_fill(s_mask, float('-inf'))[...][None])
                    # proposal_aff = proposal_aff.masked_fill_(s_mask.new_ones(s_mask.shape), 1.0)
                    # proposal_aff = proposal_aff.masked_fill_(s_mask, float('-inf'))
                    inds.extend([c_proposal_ind, nc_proposal_ind])
                else:
                    # Get not_negative mask and affinity vector
                    s_mask = proposal_aff.new_ones(proposal_aff.shape)
                    s_mask = s_mask.type(torch.bool)
                    # mask = proposal_aff.new_ones(proposal_aff.shape)
                    mask[...] = 0
                    mask[0][cls_map[cls_id][1]] = 1
                    nc_frame_aff = proposal_aff.mul(mask)
                    nc_frame_aff[torch.where(mask==0)] = float('-inf')
                    nc_proposal_ind = torch.argmax(nc_frame_aff)
                    nc_frame_aff[0][nc_proposal_ind] = float('-inf')
                    # s_mask[0][nc_proposal_ind] = 0
                    inds.append(nc_proposal_ind)
                    nc_proposal_ind = torch.argmax(nc_frame_aff)
                    # s_mask[0][nc_proposal_ind] = 0
                    inds.append(nc_proposal_ind)
                    if inds[0] == inds[1]:
                        print('debug')
                    # # ! check if the mask could be well-broadcastable with `proposal_aff`
                    # # m_proposal_aff.append(proposal_aff.masked_fill(s_mask, float('-inf'))[...][None])
                    # proposal_aff = proposal_aff.masked_fill_(s_mask.new_ones(s_mask.shape), 1.0)
                    # proposal_aff = proposal_aff.masked_fill_(s_mask, float('-inf'))
                inds_chosen.append(inds)
            # aff_softmax_reshape = inds / 2.
        
        output_t_list = []
        for proposal_ind, inds in enumerate(inds_chosen):
            feat_q_in = roi_feat[proposal_ind:proposal_ind+1]
            feat_k_in = torch.cat([roi_feat[inds[0]][None], roi_feat[inds[1]][None]])
            q_data = getattr(self, 'selsa_%d'%index)['q_data_fc_%d'%index](feat_q_in)
            q_data_batch = torch.reshape(q_data, shape=(-1,1,dim[0]))
            q_data_batch = q_data_batch.permute(1, 0, 2)
            k_data = getattr(self, 'selsa_%d'%index)['k_data_fc_%d'%index](feat_k_in)
            k_data_batch = torch.reshape(k_data, shape=(-1,1,dim[1])) 
            k_data_batch = k_data_batch.permute(1, 2, 0)   
            v_data = feat_k_in
            #!!Caution: the shape of the batch dot result
            # print(q_data.shape)
            # print(v_data.shape)
            aff = torch.bmm(q_data_batch, k_data_batch)
            aff_scale = (1.0 / math.sqrt(float(dim[1]))) * aff
            # weighted_aff = torch.cat(m_proposal_aff, dim=0)
            weighted_aff = aff_scale
            aff_softmax = getattr(self, 'selsa_%d'%index)['aff_softmax_%d'%index](weighted_aff)
            #!!Caution: the copy of the tensor, may cause problem 
            # similarity = torch.empty(aff_softmax.shape, 
            #                         dtype=aff_softmax.dtype, 
            #                         device=aff_softmax.device).copy_(aff_softmax)
            
            # aff_softmax_reshape = aff_softmax.view(-1, aff_softmax.shape[2])
            aff_softmax_reshape = aff_softmax.view(k_data_batch.shape[0], -1)
            # aff_softmax_copy = torch.empty(aff_softmax_reshape.shape, 
            #                         dtype=aff_softmax_reshape.dtype, 
            #                         device=aff_softmax_reshape.device).copy_(aff_softmax_reshape)
            if conv_g[index-1]:
                v_data = getattr(self, 'selsa_%d'%index)['v_data_fc_%d'%index](v_data)
            output_t = torch.mm(aff_softmax_reshape, v_data)
            output_t = torch.reshape(output_t, shape=(-1, self.fc_feat_dim, 1, 1))
            output_t_list.append(output_t)

        output_t = torch.cat(output_t_list, dim=0)
        if conv_z[index-1]:
            linear_out = getattr(self, 'selsa_%d'%index)['linear_out_%d'%index](output_t)
        else:
            linear_out = output_t
        
        output = torch.reshape(linear_out, shape=linear_out.shape[:2])

        return output, None

    def forward_single_selsa_with_mining_2(self, 
                                        roi_feat, 
                                        key_dim,
                                        nongt_dim, 
                                        index, 
                                        cur_range_s=None,
                                        non_cur_space=False, 
                                        idx_output_cur_only=False,
                                        labels=None,
                                        all_labels=None,
                                        inds_chosen=None,
                                        test=False,
                                        metric_loss=None):
        """Proposal level mining - with common k-data and q-data feature embedding
        """
        # assert not non_cur_space and not idx_output_cur_only, "Under HRNMPBBoxHead, these two options haven't been carefully reviewed"
        assert not non_cur_space, "Under HRNMPBBoxHead, `non_cur_space` hasn't been carefully reviewed"
        if not test:
            assert labels is not None, "labels should be specified when mining within proposals"
        # assert inds_chosen is not None, "inds_chosen should be specified when perform third non-local module"
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

        # TODO: Fix this function under HRNMPBBoxHead
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
        # aff_scale.permute(1, 0, 2)

        # for frame_range in cur_range_s:
        #     feat_strt_dim = frame_range['start']
        #     feat_len = frame_range['length']

        #     # Get frame mask
        #     # frame_aff = aff_softmax_copy[feat_strt_dim:feat_strt_dim+feat_len, :]
        #     # mask = frame_aff.new_ones(frame_aff.shape)
        #     # mask[:, feat_strt_dim:feat_strt_dim+feat_len] = 0
        #     # frame_aff = frame_aff.mul(mask)
        #     # del mask

        # Get class mask
        # inds = aff_softmax_reshape.new_zeros(aff_scale.shape)
        mask = torch.ones([1, aff_scale.shape[-1]], device=aff_scale.device)
        # mask[...] = float('inf')
        # m_proposal_aff = []
        inds_chosen = []
        cls_map = dict()
        if not test:
            for proposal_ind in range(roi_feat.shape[0]):
                cls_id = labels[proposal_ind]
                cls_map.setdefault(cls_id, [torch.where(all_labels==cls_id), torch.where(all_labels!=cls_id)])
                proposal_aff = aff_scale[0][proposal_ind:proposal_ind+1, :]
                inds = torch.ones([2], dtype=torch.long, device=aff_scale.device)
                if cls_id != 0:
                    # Get cur_class mask and affinity vector of cls_id
                    # mask = proposal_aff.new_ones(proposal_aff.shape)
                    mask[...] = 1
                    c_frame_aff = proposal_aff.mul(mask)
                    # ! check the shape and the value of mask and c_frame_aff
                    c_frame_aff[0][cls_map[cls_id][1]] = float('inf')
                    c_proposal_ind = torch.argmin(c_frame_aff)
                    
                    # Get not_cur_class mask and affinity vector 
                    mask[...] = 0
                    mask[0][cls_map[cls_id][1]] = 1
                    nc_frame_aff = proposal_aff.mul(mask)
                    nc_frame_aff[torch.where(mask==0)] = float('-inf')
                    nc_proposal_ind = torch.argmax(nc_frame_aff)
                    
                    assert c_proposal_ind != nc_proposal_ind

                    # # mask = proposal_aff.new_ones(proposal_aff.shape)
                    # s_mask = proposal_aff.new_ones(proposal_aff.shape)
                    # s_mask = s_mask.type(torch.bool)
                    # s_mask[0][c_proposal_ind] = 0
                    # s_mask[0][nc_proposal_ind] = 0
                    # # ! check if the mask could be well-broadcastable with `proposal_aff`
                    # # m_proposal_aff.append(proposal_aff.masked_fill(s_mask, float('-inf'))[...][None])
                    # proposal_aff = proposal_aff.masked_fill_(s_mask.new_ones(s_mask.shape), 1.0)
                    # proposal_aff = proposal_aff.masked_fill_(s_mask, float('-inf'))
                    inds[0] = c_proposal_ind
                    inds[1] = nc_proposal_ind
                else:
                    # Get not_negative mask and affinity vector
                    # s_mask = proposal_aff.new_ones(proposal_aff.shape)
                    # s_mask = s_mask.type(torch.bool)
                    # mask = proposal_aff.new_ones(proposal_aff.shape)
                    mask[...] = 0
                    mask[0][cls_map[cls_id][1]] = 1
                    nc_frame_aff = proposal_aff.mul(mask)
                    nc_frame_aff[torch.where(mask==0)] = float('-inf')
                    nc_proposal_ind = torch.argmax(nc_frame_aff)
                    nc_frame_aff[0][nc_proposal_ind] = float('-inf')
                    # s_mask[0][nc_proposal_ind] = 0
                    inds[0] = nc_proposal_ind
                    nc_proposal_ind = torch.argmax(nc_frame_aff)
                    # s_mask[0][nc_proposal_ind] = 0
                    inds[1] = nc_proposal_ind
                    assert inds[0] != inds[1]
                    # # ! check if the mask could be well-broadcastable with `proposal_aff`
                    # # m_proposal_aff.append(proposal_aff.masked_fill(s_mask, float('-inf'))[...][None])
                    # proposal_aff = proposal_aff.masked_fill_(s_mask.new_ones(s_mask.shape), 1.0)
                    # proposal_aff = proposal_aff.masked_fill_(s_mask, float('-inf'))
                inds_chosen.append(inds)
        else:
            inds_chosen = torch.split(aff_scale[0].topk(2).indices.to(torch.long), 1, dim=0)
        # aff_softmax_reshape = inds / 2.
        # ! the function below is not completed
        if metric_loss is not None:
            q_data_metric = q_data_batch.squeeze()
            k_data_metric = k_data_batch.permute(0, 2, 1).squeeze()
            not_bg_inds = torch.where(labels!=0)
            pos_indices = torch.cat(inds_chosen, dim=0)[not_bg_inds][:,0]
            neg_indices = torch.cat(inds_chosen, dim=0)[not_bg_inds][:,1]
            m_loss = metric_loss(q_data_metric, k_data_metric, labels, 
                                    [not_bg_inds, pos_indices, neg_indices])
        
        output_t_list = []
        for proposal_ind, inds in enumerate(inds_chosen):
            inds.squeeze_()
            q_data_batch_sep = q_data_batch[:, proposal_ind:proposal_ind+1, :]
            k_data_batch_sep = torch.cat([k_data_batch[:,:,inds[0]:inds[0]+1], k_data_batch[:,:,inds[1]:inds[1]+1]], dim=2)
            v_data = torch.cat([nongt_roi_feat[inds[0]:inds[0]+1,:], nongt_roi_feat[inds[1]:inds[1]+1,:]], dim=0)
            #!!Caution: the shape of the batch dot result
            # print(q_data.shape)
            # print(v_data.shape)
            aff = torch.bmm(q_data_batch_sep, k_data_batch_sep)
            aff_scale = (1.0 / math.sqrt(float(dim[1]))) * aff
            # weighted_aff = torch.cat(m_proposal_aff, dim=0)
            weighted_aff = aff_scale
            aff_softmax = getattr(self, 'selsa_%d'%index)['aff_softmax_%d'%index](weighted_aff)
            #!!Caution: the copy of the tensor, may cause problem 
            # similarity = torch.empty(aff_softmax.shape, 
            #                         dtype=aff_softmax.dtype, 
            #                         device=aff_softmax.device).copy_(aff_softmax)
            
            # aff_softmax_reshape = aff_softmax.view(-1, aff_softmax.shape[2])
            aff_softmax_reshape = aff_softmax.view(k_data_batch_sep.shape[0], -1)
            # aff_softmax_copy = torch.empty(aff_softmax_reshape.shape, 
            #                         dtype=aff_softmax_reshape.dtype, 
            #                         device=aff_softmax_reshape.device).copy_(aff_softmax_reshape)
            if conv_g[index-1]:
                v_data = getattr(self, 'selsa_%d'%index)['v_data_fc_%d'%index](v_data)
            output_t = torch.mm(aff_softmax_reshape, v_data)
            output_t = torch.reshape(output_t, shape=(-1, self.fc_feat_dim, 1, 1))
            output_t_list.append(output_t)

        output_t = torch.cat(output_t_list, dim=0)
        if conv_z[index-1]:
            linear_out = getattr(self, 'selsa_%d'%index)['linear_out_%d'%index](output_t)
        else:
            linear_out = output_t
        
        output = torch.reshape(linear_out, shape=linear_out.shape[:2])

        # return output, None#, inds_chosen
        if metric_loss is not None:
            return output, m_loss, None
        else:
            return outpupt, None
    
    
    def forward_single_selsa_with_mining_inplace(self, 
                                        roi_feat, 
                                        key_dim,
                                        nongt_dim, 
                                        index,
                                        cur_range_s=None,
                                        non_cur_space=False,
                                        idx_output_cur_only=False,
                                        labels=None,
                                        all_labels=None,
                                        inds_chosen=None,
                                        test=False,
                                        k=2,
                                        metric_loss=None):
        """Proposal level mining - with common k-data and q-data feature embedding
        """
        # assert not non_cur_space and not idx_output_cur_only, "Under HRNMPBBoxHead, these two options haven't been carefully reviewed"
        assert not non_cur_space, "Under HRNMPBBoxHead, `non_cur_space` hasn't been carefully reviewed"
        if not test:
            assert labels is not None, "labels should be specified when mining within proposals"
        # assert inds_chosen is not None, "inds_chosen should be specified when perform third non-local module"
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
        assert nongt_dim >= roi_feat.shape[0]
        nongt_roi_feat = roi_feat[0:nongt_dim]

        # TODO: Fix this function under HRNMPBBoxHead
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
        # aff_scale.permute(1, 0, 2)

        # for frame_range in cur_range_s:
        #     feat_strt_dim = frame_range['start']
        #     feat_len = frame_range['length']

        #     # Get frame mask
        #     # frame_aff = aff_softmax_copy[feat_strt_dim:feat_strt_dim+feat_len, :]
        #     # mask = frame_aff.new_ones(frame_aff.shape)
        #     # mask[:, feat_strt_dim:feat_strt_dim+feat_len] = 0
        #     # frame_aff = frame_aff.mul(mask)
        #     # del mask

        # Get class mask
        if not test:
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

            all_inds_chosen = inds_for_bg.clone().detach()
            all_inds_chosen[ind_cls_not_bg]=torch.cat([inds_for_pos_sm,inds_for_pos_nsm], dim=1)[ind_cls_not_bg]

            if metric_loss is not None:
                q_data_metric = q_data_batch.squeeze()
                k_data_metric = k_data_batch.permute(0, 2, 1).squeeze()
                not_bg_inds = torch.where(labels!=0)
                pos_indices = torch.gather(inds_for_pos_sm.squeeze(), 0, ind_cls_not_bg)
                neg_indices = torch.gather(inds_for_pos_nsm.squeeze(), 0, ind_cls_not_bg)
                # ! inplace metric loss mining still uses the inversed `pos_indices` and `neg_indices`
                m_loss = metric_loss.compute_loss(q_data_metric, k_data_metric, labels, 
                                        [ind_cls_not_bg, pos_indices, neg_indices])
        else:
            inds_collect = aff_scale.new_zeros(aff_scale.shape).squeeze_()
            inds_chosen = aff_scale.topk(k).indices
            inds_collect.scatter_(1, inds_chosen.squeeze_(dim=0), inds_chosen.new_ones(inds_chosen.shape).to(torch.float))
        aff_scale.masked_fill(inds_collect.to(torch.bool).logical_not(), float('-inf'))
        
        weighted_aff = aff_scale
        aff_softmax = getattr(self, 'selsa_%d'%index)['aff_softmax_%d'%index](weighted_aff)
        #!!Caution: the copy of the tensor, may cause problem 
        # similarity = torch.empty(aff_softmax.shape, 
        #                         dtype=aff_softmax.dtype, 
        #                         device=aff_softmax.device).copy_(aff_softmax)
        similarity = aff_softmax.clone().detach()

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

        # return output, None, [inds_for_bg, inds_for_pos_sm, inds_for_pos_nsm]  
        if metric_loss is not None:
            # return output, m_loss, dict(similarity=dict(aff=similarity.cpu().numpy(), 
            #                                             labels=labels.cpu().numpy(), 
            #                                             range=cur_range_s,
            #                                             q_fc=q_data.cpu().numpy(),
            #                                             k_fc=k_data.cpu().numpy(),
            #                                             chosen_proposal_inds=all_inds_chosen.cpu().numpy()))
            return output, m_loss, None
        else:
            # return output, dict(similarity=dict(aff=similarity.cpu().numpy(), 
            #                                             labels=labels.cpu().numpy(), 
            #                                             range=cur_range_s,
            #                                             q_fc=q_data.cpu().numpy(),
            #                                             k_fc=k_data.cpu().numpy(),
            #                                             chosen_proposal_inds=all_inds_chosen.cpu().numpy()))
            return output, None

    #TODO: Rewrite this function, most of the functionality needs to be implemented here
    def forward_v1(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False, others=None, dynamic=False, all_labels=None, post_sampler=None, bbox_targets_key=None):
        r"""v1
        Main Selsa Functionality implementation
                    __________________________________
                    |                            loss |    
                    |                           /     |    
        video_a  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3
                    |_________________________________|         \
                    __________________________________           \
                    |                          loss   |           \                          loss   trip_loss
                    |                           /     |            \                        /       /
        video_b  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3     |-> fc_4 -> NL_4 [key frames hardest proposal 
                    |_________________________________|            /                        mining and aggregation]
                    __________________________________            /
                    |                          loss   |          /
                    |                           /     |         /
        video_c  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3
                    |_________________________________|

        Args:
            bbox_feat: Conved style features in shape [roi_nums, num_channel, roi_feat, roi_feat]

        Returns:

        """
        assert post_sampler is None, "Post sampler not implemented here"
        assert cur_range_s is not None, "Feature num range along axis needs specified, \
                                        as length of features for each frame could be different"
        if dynamic and all_labels is None:
            raise AssertionError("`all_labels` should be specified when `dynamic` is `True`")
        
        self.key_dim = key_dim
        self.nongt_dim = self.sampler_num * self.t_dim
        
        loss_additional = dict()
        num_videos = len(cur_range_s)
        video_feats = []
        bbox_num_per_video = []
        cls_res_branches = []
        reg_res_branches = []
        imgs_per_video = self.imgs_per_video
        sampler_num = self.sampler_num
        for i, bbox_feat in enumerate(bbox_feat_s):
            # ! ##################
            # if i != 0:
            #     break
            cur_range = cur_range_s[i]
            feat_strt_dim = cur_range['start']
            feat_len = cur_range['length']
            # print('enter aggregation module')
            bbox_feat = bbox_feat.view(bbox_feat.size(0), -1)
            fc_new_feat_1 = self.fc_new_1(bbox_feat)
            attention_1, _ = self.forward_single_selsa(fc_new_feat_1, self.key_dim, 
                                    imgs_per_video*sampler_num, index=1, idx_output_cur_only=False, cur_range_s=[cur_range])
            fc_all_1 = fc_new_feat_1 + attention_1
            # del fc_new_feat_1
            del attention_1
            fc_all_1_relu = self.relu(fc_all_1)
            del fc_all_1
        
            # ! start of second non-local module
            fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_2, m_loss, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_2, self.key_dim,  
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range], 
                                        labels=cur_labels, all_labels=all_labels[i], metric_loss=loss_metric)
                loss_additional.setdefault('loss_trip_video', None)
                if loss_additional['loss_trip_video'] is None:
                    loss_additional.update(dict(loss_trip_video=m_loss))
                else:
                    loss_additional['loss_trip_video'] += m_loss
            else:
                attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range])
            # if self.output_cur_only:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
            fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
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

            fc_all_2_branch_relu = fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len]
            cls_res_branches.append(self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None) 
            reg_res_branches.append(self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None)
            del fc_all_2_branch_relu

            # ! start of third non-local module
            fc_3_in = [fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len],
                       fc_new_feat_1[feat_strt_dim+feat_len:]]
            fc_3_in = torch.cat(fc_3_in, dim=0)
            fc_new_feat_3 = self.fc_new_3(fc_3_in)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_3, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_3, self.key_dim,  
                                        imgs_per_video*sampler_num, index=3, idx_output_cur_only=True, cur_range_s=[cur_range], 
                                        labels=cur_labels, all_labels=all_labels[i], metric_loss=None)
            else:
                attention_3, _ = self.forward_single_selsa(fc_new_feat_3, self.key_dim, 
                                        imgs_per_video*sampler_num, index=3, idx_output_cur_only=True, cur_range_s=[cur_range])
            # if self.output_cur_only:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
            fc_new_feat_3 = fc_new_feat_3[feat_strt_dim:feat_strt_dim+feat_len]
            fc_all_3 = fc_new_feat_3 + attention_3
            del fc_new_feat_3
            del attention_3

            # ############
            # if not self.output_cur_only:
            #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            #     # fc_all_3 = torch.index_select(fc_all_3, dim=0, index=feat_select_tensor)
            #     fc_all_3 = fc_all_3[feat_strt_dim:feat_strt_dim+feat_len]
            # ############
            fc_all_3_relu = self.relu(fc_all_3)
            del fc_all_3
            bbox_num_per_video.append(fc_all_3_relu.size(0))
            video_feats.append(fc_all_3_relu)

        cur_only_for_4 = False
        loss_metric = TripletNonLocalLoss(margin=10)
        video_feats = torch.cat(video_feats, dim=0)
        fc_new_feat_4 = self.fc_new_4(video_feats)
        target_ranges_4 = []
        for i, cur_range in enumerate(cur_range_s):
            feat_strt_dim = cur_range['start'] + sum(bbox_num_per_video[:i])
            feat_len = cur_range['length']
            target_ranges_4.append(dict(start=feat_strt_dim, length=feat_len))
        # TODO: Forward selsa with mining needs considering whether mining should be involved in or not
        attention_4, m_loss, similarity_ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_4, None, self.nongt_dim, 
                                        index=4, idx_output_cur_only=cur_only_for_4, cur_range_s=target_ranges_4, 
                                        labels=others, all_labels=others, metric_loss=loss_metric)
        loss_additional.update(dict(loss_trip=m_loss))
        # attention_3, _, inds2 = self.forward_single_selsa_with_mining_2(fc_new_feat_3, None, 
        #                           self.nongt_dim, index=3, idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3, labels=others, all_labels=others)
        if cur_only_for_4:
            # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            feats_key_frames = []
            for i, cur_range in enumerate(target_ranges_4):
                feat_strt_dim = cur_range['start']
                feat_len = cur_range['length']
                feats_key_frames.append(fc_new_feat_4[feat_strt_dim : feat_strt_dim+feat_len])
            del fc_new_feat_4
            fc_new_feat_4 = torch.cat(feats_key_frames, dim=0)

        fc_all_4 = fc_new_feat_4 + attention_4
        del fc_new_feat_4
        del attention_4

        ############
        if not cur_only_for_4:
            # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            feats_key_frames = []
            for i, cur_range in enumerate(target_ranges_4):
                feat_strt_dim = cur_range['start']
                feat_len = cur_range['length']
                feats_key_frames.append(fc_all_4[feat_strt_dim : feat_strt_dim+feat_len])
            del fc_all_4
            fc_all_4 = torch.cat(feats_key_frames, dim=0)

        ############
        fc_all_4_relu = self.relu(fc_all_4)

        if similarity_ is not None:
            similarity_['similarity'].update(feats_no_relu=fc_all_4.cpu().numpy())
        cls_score = self.fc_cls_2(fc_all_4_relu) if self.with_cls else None
        bbox_pred = self.fc_reg_2(fc_all_4_relu) if self.with_reg else None

        cls_score_branch = torch.cat(cls_res_branches, dim=0)
        bbox_pred_branch = torch.cat(reg_res_branches, dim=0)
        assert cls_score_branch.shape[0] == bbox_pred_branch.shape[0]
        assert cls_score_branch.shape[0] == cls_score.shape[0]
        
        return [cls_score_branch, cls_score], [bbox_pred_branch, bbox_pred], loss_additional, similarity_ #!

        # return [cls_score_branch], [bbox_pred_branch], None, None #!
        # return [cls_score], [bbox_pred] #!

    def ohem_(self, post_bbox_sampler, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights):
        with torch.no_grad():
            loss_cls = self.loss(
                    cls_score=[cls_score], 
                    bbox_pred=None, 
                    labels=labels,
                    label_weights=cls_score.new_ones(cls_score.size(0)),
                    bbox_targets=None,
                    bbox_weights=None,
                    reduction_override='none')['loss_cls_1']
            l_weights, b_weights, pos_inds, neg_inds = \
                            post_bbox_sampler.get_ohem_weights(labels,
                                                            label_weights,
                                                            bbox_weights,
                                                            loss_cls)
        all_inds = torch.cat([pos_inds, neg_inds], dim=0)
        # loss_bbox = self.bbox_head.loss(
        #             cls_score=[cls_score[all_inds]], 
        #             bbox_pred=[bbox_pred[all_inds]], 
        #             labels=labels[all_inds],
        #             label_weights=l_weights[all_inds],
        #             bbox_targets=bbox_targets[all_inds],
        #             bbox_weights=b_weights[all_inds],
        #             reduction_override=None)

        # losses['loss_cls_{}'.format(i+1)] = loss_bbox['loss_cls_1']
        # losses['loss_bbox_{}'.format(i+1)] = loss_bbox['loss_bbox_1']
        # losses['acc_{}'.format(i+1)] = loss_bbox['acc_1']

        return all_inds

    #TODO: Rewrite this function, most of the functionality needs to be implemented here
    def forward_v1_ohem(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False, others=None, dynamic=False, all_labels=None, post_sampler=None, bbox_targets_key=None):
        r"""v1-ohem
        Main Selsa Functionality implementation
                    __________________________________
                    |                            loss |    
                    |                           /     |    
        video_a  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3
                    |_________________________________|         \
                    __________________________________           \
                    |                          loss   |           \                          loss   trip_loss
                    |                           /     |            \                        /       /
        video_b  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3     |-> fc_4 -> NL_4 [key frames hardest proposal 
                    |_________________________________|            /                        mining and aggregation]
                    __________________________________            /
                    |                          loss   |          /
                    |                           /     |         /
        video_c  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3
                    |_________________________________|

        Args:
            bbox_feat: Conved style features in shape [roi_nums, num_channel, roi_feat, roi_feat]

        Returns:

        """
        if post_sampler is not None:
            assert bbox_targets_key is not None, "Post sampler exiists, but bbox_target_key is None."
        assert cur_range_s is not None, "Feature num range along axis needs specified, \
                                        as length of features for each frame could be different"
        if dynamic and all_labels is None:
            raise AssertionError("`all_labels` should be specified when `dynamic` is `True`")
        
        self.key_dim = key_dim
        # self.nongt_dim = self.sampler_num * self.t_dim
        
        ohem_inds_r = []
        cur_range_s_ohem_ = []
        loss_additional = dict()
        num_videos = len(cur_range_s)
        video_feats = []
        bbox_num_per_video = []
        cls_res_branches = []
        reg_res_branches = []
        imgs_per_video = self.imgs_per_video
        sampler_num = self.sampler_num
        for i, bbox_feat in enumerate(bbox_feat_s):
            # ! ##################
            # if i != 0:
            #     break
            cur_range = cur_range_s[i]
            feat_strt_dim = cur_range['start']
            feat_len = cur_range['length']
            # print('enter aggregation module')
            bbox_feat = bbox_feat.view(bbox_feat.size(0), -1)
            fc_new_feat_1 = self.fc_new_1(bbox_feat)
            attention_1, _ = self.forward_single_selsa(fc_new_feat_1, self.key_dim, 
                                    imgs_per_video*sampler_num, index=1, idx_output_cur_only=False, cur_range_s=[cur_range])
            fc_all_1 = fc_new_feat_1 + attention_1
            # del fc_new_feat_1
            del attention_1
            fc_all_1_relu = self.relu(fc_all_1)
            del fc_all_1
        
            # ! start of second non-local module
            fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_2, m_loss, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_2, self.key_dim,  
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range], 
                                        labels=cur_labels, all_labels=all_labels[i], metric_loss=loss_metric)
                loss_additional.setdefault('loss_trip_video', None)
                if loss_additional['loss_trip_video'] is None:
                    loss_additional.update(dict(loss_trip_video=m_loss))
                else:
                    loss_additional['loss_trip_video'] += m_loss
            else:
                attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range])
            # if self.output_cur_only:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
            fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
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

            fc_all_2_branch_relu = fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len]
            bbox_targets_key_start_dim = sum(bbox_num_per_video[:i])
            cls_branch_ = self.fc_cls(fc_all_2_branch_relu)
            reg_branch_ = self.fc_reg(fc_all_2_branch_relu)

            ohem_inds_ = self.ohem_(post_sampler, cls_branch_, reg_branch_,
                        labels=bbox_targets_key[0][bbox_targets_key_start_dim:bbox_targets_key_start_dim+feat_len],
                        label_weights=bbox_targets_key[1][bbox_targets_key_start_dim:bbox_targets_key_start_dim+feat_len],
                        bbox_targets=bbox_targets_key[2][bbox_targets_key_start_dim:bbox_targets_key_start_dim+feat_len],
                        bbox_weights=bbox_targets_key[3][bbox_targets_key_start_dim:bbox_targets_key_start_dim+feat_len])

            ohem_inds_r.append(ohem_inds_+bbox_targets_key_start_dim)
            
            cls_res_branches.append(cls_branch_[ohem_inds_] if self.with_cls else None)
            reg_res_branches.append(reg_branch_[ohem_inds_] if self.with_reg else None)
            del fc_all_2_branch_relu

            cur_range_ohem_ = dict(start=feat_strt_dim, length=ohem_inds_.shape[0])
            cur_range_s_ohem_.append(cur_range_ohem_)
            feat_strt_dim_ohem_ = cur_range_ohem_['start']
            feat_len_ohem_ = cur_range_ohem_['length']

            # ! start of third non-local module
            fc_3_in = [fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len][ohem_inds_],
                       fc_new_feat_1[feat_strt_dim+feat_len:]]
            fc_3_in = torch.cat(fc_3_in, dim=0)
            fc_new_feat_3 = self.fc_new_3(fc_3_in)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_3, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_3, self.key_dim,  
                                        fc_3_in.shape[0], index=3, idx_output_cur_only=True, cur_range_s=[cur_range_ohem_], 
                                        labels=cur_labels, all_labels=all_labels[i], metric_loss=None)
            else:
                attention_3, _ = self.forward_single_selsa(fc_new_feat_3, self.key_dim, 
                                        fc_3_in.shape[0], index=3, 
                                        idx_output_cur_only=True, cur_range_s=[cur_range_ohem_])
            # if self.output_cur_only:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim_ohem_:feat_strt_dim_ohem_+feat_len_ohem_]
            fc_new_feat_3 = fc_new_feat_3[feat_strt_dim_ohem_:feat_strt_dim_ohem_+feat_len_ohem_]
            fc_all_3 = fc_new_feat_3 + attention_3
            del fc_new_feat_3
            del attention_3

            # ############
            # if not self.output_cur_only:
            #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            #     # fc_all_3 = torch.index_select(fc_all_3, dim=0, index=feat_select_tensor)
            #     fc_all_3 = fc_all_3[feat_strt_dim_ohem_:feat_strt_dim_ohem_+feat_len_ohem_]
            # ############
            fc_all_3_relu = self.relu(fc_all_3)
            del fc_all_3
            bbox_num_per_video.append(fc_all_3_relu.size(0))
            video_feats.append(fc_all_3_relu)

        ohem_inds_r = torch.cat(ohem_inds_r, dim=0)

        cur_only_for_4 = False
        loss_metric = TripletNonLocalLoss(margin=10)
        video_feats = torch.cat(video_feats, dim=0)
        fc_new_feat_4 = self.fc_new_4(video_feats)
        target_ranges_4 = []
        for i, cur_range_ohem_ in enumerate(cur_range_s_ohem_):
            feat_strt_dim = cur_range_ohem_['start'] + sum(bbox_num_per_video[:i])
            feat_len = cur_range_ohem_['length']
            target_ranges_4.append(dict(start=feat_strt_dim, length=feat_len))
        # TODO: Forward selsa with mining needs considering whether mining should be involved in or not
        attention_4, m_loss, similarity_ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_4, None, fc_new_feat_4.shape[0], 
                                        index=4, idx_output_cur_only=cur_only_for_4, cur_range_s=target_ranges_4, 
                                        labels=others[ohem_inds_r], all_labels=others[ohem_inds_r], metric_loss=loss_metric)
        loss_additional.update(dict(loss_trip=m_loss))
        # attention_3, _, inds2 = self.forward_single_selsa_with_mining_2(fc_new_feat_3, None, 
        #                           self.nongt_dim, index=3, idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3, labels=others, all_labels=others)
        if cur_only_for_4:
            # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            feats_key_frames = []
            for i, cur_range_ohem_ in enumerate(target_ranges_4):
                feat_strt_dim = cur_range_ohem_['start']
                feat_len = cur_range_ohem_['length']
                feats_key_frames.append(fc_new_feat_4[feat_strt_dim : feat_strt_dim+feat_len])
            del fc_new_feat_4
            fc_new_feat_4 = torch.cat(feats_key_frames, dim=0)

        fc_all_4 = fc_new_feat_4 + attention_4
        del fc_new_feat_4
        del attention_4

        ############
        if not cur_only_for_4:
            # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            feats_key_frames = []
            for i, cur_range_ohem_ in enumerate(target_ranges_4):
                feat_strt_dim = cur_range_ohem_['start']
                feat_len = cur_range_ohem_['length']
                feats_key_frames.append(fc_all_4[feat_strt_dim : feat_strt_dim+feat_len])
            del fc_all_4
            fc_all_4 = torch.cat(feats_key_frames, dim=0)

        ############
        fc_all_4_relu = self.relu(fc_all_4)

        if similarity_ is not None:
            similarity_['similarity'].update(feats_no_relu=fc_all_4.cpu().numpy())
        cls_score = self.fc_cls_2(fc_all_4_relu) if self.with_cls else None
        bbox_pred = self.fc_reg_2(fc_all_4_relu) if self.with_reg else None

        cls_score_branch = torch.cat(cls_res_branches, dim=0)
        bbox_pred_branch = torch.cat(reg_res_branches, dim=0)
        assert cls_score_branch.shape[0] == bbox_pred_branch.shape[0]
        assert cls_score_branch.shape[0] == cls_score.shape[0]
        
        return [cls_score_branch, cls_score], [bbox_pred_branch, bbox_pred],  \
                loss_additional, similarity_,  ohem_inds_r #!

        # return [cls_score_branch], [bbox_pred_branch], None, None #!
        # return [cls_score], [bbox_pred] #!


    #TODO: Rewrite this function, most of the functionality needs to be implemented here
    def forward_v1_ohem2(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False, others=None, dynamic=False, all_labels=None, post_sampler=None, bbox_targets_key=None):
        r"""v1-ohem2
        Main Selsa Functionality implementation
                    __________________________________
                    |                            loss |    
                    |                           /     |    
        video_a  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3
                    |_________________________________|         \
                    __________________________________           \
                    |                          loss   |           \                          loss   trip_loss
                    |                           /     |            \                        /       /
        video_b  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3     |-> fc_4 -> NL_4 [key frames hardest proposal 
                    |_________________________________|            /                        mining and aggregation]
                    __________________________________            /
                    |                          loss   |          /
                    |                           /     |         /
        video_c  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3
                    |_________________________________|

        Args:
            bbox_feat: Conved style features in shape [roi_nums, num_channel, roi_feat, roi_feat]

        Returns:

        """
        if post_sampler is not None:
            assert bbox_targets_key is not None, "Post sampler exiists, but bbox_target_key is None."
        assert cur_range_s is not None, "Feature num range along axis needs specified, \
                                        as length of features for each frame could be different"
        if dynamic and all_labels is None:
            raise AssertionError("`all_labels` should be specified when `dynamic` is `True`")
        
        self.key_dim = key_dim
        # self.nongt_dim = self.sampler_num * self.t_dim
        
        ohem_inds_r = []
        cur_range_s_ohem_ = []
        loss_additional = dict()
        num_videos = len(cur_range_s)
        video_feats = []
        bbox_num_per_video = []
        cls_res_branches = []
        reg_res_branches = []
        imgs_per_video = self.imgs_per_video
        sampler_num = self.sampler_num
        for i, bbox_feat in enumerate(bbox_feat_s):
            # ! ##################
            # if i != 0:
            #     break
            cur_range = cur_range_s[i]
            feat_strt_dim = cur_range['start']
            feat_len = cur_range['length']
            # print('enter aggregation module')
            bbox_feat = bbox_feat.view(bbox_feat.size(0), -1)
            fc_new_feat_1 = self.fc_new_1(bbox_feat)
            attention_1, _ = self.forward_single_selsa(fc_new_feat_1, self.key_dim, 
                                    imgs_per_video*sampler_num, index=1, idx_output_cur_only=False, cur_range_s=[cur_range])
            fc_all_1 = fc_new_feat_1 + attention_1
            # del fc_new_feat_1
            del attention_1
            fc_all_1_relu = self.relu(fc_all_1)
            del fc_all_1
        
            # ! start of second non-local module
            fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_2, m_loss, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_2, self.key_dim,  
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range], 
                                        labels=cur_labels, all_labels=all_labels[i], metric_loss=loss_metric)
                loss_additional.setdefault('loss_trip_video', None)
                if loss_additional['loss_trip_video'] is None:
                    loss_additional.update(dict(loss_trip_video=m_loss))
                else:
                    loss_additional['loss_trip_video'] += m_loss
            else:
                attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range])
            # if self.output_cur_only:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
            fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
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

            fc_all_2_branch_relu = fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len]
            bbox_targets_key_start_dim = sum(bbox_num_per_video[:i])
            cls_branch_ = self.fc_cls(fc_all_2_branch_relu)
            reg_branch_ = self.fc_reg(fc_all_2_branch_relu)

            ohem_inds_ = self.ohem_(post_sampler, cls_branch_, reg_branch_,
                        labels=bbox_targets_key[0][bbox_targets_key_start_dim:bbox_targets_key_start_dim+feat_len],
                        label_weights=bbox_targets_key[1][bbox_targets_key_start_dim:bbox_targets_key_start_dim+feat_len],
                        bbox_targets=bbox_targets_key[2][bbox_targets_key_start_dim:bbox_targets_key_start_dim+feat_len],
                        bbox_weights=bbox_targets_key[3][bbox_targets_key_start_dim:bbox_targets_key_start_dim+feat_len])

            ohem_inds_r.append(ohem_inds_+bbox_targets_key_start_dim)
            
            cls_res_branches.append(cls_branch_[ohem_inds_] if self.with_cls else None)
            reg_res_branches.append(reg_branch_[ohem_inds_] if self.with_reg else None)
            del fc_all_2_branch_relu

            cur_range_ohem_ = dict(start=feat_strt_dim, length=feat_len) # length=ohem_inds_.shape[0]) #! all proposals 
            cur_range_s_ohem_.append(cur_range_ohem_)
            feat_strt_dim_ohem_ = cur_range_ohem_['start']
            feat_len_ohem_ = cur_range_ohem_['length']

            # ! start of third non-local module
            fc_3_in = [fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len], #!use all proposals from nl-2
                       fc_new_feat_1[feat_strt_dim+feat_len:]]
            fc_3_in = torch.cat(fc_3_in, dim=0)
            fc_new_feat_3 = self.fc_new_3(fc_3_in)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_3, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_3, self.key_dim,  
                                        fc_3_in.shape[0], index=3, idx_output_cur_only=True, cur_range_s=[cur_range_ohem_], 
                                        labels=cur_labels, all_labels=all_labels[i], metric_loss=None)
            else:
                attention_3, _ = self.forward_single_selsa(fc_new_feat_3, self.key_dim, 
                                        fc_3_in.shape[0], index=3, 
                                        idx_output_cur_only=True, cur_range_s=[cur_range_ohem_])
            # if self.output_cur_only:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim_ohem_:feat_strt_dim_ohem_+feat_len_ohem_]
            fc_new_feat_3 = fc_new_feat_3[feat_strt_dim_ohem_:feat_strt_dim_ohem_+feat_len_ohem_]
            fc_all_3 = fc_new_feat_3 + attention_3
            del fc_new_feat_3
            del attention_3

            # ############
            # if not self.output_cur_only:
            #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            #     # fc_all_3 = torch.index_select(fc_all_3, dim=0, index=feat_select_tensor)
            #     fc_all_3 = fc_all_3[feat_strt_dim_ohem_:feat_strt_dim_ohem_+feat_len_ohem_]
            # ############
            fc_all_3_relu = self.relu(fc_all_3)
            del fc_all_3
            bbox_num_per_video.append(fc_all_3_relu.size(0))
            video_feats.append(fc_all_3_relu)

        ohem_inds_r = torch.cat(ohem_inds_r, dim=0)

        cur_only_for_4 = False
        loss_metric = TripletNonLocalLoss(margin=10)
        video_feats = torch.cat(video_feats, dim=0)
        fc_new_feat_4 = self.fc_new_4(video_feats)
        target_ranges_4 = []
        for i, cur_range_ohem_ in enumerate(cur_range_s_ohem_):
            feat_strt_dim = cur_range_ohem_['start'] + sum(bbox_num_per_video[:i])
            feat_len = cur_range_ohem_['length']
            target_ranges_4.append(dict(start=feat_strt_dim, length=feat_len))
        # TODO: Forward selsa with mining needs considering whether mining should be involved in or not
        attention_4, m_loss, similarity_ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_4, None, fc_new_feat_4.shape[0], 
                                        index=4, idx_output_cur_only=cur_only_for_4, cur_range_s=target_ranges_4, 
                                        labels=others, all_labels=others, metric_loss=loss_metric)
        loss_additional.update(dict(loss_trip=m_loss))
        # attention_3, _, inds2 = self.forward_single_selsa_with_mining_2(fc_new_feat_3, None, 
        #                           self.nongt_dim, index=3, idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3, labels=others, all_labels=others)
        if cur_only_for_4:
            # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            feats_key_frames = []
            for i, cur_range_ohem_ in enumerate(target_ranges_4):
                feat_strt_dim = cur_range_ohem_['start']
                feat_len = cur_range_ohem_['length']
                feats_key_frames.append(fc_new_feat_4[feat_strt_dim : feat_strt_dim+feat_len])
            del fc_new_feat_4
            fc_new_feat_4 = torch.cat(feats_key_frames, dim=0)

        fc_all_4 = fc_new_feat_4 + attention_4
        del fc_new_feat_4
        del attention_4

        ############
        if not cur_only_for_4:
            # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            feats_key_frames = []
            for i, cur_range_ohem_ in enumerate(target_ranges_4):
                feat_strt_dim = cur_range_ohem_['start']
                feat_len = cur_range_ohem_['length']
                feats_key_frames.append(fc_all_4[feat_strt_dim : feat_strt_dim+feat_len])
            del fc_all_4
            fc_all_4 = torch.cat(feats_key_frames, dim=0)

        ############
        fc_all_4_relu = self.relu(fc_all_4)

        if similarity_ is not None:
            similarity_['similarity'].update(feats_no_relu=fc_all_4.cpu().numpy())
        cls_score_ = self.fc_cls_2(fc_all_4_relu) if self.with_cls else None
        bbox_pred_ = self.fc_reg_2(fc_all_4_relu) if self.with_reg else None

        cls_score_key_frames = []
        bbox_pred_key_frames = []
        ohem_inds_r_2 = []
        for target_range in target_ranges_4:
            bbox_targets_key_start_dim = target_range['start']
            feat_len = target_range['length']
            ohem_inds_ = self.ohem_(post_sampler, cls_score_, bbox_pred_,
                        labels=bbox_targets_key[0][bbox_targets_key_start_dim:bbox_targets_key_start_dim+feat_len],
                        label_weights=bbox_targets_key[1][bbox_targets_key_start_dim:bbox_targets_key_start_dim+feat_len],
                        bbox_targets=bbox_targets_key[2][bbox_targets_key_start_dim:bbox_targets_key_start_dim+feat_len],
                        bbox_weights=bbox_targets_key[3][bbox_targets_key_start_dim:bbox_targets_key_start_dim+feat_len])
            ohem_inds_r_2.append(ohem_inds_+bbox_targets_key_start_dim)
            cls_score_key_frames.append(cls_score_[ohem_inds_])
            bbox_pred_key_frames.append(bbox_pred_[ohem_inds_])

        ohem_inds_r_2 = torch.cat(ohem_inds_r_2, dim=0)
        cls_score_branch = torch.cat(cls_res_branches, dim=0)
        bbox_pred_branch = torch.cat(reg_res_branches, dim=0)
        cls_score = torch.cat(cls_score_key_frames, dim=0)
        bbox_pred = torch.cat(bbox_pred_key_frames, dim=0)
        assert cls_score_branch.shape[0] == bbox_pred_branch.shape[0]
        assert cls_score_branch.shape[0] == cls_score.shape[0]
        
        return [cls_score_branch, cls_score], [bbox_pred_branch, bbox_pred],  \
                loss_additional, similarity_,  [ohem_inds_r, ohem_inds_r_2] #!

        # return [cls_score_branch], [bbox_pred_branch], None, None #!
        # return [cls_score], [bbox_pred] #!

    def forward_v2(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False, others=None, dynamic=False, all_labels=None):
        r"""Main Selsa Functionality implementation
                       ___________________________________________________________________
                      |                     loss                                          |       loss       
                      |                      /                                            |        /
        video_a  -> fc_1 -> NL_1 -> fc_2 -> NL_2                                      -> fc_4 --- NL_4
                      |___________________________\ _______________________________ /_____|
                       ____________________________\ _____________________________ / _____
                      |                      loss   \           loss trip_loss    /       |       loss 
                      |                       /      \             \ /           /        |        /
        video_b  -> fc_1 -> NL_1 -> fc_2 -> NL_2      |-> fc_3 -> NL_3  ---------     -> fc_4 --- NL_4
                      |_____________________________ /__[key frames hardest______\ _______|
                       ____________________________ /______proposal mining________\ ______
                      |                      loss  /        and aggregation]       \      |       loss
                      |                       /   /                                 \     |        /
        video_c  -> fc_1 -> NL_1 -> fc_2 -> NL_2                                      -> fc_4 --- NL_4
                      |___________________________________________________________________|
        Args:
            bbox_feat: Conved style features in shape [roi_nums, num_channel, roi_feat, roi_feat]

        Returns:

        """
        assert cur_range_s is not None, "Feature num range along axis needs specified, \
                                        as length of features for each frame could be different"
        if dynamic and all_labels is None:
            raise AssertionError("`all_labels` should be specified when `dynamic` is `True`")
        
        self.key_dim = key_dim
        self.nongt_dim = self.sampler_num * self.t_dim
        
        loss_additional = dict()
        num_videos = len(cur_range_s)
        video_feats = []
        bbox_num_per_video_2 = []
        cls_res_branches = []
        reg_res_branches = []
        fc_1_feats = []
        imgs_per_video = self.imgs_per_video
        sampler_num = self.sampler_num
        for i, bbox_feat in enumerate(bbox_feat_s):
            cur_range = cur_range_s[i]
            feat_strt_dim = cur_range['start']
            feat_len = cur_range['length']
            # print('enter aggregation module')
            bbox_feat = bbox_feat.view(bbox_feat.size(0), -1)
            fc_new_feat_1 = self.fc_new_1(bbox_feat)
            fc_1_feats.append(fc_new_feat_1)
            attention_1, _ = self.forward_single_selsa(fc_new_feat_1, self.key_dim, 
                                    imgs_per_video*sampler_num, index=1, idx_output_cur_only=False, cur_range_s=[cur_range])
            fc_all_1 = fc_new_feat_1 + attention_1
            # del fc_new_feat_1
            del attention_1
            fc_all_1_relu = self.relu(fc_all_1)
            del fc_all_1
        
            # ! start of second non-local module
            fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_2, m_loss, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_2, self.key_dim,  
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range], 
                                        labels=cur_labels, all_labels=all_labels[i], metric_loss=loss_metric)
                loss_additional.setdefault('loss_trip_video', None)
                if loss_additional['loss_trip_video'] is None:
                    loss_additional.update(dict(loss_trip_video=m_loss))
                else:
                    loss_additional['loss_trip_video'] += m_loss
            else:
                attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range])
            if self.output_cur_only:
                # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
                fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
            fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
            fc_all_2 = fc_new_feat_2 + attention_2
            del fc_new_feat_2
            del attention_2

            ############
            if not self.output_cur_only:
                # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
                # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
                fc_all_2 = fc_all_2[feat_strt_dim:feat_strt_dim+feat_len]
            ############
            fc_all_2_relu = self.relu(fc_all_2)
            del fc_all_2

            fc_all_2_branch_relu = fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len]
            cls_res_branches.append(self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None) 
            reg_res_branches.append(self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None)
            del fc_all_2_branch_relu

            bbox_num_per_video_2.append(fc_all_2_relu.size(0))
            video_feats.append(fc_all_2_relu)

        cur_only_for_3 = True
        bbox_num_per_video_nl_3 = []
        loss_metric = TripletNonLocalLoss(margin=0.2)
        video_feats = torch.cat(video_feats, dim=0)
        fc_new_feat_3 = self.fc_new_3(video_feats)
        target_ranges_3 = []
        for i, cur_range in enumerate(cur_range_s):
            feat_strt_dim = cur_range['start'] + sum(bbox_num_per_video_2[:i])
            feat_len = cur_range['length']
            target_ranges_3.append(dict(start=feat_strt_dim, length=feat_len))
        # TODO: Forward selsa with mining needs considering whether mining should be involved in or not
        attention_3, _ = self.forward_single_selsa(fc_new_feat_3, None, self.nongt_dim, 
                                                           index=3, idx_output_cur_only=cur_only_for_3, 
                                                           cur_range_s=target_ranges_3, mining=False, 
                                                           labels=others, all_labels=all_labels, 
                                                           metric_loss=None)
        # attention_3, m_loss, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_3, None, self.nongt_dim, 
        #                                 index=3, idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3, 
        #                                 labels=others, all_labels=others, metric_loss=loss_metric)
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

        # ############
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
        ############
        fc_all_3_relu = self.relu(fc_all_3)

        labels = others
        fc_all_3_relu_aff = torch.matmul(fc_all_3_relu, fc_all_3_relu.t()).unsqueeze(dim=0)
        anchor_idx, hardest_pos_idx, hardest_neg_idx = self.hardest_proposal_mining(labels, 
                                                                                    labels, 
                                                                                    fc_all_3_relu_aff, 
                                                                                    None)
        m_loss = loss_metric.compute_loss(fc_all_3_relu, fc_all_3_relu, labels, 
                                    [anchor_idx, hardest_pos_idx, hardest_neg_idx])
        loss_additional.update(dict(loss_trip=m_loss))

        cls_score = self.fc_cls_2(fc_all_3_relu) if self.with_cls else None
        bbox_pred = self.fc_reg_2(fc_all_3_relu) if self.with_reg else None
        bbox_num_per_video_nl_3.append(fc_all_3_relu.size(0))
        
        cur_only_for_4 = False
        cls_res_branches_2 = []
        reg_res_branches_2 = []
        bbox_num_per_video_nl_4 = [b_n+cur_range_s[i]['length'] for i,b_n in enumerate(bbox_num_per_video_nl_3)]
        for i, bbox_feat in enumerate(bbox_feat_s):
            cur_range = cur_range_s[i]
            feat_strt_dim = cur_range['start']
            feat_len = cur_range['length']

            fc_new_feat_1 = fc_1_feats[i]
            fc_4_in = torch.cat([fc_all_3_relu[feat_strt_dim + sum(bbox_num_per_video_nl_3[:i]):
                                               feat_strt_dim + sum(bbox_num_per_video_nl_3[:i]) + feat_len], 
                                 fc_new_feat_1], 
                                 dim=0)
            fc_new_feat_4 = self.fc_new_4(fc_4_in)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_4, _ = self.forward_single_selsa_with_mining_inplace(
                                                        fc_new_feat_4, self.key_dim,  
                                                        imgs_per_video*sampler_num, index=4, 
                                                        idx_output_cur_only=cur_only_for_4, 
                                                        cur_range_s=[cur_range], 
                                                        labels=cur_labels, all_labels=all_labels[i], 
                                                        metric_loss=None)
            else:
                attention_4, _ = self.forward_single_selsa(fc_new_feat_4, self.key_dim, 
                                        imgs_per_video*sampler_num, index=4, idx_output_cur_only=cur_only_for_4, cur_range_s=[cur_range])
            if cur_only_for_4:
                # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
                fc_new_feat_4 = fc_new_feat_4[feat_strt_dim:feat_strt_dim+feat_len]
            fc_all_4 = fc_new_feat_4 + attention_4
            del fc_new_feat_4
            del attention_4

            ############
            if not cur_only_for_4:
                fc_all_4 = fc_all_4[feat_strt_dim:feat_strt_dim+feat_len]
            ############
            fc_all_4_relu = self.relu(fc_all_4)
            del fc_all_4

            fc_all_4_branch_relu = fc_all_4_relu
            cls_res_branches_2.append(self.fc_cls_3(fc_all_4_branch_relu) if self.with_cls else None) 
            reg_res_branches_2.append(self.fc_reg_3(fc_all_4_branch_relu) if self.with_reg else None)
            del fc_all_4_branch_relu

        cls_score_branch = torch.cat(cls_res_branches, dim=0)
        bbox_pred_branch = torch.cat(reg_res_branches, dim=0)
        cls_score_branch_2 = torch.cat(cls_res_branches_2, dim=0)
        bbox_pred_branch_2 = torch.cat(reg_res_branches_2, dim=0)
        assert cls_score_branch.shape[0] == bbox_pred_branch.shape[0]
        assert cls_score_branch.shape[0] == cls_score.shape[0]
        assert cls_score_branch.shape[0] == cls_score_branch_2.shape[0]
        
        return [cls_score_branch, cls_score, cls_score_branch_2], \
               [bbox_pred_branch, bbox_pred, bbox_pred_branch_2], loss_additional
        # return [cls_score], [bbox_pred]

    def forward_v3(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False, others=None, dynamic=False, all_labels=None):
        r"""Main Selsa Functionality implementation
                    _______________________________________________________________________
                    |                            loss     |                             /  |
                    |                           /         |                            |   |
        video_a  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3                        |   |
                    |_____________________________________|     \   loss  trip_loss    |   |                      loss
                    _____________________________________        \     \    /          /   |                        /
                    |                          loss       |       \     NL_4     ->  --  fc_5 -> NL-5 -> fc_6 -> NL-6              
                    |                           /         |        \      |            \   |         
        video_b  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3     |-> fc_4           |   |
                    |_____________________________________|        / [key frames       |   |
                    _____________________________________         /  hardest proposal  |   |
                    |                          loss       |      /      mining         |   |
                    |                           /         |     /  and aggregation]    |   |
        video_c  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3                        \   |  # ! Operations after NL-4 are within video
                    |_____________________________________|________________________________|

        Args:
            bbox_feat: Conved style features in shape [roi_nums, num_channel, roi_feat, roi_feat]

        Returns:

        """
        assert cur_range_s is not None, "Feature num range along axis needs specified, \
                                        as length of features for each frame could be different"
        if dynamic and all_labels is None:
            raise AssertionError("`all_labels` should be specified when `dynamic` is `True`")
        
        self.key_dim = key_dim
        self.nongt_dim = self.sampler_num * self.t_dim
        
        with torch.no_grad():
            loss_additional = dict()
            num_videos = len(cur_range_s)
            video_feats = []
            bbox_num_per_video = []
            cls_res_branches = []
            reg_res_branches = []
            fc_1_feats = []
            imgs_per_video = self.imgs_per_video
            sampler_num = self.sampler_num
            for i, bbox_feat in enumerate(bbox_feat_s):
                cur_range = cur_range_s[i]
                feat_strt_dim = cur_range['start']
                feat_len = cur_range['length']
                # print('enter aggregation module')
                bbox_feat = bbox_feat.view(bbox_feat.size(0), -1)
                fc_new_feat_1 = self.fc_new_1(bbox_feat)
                fc_1_feats.append(fc_new_feat_1)
                attention_1, _ = self.forward_single_selsa(fc_new_feat_1, self.key_dim, 
                                        imgs_per_video*sampler_num, index=1, idx_output_cur_only=False, cur_range_s=[cur_range])
                fc_all_1 = fc_new_feat_1 + attention_1
                # del fc_new_feat_1
                del attention_1
                fc_all_1_relu = self.relu(fc_all_1)
                del fc_all_1
            
                # ! start of second non-local module
                fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
                attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                            imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range])
                # if self.output_cur_only:
                #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
                #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
                fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
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

                fc_all_2_branch_relu = fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len]
                #! cls_res_branches.append(self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None) 
                #! reg_res_branches.append(self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None)
                del fc_all_2_branch_relu

                # ! start of third non-local module
                fc_3_in = [fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len],
                        fc_new_feat_1[feat_strt_dim+feat_len:]]
                fc_3_in = torch.cat(fc_3_in, dim=0)
                fc_new_feat_3 = self.fc_new_3(fc_3_in)
                if dynamic:
                    loss_metric = TripletNonLocalLoss(margin=50)
                    cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                    attention_3, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_3, self.key_dim,  
                                            imgs_per_video*sampler_num, index=3, idx_output_cur_only=True, cur_range_s=[cur_range], 
                                            labels=cur_labels, all_labels=all_labels[i], metric_loss=None)
                else:
                    attention_3, _ = self.forward_single_selsa(fc_new_feat_3, self.key_dim, 
                                            imgs_per_video*sampler_num, index=3, idx_output_cur_only=True, cur_range_s=[cur_range])
                # if self.output_cur_only:
                #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
                #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
                fc_new_feat_3 = fc_new_feat_3[feat_strt_dim:feat_strt_dim+feat_len]
                fc_all_3 = fc_new_feat_3 + attention_3
                del fc_new_feat_3
                del attention_3

                # ############
                # if not self.output_cur_only:
                #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
                #     # fc_all_3 = torch.index_select(fc_all_3, dim=0, index=feat_select_tensor)
                #     fc_all_3 = fc_all_3[feat_strt_dim:feat_strt_dim+feat_len]
                # ############
                fc_all_3_relu = self.relu(fc_all_3)
                del fc_all_3
                bbox_num_per_video.append(fc_all_3_relu.size(0))
                video_feats.append(fc_all_3_relu)

            cur_only_for_4 = False
            loss_metric = TripletNonLocalLoss(margin=10)
            video_feats = torch.cat(video_feats, dim=0)
            fc_new_feat_4 = self.fc_new_4(video_feats)
            target_ranges_4 = []
            for i, cur_range in enumerate(cur_range_s):
                feat_strt_dim = cur_range['start'] + sum(bbox_num_per_video[:i])
                feat_len = cur_range['length']
                target_ranges_4.append(dict(start=feat_strt_dim, length=feat_len))
            # TODO: Forward selsa with mining needs considering whether mining should be involved in or not
            attention_4, m_loss, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_4, None, self.nongt_dim, 
                                            index=4, idx_output_cur_only=cur_only_for_4, cur_range_s=target_ranges_4, 
                                            labels=others, all_labels=others, metric_loss=loss_metric)
            #! loss_additional.update(dict(loss_trip=m_loss))
            # attention_3, _, inds2 = self.forward_single_selsa_with_mining_2(fc_new_feat_3, None, 
            #                           self.nongt_dim, index=3, idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3, labels=others, all_labels=others)
            if cur_only_for_4:
                # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
                feats_key_frames = []
                for i, cur_range in enumerate(target_ranges_4):
                    feat_strt_dim = cur_range['start']
                    feat_len = cur_range['length']
                    feats_key_frames.append(fc_new_feat_4[feat_strt_dim : feat_strt_dim+feat_len])
                del fc_new_feat_4
                fc_new_feat_4 = torch.cat(feats_key_frames, dim=0)

            fc_all_4 = fc_new_feat_4 + attention_4
            del fc_new_feat_4
            del attention_4

            ############
            if not cur_only_for_4:
                # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
                # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
                feats_key_frames = []
                for i, cur_range in enumerate(target_ranges_4):
                    feat_strt_dim = cur_range['start']
                    feat_len = cur_range['length']
                    feats_key_frames.append(fc_all_4[feat_strt_dim : feat_strt_dim+feat_len])
                del fc_all_4
                fc_all_4 = torch.cat(feats_key_frames, dim=0)

            ############
            fc_all_4_relu = self.relu(fc_all_4)

            #! cls_score = self.fc_cls_2(fc_all_4_relu) if self.with_cls else None
            #! bbox_pred = self.fc_reg_2(fc_all_4_relu) if self.with_reg else None

        cls_res_branches_2 = []
        reg_res_branches_2 = []
        bbox_num_per_video_nl_4 = [cur_range['length'] for cur_range in cur_range_s]
        for i,cur_range in enumerate(cur_range_s):
            feat_strt_dim = cur_range['start']
            feat_len = cur_range['length']

            fc_new_feat_1 = fc_1_feats[i]
            fc_5_in = torch.cat([fc_all_4_relu[sum(bbox_num_per_video_nl_4[:i]):
                                               sum(bbox_num_per_video_nl_4[:i]) + feat_len], 
                                 fc_new_feat_1[feat_strt_dim+feat_len:]], 
                                 dim=0)
            fc_new_feat_5 = self.fc_new_5(fc_5_in)
            attention_5, _ = self.forward_single_selsa(fc_new_feat_5, self.key_dim, 
                                                        imgs_per_video*sampler_num, index=5, 
                                                        idx_output_cur_only=False, cur_range_s=[cur_range])
            fc_all_5 = fc_new_feat_5 + attention_5
            del fc_new_feat_5
            del attention_5

            fc_all_5_relu = self.relu(fc_all_5)
            del fc_all_5
            
            cur_only_for_6 = True
            fc_new_feat_6 = self.fc_new_6(fc_all_5_relu)
            attention_6, _ = self.forward_single_selsa(fc_new_feat_6, self.key_dim, 
                                                        imgs_per_video*sampler_num, index=6, 
                                                        idx_output_cur_only=cur_only_for_6, cur_range_s=[cur_range])
            if cur_only_for_6:
                # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
                fc_new_feat_6 = fc_new_feat_6[feat_strt_dim:feat_strt_dim+feat_len]
            fc_all_6 = fc_new_feat_6 + attention_6
            del fc_new_feat_6
            del attention_6

            ############
            if not cur_only_for_6:
                fc_all_6 = fc_all_6[feat_strt_dim:feat_strt_dim+feat_len]
            ############
            fc_all_6_relu = self.relu(fc_all_6)
            del fc_all_6

            fc_all_6_branch_relu = fc_all_6_relu
            cls_res_branches_2.append(self.fc_cls_3(fc_all_6_branch_relu) if self.with_cls else None) 
            reg_res_branches_2.append(self.fc_reg_3(fc_all_6_branch_relu) if self.with_reg else None)
            del fc_all_6_branch_relu

        # cls_score_branch = torch.cat(cls_res_branches, dim=0)
        # bbox_pred_branch = torch.cat(reg_res_branches, dim=0)
        cls_score_branch_2 = torch.cat(cls_res_branches_2, dim=0)
        bbox_pred_branch_2 = torch.cat(reg_res_branches_2, dim=0)
        # assert cls_score_branch.shape[0] == bbox_pred_branch.shape[0]
        # assert cls_score_branch.shape[0] == cls_score.shape[0]
        # assert cls_score_branch.shape[0] == cls_score_branch_2.shape[0]
        # # return [cls_score], [bbox_pred]

        # return [cls_score_branch, cls_score, cls_score_branch_2], \
        #        [bbox_pred_branch, bbox_pred, bbox_pred_branch_2], loss_additional, None
        return [cls_score_branch_2], \
               [bbox_pred_branch_2], loss_additional, None

    def forward_v4(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False, others=None, dynamic=False, all_labels=None):
        r"""Main Selsa Functionality implementation
                    _____________________________________
                    |                            loss     |
                    |                           /         |
        video_a  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3
                    |_____________________________________|     \
                    _____________________________________        \
                    |                          loss       |       \                          loss   
                    |                           /         |        \                        /       
        video_b  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3     |-> fc_4 -> NL_4 [key frames hardest proposal 
                    |_____________________________________|        /                        mining and aggregation]
                    _____________________________________         /
                    |                          loss       |      /
                    |                           /         |     /
        video_c  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3
                    |_____________________________________|

        Args:
            bbox_feat: Conved style features in shape [roi_nums, num_channel, roi_feat, roi_feat]

        Returns:

        """
        assert cur_range_s is not None, "Feature num range along axis needs specified, \
                                        as length of features for each frame could be different"
        if dynamic and all_labels is None:
            raise AssertionError("`all_labels` should be specified when `dynamic` is `True`")
        
        self.key_dim = key_dim
        self.nongt_dim = self.sampler_num * self.t_dim
        
        loss_additional = dict()
        num_videos = len(cur_range_s)
        video_feats = []
        bbox_num_per_video = []
        cls_res_branches = []
        reg_res_branches = []
        imgs_per_video = self.imgs_per_video
        sampler_num = self.sampler_num
        for i, bbox_feat in enumerate(bbox_feat_s):
            cur_range = cur_range_s[i]
            feat_strt_dim = cur_range['start']
            feat_len = cur_range['length']
            # print('enter aggregation module')
            bbox_feat = bbox_feat.view(bbox_feat.size(0), -1)
            fc_new_feat_1 = self.fc_new_1(bbox_feat)
            attention_1, _ = self.forward_single_selsa(fc_new_feat_1, self.key_dim, 
                                    imgs_per_video*sampler_num, index=1, idx_output_cur_only=False, cur_range_s=[cur_range])
            fc_all_1 = fc_new_feat_1 + attention_1
            # del fc_new_feat_1
            del attention_1
            fc_all_1_relu = self.relu(fc_all_1)
            del fc_all_1
        
            # ! start of second non-local module
            fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_2, m_loss, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_2, self.key_dim,  
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range], 
                                        labels=cur_labels, all_labels=all_labels[i], metric_loss=loss_metric)
                loss_additional.setdefault('loss_trip_video', None)
                if loss_additional['loss_trip_video'] is None:
                    loss_additional.update(dict(loss_trip_video=m_loss))
                else:
                    loss_additional['loss_trip_video'] += m_loss
            else:
                attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range])
            # if self.output_cur_only:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
            fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
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

            fc_all_2_branch_relu = fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len]
            cls_res_branches.append(self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None) 
            reg_res_branches.append(self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None)
            del fc_all_2_branch_relu

            # ! start of third non-local module
            fc_3_in = [fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len],
                       fc_new_feat_1[feat_strt_dim+feat_len:]]
            fc_3_in = torch.cat(fc_3_in, dim=0)
            fc_new_feat_3 = self.fc_new_3(fc_3_in)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_3, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_3, self.key_dim,  
                                        imgs_per_video*sampler_num, index=3, idx_output_cur_only=True, cur_range_s=[cur_range], 
                                        labels=cur_labels, all_labels=all_labels[i], metric_loss=None)
            else:
                attention_3, _ = self.forward_single_selsa(fc_new_feat_3, self.key_dim, 
                                        imgs_per_video*sampler_num, index=3, idx_output_cur_only=True, cur_range_s=[cur_range])
            # if self.output_cur_only:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
            fc_new_feat_3 = fc_new_feat_3[feat_strt_dim:feat_strt_dim+feat_len]
            fc_all_3 = fc_new_feat_3 + attention_3
            del fc_new_feat_3
            del attention_3

            # ############
            # if not self.output_cur_only:
            #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            #     # fc_all_3 = torch.index_select(fc_all_3, dim=0, index=feat_select_tensor)
            #     fc_all_3 = fc_all_3[feat_strt_dim:feat_strt_dim+feat_len]
            # ############
            fc_all_3_relu = self.relu(fc_all_3)
            del fc_all_3
            bbox_num_per_video.append(fc_all_3_relu.size(0))
            video_feats.append(fc_all_3_relu)

        cur_only_for_4 = False
        loss_metric = None # ! TripletNonLocalLoss(margin=10)
        video_feats = torch.cat(video_feats, dim=0)
        fc_new_feat_4 = self.fc_new_4(video_feats)
        target_ranges_4 = []
        for i, cur_range in enumerate(cur_range_s):
            feat_strt_dim = cur_range['start'] + sum(bbox_num_per_video[:i])
            feat_len = cur_range['length']
            target_ranges_4.append(dict(start=feat_strt_dim, length=feat_len))
        # TODO: Forward selsa with mining needs considering whether mining should be involved in or not
        if loss_metric is not None:
            attention_4, m_loss, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_4, None, self.nongt_dim, 
                                            index=4, idx_output_cur_only=cur_only_for_4, cur_range_s=target_ranges_4, 
                                            labels=others, all_labels=others, metric_loss=loss_metric)
            loss_additional.update(dict(loss_trip=m_loss))
        else:
            attention_4, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_4, None, self.nongt_dim, 
                                            index=4, idx_output_cur_only=cur_only_for_4, cur_range_s=target_ranges_4, 
                                            labels=others, all_labels=others, metric_loss=None)
        # attention_3, _, inds2 = self.forward_single_selsa_with_mining_2(fc_new_feat_3, None, 
        #                           self.nongt_dim, index=3, idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3, labels=others, all_labels=others)
        if cur_only_for_4:
            # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            feats_key_frames = []
            for i, cur_range in enumerate(target_ranges_4):
                feat_strt_dim = cur_range['start']
                feat_len = cur_range['length']
                feats_key_frames.append(fc_new_feat_4[feat_strt_dim : feat_strt_dim+feat_len])
            del fc_new_feat_4
            fc_new_feat_4 = torch.cat(feats_key_frames, dim=0)

        fc_all_4 = fc_new_feat_4 + attention_4
        del fc_new_feat_4
        del attention_4

        ############
        if not cur_only_for_4:
            # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            feats_key_frames = []
            for i, cur_range in enumerate(target_ranges_4):
                feat_strt_dim = cur_range['start']
                feat_len = cur_range['length']
                feats_key_frames.append(fc_all_4[feat_strt_dim : feat_strt_dim+feat_len])
            del fc_all_4
            fc_all_4 = torch.cat(feats_key_frames, dim=0)

        ############
        fc_all_4_relu = self.relu(fc_all_4)

        cls_score = self.fc_cls_2(fc_all_4_relu) if self.with_cls else None
        bbox_pred = self.fc_reg_2(fc_all_4_relu) if self.with_reg else None

        cls_score_branch = torch.cat(cls_res_branches, dim=0)
        bbox_pred_branch = torch.cat(reg_res_branches, dim=0)
        assert cls_score_branch.shape[0] == bbox_pred_branch.shape[0]
        assert cls_score_branch.shape[0] == cls_score.shape[0]
        
        return [cls_score_branch, cls_score], [bbox_pred_branch, bbox_pred], loss_additional, None
        # return [cls_score], [bbox_pred]

    def forward_v5(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False, others=None, dynamic=False, all_labels=None):
        r"""Main Selsa Functionality implementation
                    _______________________________________________________________________________________________________
                    |                            loss     |                             /  |                               |          \
                    |                           /         |                            |   |                               |           |
        video_a  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3                        |   |                               |           |
                    |_____________________________________|     \   loss  trip_loss    |   |                      loss     |           |         loss  metric_loss
                    _____________________________________        \     \    /          /   |                        /      |           |           \    /
                    |                          loss       |       \     NL_4     ->  --  fc_5 -> NL-5 -> fc_6 -> NL-6 -> fc-7 -> NL-7 ----> fc_8 -> NL-8
                    |                           /         |        \      |            \   |                               |           |
        video_b  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3     |-> fc_4           |   |                               |           |
                    |_____________________________________|        / [key frames       |   |                               |           |
                    _____________________________________         /  hardest proposal  |   |                               |           |
                    |                          loss       |      /      mining         |   |                               |           |
                    |                           /         |     /  and aggregation]    |   |                               |           |
        video_c  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3                        \   |                               |           |
                    |_____________________________________|________________________________|_______________________________|          /
        
        Args:
            bbox_feat: Conved style features in shape [roi_nums, num_channel, roi_feat, roi_feat]

        Returns:

        """
        assert cur_range_s is not None, "Feature num range along axis needs specified, \
                                        as length of features for each frame could be different"
        if dynamic and all_labels is None:
            raise AssertionError("`all_labels` should be specified when `dynamic` is `True`")
        
        self.key_dim = key_dim
        self.nongt_dim = self.sampler_num * self.t_dim
        
        with torch.no_grad():
            loss_additional = dict()
            num_videos = len(cur_range_s)
            video_feats = []
            bbox_num_per_video = []
            cls_res_branches = []
            reg_res_branches = []
            fc_1_feats = []
            imgs_per_video = self.imgs_per_video
            sampler_num = self.sampler_num
            for i, bbox_feat in enumerate(bbox_feat_s):
                cur_range = cur_range_s[i]
                feat_strt_dim = cur_range['start']
                feat_len = cur_range['length']
                # print('enter aggregation module')
                bbox_feat = bbox_feat.view(bbox_feat.size(0), -1)
                fc_new_feat_1 = self.fc_new_1(bbox_feat)
                fc_1_feats.append(fc_new_feat_1)
                attention_1, _ = self.forward_single_selsa(fc_new_feat_1, self.key_dim, 
                                        imgs_per_video*sampler_num, index=1, idx_output_cur_only=False, cur_range_s=[cur_range])
                fc_all_1 = fc_new_feat_1 + attention_1
                # del fc_new_feat_1
                del attention_1
                fc_all_1_relu = self.relu(fc_all_1)
                del fc_all_1
            
                # ! start of second non-local module
                fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
                attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                            imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range])
                # if self.output_cur_only:
                #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
                #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
                fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
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

                fc_all_2_branch_relu = fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len]
                #! cls_res_branches.append(self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None) 
                #! reg_res_branches.append(self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None)
                del fc_all_2_branch_relu

                # ! start of third non-local module
                fc_3_in = [fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len],
                        fc_new_feat_1[feat_strt_dim+feat_len:]]
                fc_3_in = torch.cat(fc_3_in, dim=0)
                fc_new_feat_3 = self.fc_new_3(fc_3_in)
                if dynamic:
                    loss_metric = TripletNonLocalLoss(margin=50)
                    cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                    attention_3, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_3, self.key_dim,  
                                            imgs_per_video*sampler_num, index=3, idx_output_cur_only=True, cur_range_s=[cur_range], 
                                            labels=cur_labels, all_labels=all_labels[i], metric_loss=None)
                else:
                    attention_3, _ = self.forward_single_selsa(fc_new_feat_3, self.key_dim, 
                                            imgs_per_video*sampler_num, index=3, idx_output_cur_only=True, cur_range_s=[cur_range])
                # if self.output_cur_only:
                #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
                #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
                fc_new_feat_3 = fc_new_feat_3[feat_strt_dim:feat_strt_dim+feat_len]
                fc_all_3 = fc_new_feat_3 + attention_3
                del fc_new_feat_3
                del attention_3

                # ############
                # if not self.output_cur_only:
                #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
                #     # fc_all_3 = torch.index_select(fc_all_3, dim=0, index=feat_select_tensor)
                #     fc_all_3 = fc_all_3[feat_strt_dim:feat_strt_dim+feat_len]
                # ############
                fc_all_3_relu = self.relu(fc_all_3)
                del fc_all_3
                bbox_num_per_video.append(fc_all_3_relu.size(0))
                video_feats.append(fc_all_3_relu)

            cur_only_for_4 = False
            loss_metric = TripletNonLocalLoss(margin=10)
            video_feats = torch.cat(video_feats, dim=0)
            fc_new_feat_4 = self.fc_new_4(video_feats)
            target_ranges_4 = []
            for i, cur_range in enumerate(cur_range_s):
                feat_strt_dim = cur_range['start'] + sum(bbox_num_per_video[:i])
                feat_len = cur_range['length']
                target_ranges_4.append(dict(start=feat_strt_dim, length=feat_len))
            # TODO: Forward selsa with mining needs considering whether mining should be involved in or not
            attention_4, m_loss, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_4, None, self.nongt_dim, 
                                            index=4, idx_output_cur_only=cur_only_for_4, cur_range_s=target_ranges_4, 
                                            labels=others, all_labels=others, metric_loss=loss_metric)
            #! loss_additional.update(dict(loss_trip_1=m_loss))
            # attention_3, _, inds2 = self.forward_single_selsa_with_mining_2(fc_new_feat_3, None, 
            #                           self.nongt_dim, index=3, idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3, labels=others, all_labels=others)
            if cur_only_for_4:
                # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
                feats_key_frames = []
                for i, cur_range in enumerate(target_ranges_4):
                    feat_strt_dim = cur_range['start']
                    feat_len = cur_range['length']
                    feats_key_frames.append(fc_new_feat_4[feat_strt_dim : feat_strt_dim+feat_len])
                del fc_new_feat_4
                fc_new_feat_4 = torch.cat(feats_key_frames, dim=0)

            fc_all_4 = fc_new_feat_4 + attention_4
            del fc_new_feat_4
            del attention_4

            ############
            if not cur_only_for_4:
                # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
                # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
                feats_key_frames = []
                for i, cur_range in enumerate(target_ranges_4):
                    feat_strt_dim = cur_range['start']
                    feat_len = cur_range['length']
                    feats_key_frames.append(fc_all_4[feat_strt_dim : feat_strt_dim+feat_len])
                del fc_all_4
                fc_all_4 = torch.cat(feats_key_frames, dim=0)

            ############
            fc_all_4_relu = self.relu(fc_all_4)

            #! cls_score = self.fc_cls_2(fc_all_4_relu) if self.with_cls else None
            #! bbox_pred = self.fc_reg_2(fc_all_4_relu) if self.with_reg else None

        cls_res_branches_2 = []
        reg_res_branches_2 = []
        bbox_num_per_video_nl_4 = [cur_range['length'] for cur_range in cur_range_s]
        bbox_num_per_video_nl_7 = []
        video_feats_second_intra = []
        for i,cur_range in enumerate(cur_range_s):
            feat_strt_dim = cur_range['start']
            feat_len = cur_range['length']

            fc_new_feat_1 = fc_1_feats[i]
            fc_5_in = torch.cat([fc_all_4_relu[sum(bbox_num_per_video_nl_4[:i]):
                                               sum(bbox_num_per_video_nl_4[:i]) + feat_len], 
                                 fc_new_feat_1[feat_strt_dim+feat_len:]], 
                                 dim=0)
            fc_new_feat_5 = self.fc_new_5(fc_5_in)
            attention_5, _ = self.forward_single_selsa(fc_new_feat_5, self.key_dim, 
                                                        imgs_per_video*sampler_num, index=5, 
                                                        idx_output_cur_only=False, cur_range_s=[cur_range])
            fc_all_5 = fc_new_feat_5 + attention_5
            del fc_new_feat_5
            del attention_5

            fc_all_5_relu = self.relu(fc_all_5)
            del fc_all_5
            
            fc_new_feat_6 = self.fc_new_6(fc_all_5_relu)
            attention_6, _ = self.forward_single_selsa(fc_new_feat_6, self.key_dim, 
                                                        imgs_per_video*sampler_num, index=6, 
                                                        idx_output_cur_only=False, cur_range_s=[cur_range])
            fc_all_6 = fc_new_feat_6 + attention_6
            del fc_new_feat_6
            del attention_6

            fc_all_6_relu = self.relu(fc_all_6)
            del fc_all_6

            fc_all_6_branch_relu = fc_all_6_relu[feat_strt_dim:feat_strt_dim+feat_len]
            cls_res_branches_2.append(self.fc_cls_3(fc_all_6_branch_relu) if self.with_cls else None) 
            reg_res_branches_2.append(self.fc_reg_3(fc_all_6_branch_relu) if self.with_reg else None)
            del fc_all_6_branch_relu

            fc_7_in = [fc_all_6_relu[feat_strt_dim:feat_strt_dim+feat_len],
                       fc_new_feat_1[feat_strt_dim+feat_len:]]
            fc_7_in = torch.cat(fc_7_in, dim=0)
            fc_new_feat_7 = self.fc_new_7(fc_7_in)
            attention_7, _ = self.forward_single_selsa(fc_new_feat_7, self.key_dim, 
                                        imgs_per_video*sampler_num, index=7, idx_output_cur_only=True, cur_range_s=[cur_range])
            fc_new_feat_7 = fc_new_feat_7[feat_strt_dim:feat_strt_dim+feat_len]
            fc_all_7 = fc_new_feat_7 + attention_7
            del fc_new_feat_7
            del attention_7

            fc_all_7_relu = self.relu(fc_all_7)
            del fc_all_7
            bbox_num_per_video_nl_7.append(fc_all_7_relu.size(0))
            video_feats_second_intra.append(fc_all_7_relu)

        cur_only_for_8 = False
        loss_metric_2 = TripletNonLocalLoss(margin=10)
        video_feats_second_intra = torch.cat(video_feats_second_intra, dim=0)
        fc_new_feat_8 = self.fc_new_8(video_feats_second_intra)
        target_ranges_8 = []
        for i, cur_range in enumerate(cur_range_s):
            feat_strt_dim = cur_range['start'] + sum(bbox_num_per_video_nl_7[:i])
            feat_len = cur_range['length']
            target_ranges_8.append(dict(start=feat_strt_dim, length=feat_len))
        # TODO: Forward selsa with mining needs considering whether mining should be involved in or not
        attention_8, m_loss_2, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_8, None, self.nongt_dim, 
                                        index=8, idx_output_cur_only=cur_only_for_8, cur_range_s=target_ranges_8, 
                                        labels=others, all_labels=others, metric_loss=loss_metric_2)
        loss_additional.update(dict(loss_trip_2=m_loss_2))
        # attention_3, _, inds2 = self.forward_single_selsa_with_mining_2(fc_new_feat_3, None, 
        #                           self.nongt_dim, index=3, idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3, labels=others, all_labels=others)
        if cur_only_for_8:
            # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            feats_key_frames = []
            for i, cur_range in enumerate(target_ranges_8):
                feat_strt_dim = cur_range['start']
                feat_len = cur_range['length']
                feats_key_frames.append(fc_new_feat_8[feat_strt_dim : feat_strt_dim+feat_len])
            del fc_new_feat_8
            fc_new_feat_8 = torch.cat(feats_key_frames, dim=0)

        fc_all_8 = fc_new_feat_8 + attention_8
        del fc_new_feat_8
        del attention_8

        ############
        if not cur_only_for_8:
            # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            feats_key_frames = []
            for i, cur_range in enumerate(target_ranges_8):
                feat_strt_dim = cur_range['start']
                feat_len = cur_range['length']
                feats_key_frames.append(fc_all_8[feat_strt_dim : feat_strt_dim+feat_len])
            del fc_all_8
            fc_all_8 = torch.cat(feats_key_frames, dim=0)

        ############
        fc_all_8_relu = self.relu(fc_all_8)

        cls_score_2 = self.fc_cls_4(fc_all_8_relu) if self.with_cls else None
        bbox_pred_2 = self.fc_reg_4(fc_all_8_relu) if self.with_reg else None


        # cls_score_branch = torch.cat(cls_res_branches, dim=0)
        # bbox_pred_branch = torch.cat(reg_res_branches, dim=0)
        cls_score_branch_2 = torch.cat(cls_res_branches_2, dim=0)
        bbox_pred_branch_2 = torch.cat(reg_res_branches_2, dim=0)
        # assert cls_score_branch.shape[0] == bbox_pred_branch.shape[0]
        # assert cls_score_branch.shape[0] == cls_score.shape[0]
        # assert cls_score_branch.shape[0] == cls_score_branch_2.shape[0]
        # # return [cls_score], [bbox_pred]

        # return [cls_score_branch, cls_score, cls_score_branch_2, cls_score_2], \
        #        [bbox_pred_branch, bbox_pred, bbox_pred_branch_2, bbox_pred_2], loss_additional, None
        return [cls_score_branch_2, cls_score_2], \
               [bbox_pred_branch_2, bbox_pred_2], loss_additional, None

    #TODO: Rewrite this function, most of the functionality needs to be implemented here
    def forward_v6(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False, others=None, dynamic=False, all_labels=None):
        r"""v6
        Main Selsa Functionality implementation
                    __________________________________
                    |                            loss |    
                    |                           /     |    
        video_a  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3
                    |_________________________________|         \
                    __________________________________           \
                    |                          loss   |           \                          loss   trip_loss
                    |                           /     |            \                        /       /
        video_b  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3     |-> fc_4 -> NL_4 [key frames hardest proposal 
                    |_________________________________|            /                        mining and aggregation]
                    __________________________________            /
                    |                          loss   |          /
                    |                           /     |         /
        video_c  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3
                    |_________________________________|

        Args:
            bbox_feat: Conved style features in shape [roi_nums, num_channel, roi_feat, roi_feat]

        Returns:

        """
        assert cur_range_s is not None, "Feature num range along axis needs specified, \
                                        as length of features for each frame could be different"
        if dynamic and all_labels is None:
            raise AssertionError("`all_labels` should be specified when `dynamic` is `True`")
        
        self.key_dim = key_dim
        self.nongt_dim = self.sampler_num * self.t_dim
        
        loss_additional = dict()
        num_videos = len(cur_range_s)
        video_feats = []
        bbox_num_per_video = []
        cls_res_branches = []
        reg_res_branches = []
        imgs_per_video = self.imgs_per_video
        sampler_num = self.sampler_num
        for i, bbox_feat in enumerate(bbox_feat_s):
            cur_range = cur_range_s[i]
            feat_strt_dim = cur_range['start']
            feat_len = cur_range['length']
            # print('enter aggregation module')
            bbox_feat = bbox_feat.view(bbox_feat.size(0), -1)
            fc_new_feat_1 = self.fc_new_1(bbox_feat)
            attention_1, _ = self.forward_single_selsa(fc_new_feat_1, self.key_dim, 
                                    imgs_per_video*sampler_num, index=1, idx_output_cur_only=False, cur_range_s=[cur_range])
            fc_all_1 = fc_new_feat_1 + attention_1
            # del fc_new_feat_1
            del attention_1
            fc_all_1_relu = self.relu(fc_all_1)
            del fc_all_1
        
            # ! start of second non-local module
            fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_2, m_loss, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_2, self.key_dim,  
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range], 
                                        labels=cur_labels, all_labels=all_labels[i], metric_loss=loss_metric)
                loss_additional.setdefault('loss_trip_video', None)
                if loss_additional['loss_trip_video'] is None:
                    loss_additional.update(dict(loss_trip_video=m_loss))
                else:
                    loss_additional['loss_trip_video'] += m_loss
            else:
                attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range])
            # if self.output_cur_only:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
            fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
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

            fc_all_2_branch_relu = fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len]
            cls_res_branches.append(self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None) 
            reg_res_branches.append(self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None)
            del fc_all_2_branch_relu

            # ! start of third non-local module
            fc_3_in = [fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len],
                       fc_new_feat_1[feat_strt_dim+feat_len:]]
            fc_3_in = torch.cat(fc_3_in, dim=0)
            fc_new_feat_3 = self.fc_new_3(fc_3_in)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_3, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_3, self.key_dim,  
                                        imgs_per_video*sampler_num, index=3, idx_output_cur_only=True, cur_range_s=[cur_range], 
                                        labels=cur_labels, all_labels=all_labels[i], metric_loss=None)
            else:
                attention_3, _ = self.forward_single_selsa(fc_new_feat_3, self.key_dim, 
                                        imgs_per_video*sampler_num, index=3, idx_output_cur_only=True, cur_range_s=[cur_range])
            # if self.output_cur_only:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
            fc_new_feat_3 = fc_new_feat_3[feat_strt_dim:feat_strt_dim+feat_len]
            fc_all_3 = fc_new_feat_3 + attention_3
            del fc_new_feat_3
            del attention_3

            # ############
            # if not self.output_cur_only:
            #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            #     # fc_all_3 = torch.index_select(fc_all_3, dim=0, index=feat_select_tensor)
            #     fc_all_3 = fc_all_3[feat_strt_dim:feat_strt_dim+feat_len]
            # ############
            fc_all_3_relu = self.relu(fc_all_3)
            del fc_all_3
            bbox_num_per_video.append(feat_len)
            video_feats.append(fc_all_3_relu)

        video_feats = torch.cat(video_feats, dim=0)
        fc_new_feat_4 = self.fc_new_4(video_feats)
        target_ranges_4 = []
        for i, cur_range in enumerate(cur_range_s):
            feat_strt_dim = cur_range['start'] + sum(bbox_num_per_video[:i])
            feat_len = cur_range['length']
            target_ranges_4.append(dict(start=feat_strt_dim, length=feat_len))
        # TODO: Forward selsa with mining needs considering whether mining should be involved in or not
        attention_4, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_4, None, self.nongt_dim, 
                                        index=4, idx_output_cur_only=False, cur_range_s=target_ranges_4, 
                                        labels=others, all_labels=others, metric_loss=None)

        fc_all_4 = fc_new_feat_4 + attention_4
        fc_all_4_relu = self.relu(fc_all_4)
        del fc_new_feat_4
        del attention_4
        del fc_all_4

        cur_only_for_5 = False
        loss_metric = TripletNonLocalLoss(margin=10)
        fc_new_feat_5 = self.fc_new_5(fc_all_4_relu)
        target_ranges_5 = target_ranges_4
        # TODO: Forward selsa with mining needs considering whether mining should be involved in or not
        attention_5, m_loss, similarity_ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_5, None, self.nongt_dim, 
                                        index=5, idx_output_cur_only=cur_only_for_5, cur_range_s=target_ranges_5, 
                                        labels=others, all_labels=others, metric_loss=loss_metric)
        loss_additional.update(dict(loss_trip=m_loss))
        # attention_3, _, inds2 = self.forward_single_selsa_with_mining_2(fc_new_feat_3, None, 
        #                           self.nongt_dim, index=3, idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3, labels=others, all_labels=others)
        # if cur_only_for_4:
        #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
        #     feats_key_frames = []
        #     for i, cur_range in enumerate(target_ranges_4):
        #         feat_strt_dim = cur_range['start']
        #         feat_len = cur_range['length']
        #         feats_key_frames.append(fc_new_feat_4[feat_strt_dim : feat_strt_dim+feat_len])
        #     del fc_new_feat_4
        #     fc_new_feat_4 = torch.cat(feats_key_frames, dim=0)

        fc_all_5 = fc_new_feat_5 + attention_5
        del fc_new_feat_5
        del attention_5

        ############
        # if not cur_only_for_4:
        #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
        #     # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
        #     feats_key_frames = []
        #     for i, cur_range in enumerate(target_ranges_4):
        #         feat_strt_dim = cur_range['start']
        #         feat_len = cur_range['length']
        #         feats_key_frames.append(fc_all_4[feat_strt_dim : feat_strt_dim+feat_len])
        #     del fc_all_4
        #     fc_all_4 = torch.cat(feats_key_frames, dim=0)

        ############
        fc_all_5_relu = self.relu(fc_all_5)

        if similarity_ is not None:
            similarity_['similarity'].update(feats_no_relu=fc_all_5.cpu().numpy())
        cls_score = self.fc_cls_2(fc_all_5_relu) if self.with_cls else None
        bbox_pred = self.fc_reg_2(fc_all_5_relu) if self.with_reg else None

        cls_score_branch = torch.cat(cls_res_branches, dim=0)
        bbox_pred_branch = torch.cat(reg_res_branches, dim=0)
        assert cls_score_branch.shape[0] == bbox_pred_branch.shape[0]
        assert cls_score_branch.shape[0] == cls_score.shape[0]
        
        return [cls_score_branch, cls_score], [bbox_pred_branch, bbox_pred], loss_additional, similarity_
        # return [cls_score], [bbox_pred]

    #TODO: Rewrite this function, most of the functionality needs to be implemented here
    def forward(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False, others=None, dynamic=False, all_labels=None):
        r"""v7
        Main Selsa Functionality implementation
                    __________________________________
                    |                            loss |    
                    |                           /     |    
        video_a  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3
                    |_________________________________|         \
                    __________________________________           \
                    |                          loss   |           \                          loss   trip_loss
                    |                           /     |            \                        /       /
        video_b  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3     |-> fc_4 -> NL_4 [key frames hardest proposal 
                    |_________________________________|            /                        mining and aggregation]
                    __________________________________            /
                    |                          loss   |          /
                    |                           /     |         /
        video_c  -> fc_1 -> NL_1 -> fc_2 -> NL_2 -> fc_3---NL_3
                    |_________________________________|

        Args:
            bbox_feat: Conved style features in shape [roi_nums, num_channel, roi_feat, roi_feat]

        Returns:

        """
        assert cur_range_s is not None, "Feature num range along axis needs specified, \
                                        as length of features for each frame could be different"
        if dynamic and all_labels is None:
            raise AssertionError("`all_labels` should be specified when `dynamic` is `True`")
        
        self.key_dim = key_dim
        self.nongt_dim = self.sampler_num * self.t_dim
        
        loss_additional = dict()
        num_videos = len(cur_range_s)
        video_feats = []
        bbox_num_per_video = []
        cls_res_branches = []
        reg_res_branches = []
        imgs_per_video = self.imgs_per_video
        sampler_num = self.sampler_num
        for i, bbox_feat in enumerate(bbox_feat_s):
            cur_range = cur_range_s[i]
            feat_strt_dim = cur_range['start']
            feat_len = cur_range['length']
            # print('enter aggregation module')
            bbox_feat = bbox_feat.view(bbox_feat.size(0), -1)
            fc_new_feat_1 = self.fc_new_1(bbox_feat)
            attention_1, _ = self.forward_single_selsa(fc_new_feat_1, self.key_dim, 
                                    imgs_per_video*sampler_num, index=1, idx_output_cur_only=False, cur_range_s=[cur_range])
            fc_all_1 = fc_new_feat_1 + attention_1
            # del fc_new_feat_1
            del attention_1
            fc_all_1_relu = self.relu(fc_all_1)
            del fc_all_1
        
            # ! start of second non-local module
            fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_2, m_loss, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_2, self.key_dim,  
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range], 
                                        labels=cur_labels, all_labels=all_labels[i], metric_loss=loss_metric)
                loss_additional.setdefault('loss_trip_video', None)
                if loss_additional['loss_trip_video'] is None:
                    loss_additional.update(dict(loss_trip_video=m_loss))
                else:
                    loss_additional['loss_trip_video'] += m_loss
            else:
                attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                        imgs_per_video*sampler_num, index=2, idx_output_cur_only=True, cur_range_s=[cur_range])
            # if self.output_cur_only:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
            fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
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

            fc_all_2_branch_relu = fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len]
            cls_res_branches.append(self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None) 
            reg_res_branches.append(self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None)
            del fc_all_2_branch_relu

            # ! start of third non-local module
            fc_3_in = [fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len],
                       fc_new_feat_1[feat_strt_dim+feat_len:]]
            fc_3_in = torch.cat(fc_3_in, dim=0)
            fc_new_feat_3 = self.fc_new_3(fc_3_in)
            if dynamic:
                loss_metric = TripletNonLocalLoss(margin=50)
                cur_labels = others[sum(bbox_num_per_video):sum(bbox_num_per_video)+feat_len]
                attention_3, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_3, self.key_dim,  
                                        imgs_per_video*sampler_num, index=3, idx_output_cur_only=True, cur_range_s=[cur_range], 
                                        labels=cur_labels, all_labels=all_labels[i], metric_loss=None)
            else:
                attention_3, _ = self.forward_single_selsa(fc_new_feat_3, self.key_dim, 
                                        imgs_per_video*sampler_num, index=3, idx_output_cur_only=True, cur_range_s=[cur_range])
            # if self.output_cur_only:
            #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            #     fc_new_feat_2 = fc_new_feat_2[feat_strt_dim:feat_strt_dim+feat_len]
            fc_new_feat_3 = fc_new_feat_3[feat_strt_dim:feat_strt_dim+feat_len]
            fc_all_3 = fc_new_feat_3 + attention_3
            del fc_new_feat_3
            del attention_3

            # ############
            # if not self.output_cur_only:
            #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            #     # fc_all_3 = torch.index_select(fc_all_3, dim=0, index=feat_select_tensor)
            #     fc_all_3 = fc_all_3[feat_strt_dim:feat_strt_dim+feat_len]
            # ############
            fc_all_3_relu = self.relu(fc_all_3)
            del fc_all_3
            bbox_num_per_video.append(feat_len)
            video_feats.append(fc_all_3_relu)

        video_feats = torch.cat(video_feats, dim=0)
        fc_new_feat_4 = self.fc_new_4(video_feats)
        target_ranges_4 = []
        for i, cur_range in enumerate(cur_range_s):
            feat_strt_dim = cur_range['start'] + sum(bbox_num_per_video[:i])
            feat_len = cur_range['length']
            target_ranges_4.append(dict(start=feat_strt_dim, length=feat_len))
        # TODO: Forward selsa with mining needs considering whether mining should be involved in or not
        attention_4, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_4, None, self.nongt_dim, 
                                        index=4, idx_output_cur_only=False, cur_range_s=target_ranges_4, 
                                        labels=others, all_labels=others, metric_loss=None)

        fc_all_4 = fc_new_feat_4 + attention_4
        fc_all_4_relu = self.relu(fc_all_4)
        del fc_new_feat_4
        del attention_4
        del fc_all_4

        cur_only_for_5 = False
        fc_new_feat_5 = self.fc_new_5(fc_all_4_relu)
        target_ranges_5 = target_ranges_4
        # TODO: Forward selsa with mining needs considering whether mining should be involved in or not
        attention_5, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_5, None, self.nongt_dim, 
                                        index=5, idx_output_cur_only=False, cur_range_s=target_ranges_5, 
                                        labels=others, all_labels=others, metric_loss=None)
        # attention_3, _, inds2 = self.forward_single_selsa_with_mining_2(fc_new_feat_3, None, 
        #                           self.nongt_dim, index=3, idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3, labels=others, all_labels=others)
        # if cur_only_for_4:
        #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
        #     feats_key_frames = []
        #     for i, cur_range in enumerate(target_ranges_4):
        #         feat_strt_dim = cur_range['start']
        #         feat_len = cur_range['length']
        #         feats_key_frames.append(fc_new_feat_4[feat_strt_dim : feat_strt_dim+feat_len])
        #     del fc_new_feat_4
        #     fc_new_feat_4 = torch.cat(feats_key_frames, dim=0)

        fc_all_5 = fc_new_feat_5 + attention_5
        del fc_new_feat_5
        del attention_5

        ############
        # if not cur_only_for_4:
        #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
        #     # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
        #     feats_key_frames = []
        #     for i, cur_range in enumerate(target_ranges_4):
        #         feat_strt_dim = cur_range['start']
        #         feat_len = cur_range['length']
        #         feats_key_frames.append(fc_all_4[feat_strt_dim : feat_strt_dim+feat_len])
        #     del fc_all_4
        #     fc_all_4 = torch.cat(feats_key_frames, dim=0)

        ############
        fc_all_5_relu = self.relu(fc_all_5)

        
        cur_only_for_6 = False
        loss_metric = TripletNonLocalLoss(margin=10)
        fc_new_feat_6 = self.fc_new_6(fc_all_5_relu)
        target_ranges_6 = target_ranges_5
        # TODO: Forward selsa with mining needs considering whether mining should be involved in or not
        attention_6, m_loss, similarity_ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_6, None, self.nongt_dim, 
                                        index=6, idx_output_cur_only=cur_only_for_6, cur_range_s=target_ranges_6, 
                                        labels=others, all_labels=others, metric_loss=loss_metric)
        loss_additional.update(dict(loss_trip=m_loss))
        # attention_3, _, inds2 = self.forward_single_selsa_with_mining_2(fc_new_feat_3, None, 
        #                           self.nongt_dim, index=3, idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3, labels=others, all_labels=others)
        # if cur_only_for_4:
        #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
        #     feats_key_frames = []
        #     for i, cur_range in enumerate(target_ranges_4):
        #         feat_strt_dim = cur_range['start']
        #         feat_len = cur_range['length']
        #         feats_key_frames.append(fc_new_feat_4[feat_strt_dim : feat_strt_dim+feat_len])
        #     del fc_new_feat_4
        #     fc_new_feat_4 = torch.cat(feats_key_frames, dim=0)

        fc_all_6 = fc_new_feat_6 + attention_6
        del fc_new_feat_6
        del attention_6

        ############
        # if not cur_only_for_4:
        #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
        #     # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
        #     feats_key_frames = []
        #     for i, cur_range in enumerate(target_ranges_4):
        #         feat_strt_dim = cur_range['start']
        #         feat_len = cur_range['length']
        #         feats_key_frames.append(fc_all_4[feat_strt_dim : feat_strt_dim+feat_len])
        #     del fc_all_4
        #     fc_all_4 = torch.cat(feats_key_frames, dim=0)

        ############
        fc_all_6_relu = self.relu(fc_all_6)

        if similarity_ is not None:
            similarity_['similarity'].update(feats_no_relu=fc_all_6.cpu().numpy())
        cls_score = self.fc_cls_2(fc_all_6_relu) if self.with_cls else None
        bbox_pred = self.fc_reg_2(fc_all_6_relu) if self.with_reg else None

        cls_score_branch = torch.cat(cls_res_branches, dim=0)
        bbox_pred_branch = torch.cat(reg_res_branches, dim=0)
        assert cls_score_branch.shape[0] == bbox_pred_branch.shape[0]
        assert cls_score_branch.shape[0] == cls_score.shape[0]
        
        return [cls_score_branch, cls_score], [bbox_pred_branch, bbox_pred], loss_additional, similarity_
        # return [cls_score], [bbox_pred]

    #TODO: Rewrite this function, most of the functionality needs to be implemented here
    def forward_test_v6(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False):
        """v6
        Main Selsa Functionality implementation

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
        # del fc_new_feat_1
        del attention_1
        fc_all_1_relu = self.relu(fc_all_1)
        del fc_all_1

        fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
        attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                self.nongt_dim, index=2, idx_output_cur_only=False, cur_range_s=[cur_range])
        # attention_2, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_2, self.key_dim, 
        #                           self.nongt_dim, index=2, idx_output_cur_only=self.output_cur_only, test=True, k=150)
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
        # ! for branch prediction
        cls_res_branch = self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None
        reg_res_branch = self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None
        # cls_res_branch = []
        # reg_res_branch = []

        fc_3_in = [fc_new_feat_1[:feat_strt_dim],
                   fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len],
                   fc_new_feat_1[feat_strt_dim+feat_len:]]
        fc_3_in = torch.cat(fc_3_in, dim=0)
        fc_new_feat_3 = self.fc_new_3(fc_3_in)
        attention_3, _ = self.forward_single_selsa(fc_new_feat_3, self.key_dim, 
                                self.nongt_dim, index=3, idx_output_cur_only=False, cur_range_s=[cur_range])       
        # if self.output_cur_only:
        #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
        #     fc_new_feat_3 = fc_new_feat_3[feat_strt_dim:feat_strt_dim+feat_len]
        fc_all_3 = fc_new_feat_3 + attention_3

        ############
        # if not self.output_cur_only:
        #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
        #     # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
        #     fc_all_3 = fc_all_3[feat_strt_dim:feat_strt_dim+feat_len]
        ############
        fc_all_3_relu = self.relu(fc_all_3)

        del fc_new_feat_3
        del attention_3

        fc_new_feat_4 = self.fc_new_4(fc_all_3_relu)
        attention_4, _ = self.forward_single_selsa(fc_new_feat_4, self.key_dim, 
                                self.nongt_dim, index=4, idx_output_cur_only=False, cur_range_s=[cur_range])       
        fc_all_4 = fc_new_feat_4 + attention_4

        del fc_new_feat_4
        del attention_4
        fc_all_4_relu = self.relu(fc_all_4)

        output_cur_only_5 = True
        fc_new_feat_5 = self.fc_new_5(fc_all_4_relu)
        attention_5, _ = self.forward_single_selsa(fc_new_feat_5, self.key_dim, 
                                self.nongt_dim, index=5, idx_output_cur_only=output_cur_only_5, cur_range_s=[cur_range])       
        if output_cur_only_5:
            # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            fc_new_feat_5 = fc_new_feat_5[feat_strt_dim:feat_strt_dim+feat_len]
        fc_all_5 = fc_new_feat_5 + attention_5

        ############
        if not output_cur_only_5:
            # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            fc_all_5 = fc_all_5[feat_strt_dim:feat_strt_dim+feat_len]
        ############
        fc_all_5 = self.relu(fc_all_5)

        cls_score = self.fc_cls_2(fc_all_5) if self.with_cls else None
        bbox_pred = self.fc_reg_2(fc_all_5) if self.with_reg else None
        # print('FOREARD-TEST')

        return [cls_res_branch, cls_score], [reg_res_branch, bbox_pred]

    #TODO: Rewrite this function, most of the functionality needs to be implemented here
    def forward_test_v5(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False):
        """v5
        Main Selsa Functionality implementation

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
        # del fc_new_feat_1
        del attention_1
        fc_all_1_relu = self.relu(fc_all_1)
        del fc_all_1

        fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
        attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                self.nongt_dim, index=2, idx_output_cur_only=False, cur_range_s=[cur_range])
        # attention_2, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_2, self.key_dim, 
        #                           self.nongt_dim, index=2, idx_output_cur_only=self.output_cur_only, test=True, k=150)
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
        # ! for branch prediction
        cls_res_branch = self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None
        reg_res_branch = self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None
        # cls_res_branch = []
        # reg_res_branch = []

        fc_3_in = [fc_new_feat_1[:feat_strt_dim],
                   fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len],
                   fc_new_feat_1[feat_strt_dim+feat_len:]]
        fc_3_in = torch.cat(fc_3_in, dim=0)
        fc_new_feat_3 = self.fc_new_3(fc_3_in)
        attention_3, _ = self.forward_single_selsa(fc_new_feat_3, self.key_dim, 
                                self.nongt_dim, index=3, idx_output_cur_only=False, cur_range_s=[cur_range])       
        # if self.output_cur_only:
        #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
        #     fc_new_feat_3 = fc_new_feat_3[feat_strt_dim:feat_strt_dim+feat_len]
        fc_all_3 = fc_new_feat_3 + attention_3

        ############
        # if not self.output_cur_only:
        #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
        #     # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
        #     fc_all_3 = fc_all_3[feat_strt_dim:feat_strt_dim+feat_len]
        ############
        fc_all_3_relu = self.relu(fc_all_3)

        del fc_new_feat_3
        del attention_3

        fc_new_feat_4 = self.fc_new_4(fc_all_3_relu)
        attention_4, _ = self.forward_single_selsa(fc_new_feat_4, self.key_dim, 
                                self.nongt_dim, index=4, idx_output_cur_only=False, cur_range_s=[cur_range])       
        
        fc_all_4 = fc_new_feat_4 + attention_4
        fc_all_4_relu = self.relu(fc_all_4)

        fc_all_4_branch_relu = fc_all_4_relu[feat_strt_dim:feat_strt_dim+feat_len]
        cls_score = self.fc_cls_2(fc_all_4_branch_relu) if self.with_cls else None
        bbox_pred = self.fc_reg_2(fc_all_4_branch_relu) if self.with_reg else None
        # print('FOREARD-TEST')

        fc_5_in = [fc_new_feat_1[:feat_strt_dim],
                   fc_all_4_relu[feat_strt_dim:feat_strt_dim+feat_len],
                   fc_new_feat_1[feat_strt_dim+feat_len:]]
        fc_5_in = torch.cat(fc_5_in, dim=0)
        fc_new_feat_5 = self.fc_new_5(fc_5_in)
        attention_5, _ = self.forward_single_selsa(fc_new_feat_5, self.key_dim, 
                                self.nongt_dim, index=5, idx_output_cur_only=False, cur_range_s=[cur_range])       
    
        fc_all_5 = fc_new_feat_5 + attention_5
        fc_all_5_relu = self.relu(fc_all_5)

        del fc_new_feat_5
        del attention_5

        output_cur_only_6 = False
        fc_new_feat_6 = self.fc_new_6(fc_all_5_relu)
        attention_6, _ = self.forward_single_selsa(fc_new_feat_6, self.key_dim, 
                                self.nongt_dim, index=6, idx_output_cur_only=output_cur_only_6, cur_range_s=[cur_range])       
        fc_all_6 = fc_new_feat_6 + attention_6

        fc_all_6_relu = self.relu(fc_all_6)

        fc_all_6_relu_branch = fc_all_6_relu[feat_strt_dim:feat_strt_dim+feat_len]
        cls_res_branch_2 = self.fc_cls_3(fc_all_6_relu_branch) if self.with_cls else None
        reg_res_branch_2 = self.fc_reg_3(fc_all_6_relu_branch) if self.with_reg else None
        # print('FOREARD-TEST')

        fc_7_in = [fc_new_feat_1[:feat_strt_dim],
                   fc_all_6_relu[feat_strt_dim:feat_strt_dim+feat_len],
                   fc_new_feat_1[feat_strt_dim+feat_len:]]
        fc_7_in = torch.cat(fc_7_in, dim=0)
        fc_new_feat_7 = self.fc_new_7(fc_7_in)
        attention_7, _ = self.forward_single_selsa(fc_new_feat_7, self.key_dim, 
                                self.nongt_dim, index=7, idx_output_cur_only=False, cur_range_s=[cur_range])       
        # if self.output_cur_only:
        #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
        #     fc_new_feat_3 = fc_new_feat_3[feat_strt_dim:feat_strt_dim+feat_len]
        fc_all_7 = fc_new_feat_7 + attention_7

        ############
        # if not self.output_cur_only:
        #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
        #     # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
        #     fc_all_3 = fc_all_3[feat_strt_dim:feat_strt_dim+feat_len]
        ############
        fc_all_7_relu = self.relu(fc_all_7)

        del fc_new_feat_7
        del attention_7

        output_cur_only_8 = True
        fc_new_feat_8 = self.fc_new_8(fc_all_7_relu)
        attention_8, _ = self.forward_single_selsa(fc_new_feat_8, self.key_dim, 
                                self.nongt_dim, index=8, idx_output_cur_only=output_cur_only_8, cur_range_s=[cur_range])       
        if output_cur_only_8:
            # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            fc_new_feat_8 = fc_new_feat_8[feat_strt_dim:feat_strt_dim+feat_len]
        fc_all_8 = fc_new_feat_8 + attention_8

        ############
        if not output_cur_only_8:
            # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            fc_all_8 = fc_all_8[feat_strt_dim:feat_strt_dim+feat_len]
        ############
        fc_all_8_relu = self.relu(fc_all_8)

        cls_score_2 = self.fc_cls_4(fc_all_8_relu) if self.with_cls else None
        bbox_pred_2 = self.fc_reg_4(fc_all_8_relu) if self.with_reg else None
        # print('FOREARD-TEST')

        return [cls_res_branch, cls_score, cls_res_branch_2, cls_score_2], \
                [reg_res_branch, bbox_pred, reg_res_branch_2, bbox_pred_2], None
        # return [cls_res_branch_2, cls_score_2], \
        #         [reg_res_branch_2, bbox_pred_2]
    
    #TODO: Rewrite this function, most of the functionality needs to be implemented here
    def forward_test_v3(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False):
        """v3 
        Main Selsa Functionality implementation

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
        # del fc_new_feat_1
        del attention_1
        fc_all_1_relu = self.relu(fc_all_1)
        del fc_all_1

        fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
        attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                self.nongt_dim, index=2, idx_output_cur_only=False, cur_range_s=[cur_range])
        # attention_2, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_2, self.key_dim, 
        #                           self.nongt_dim, index=2, idx_output_cur_only=self.output_cur_only, test=True, k=150)
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

        # fc_all_2_branch_relu = fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len]
        # # ! for branch prediction
        # cls_res_branch = self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None
        # reg_res_branch = self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None
        # cls_res_branch = []
        # reg_res_branch = []

        fc_3_in = [fc_new_feat_1[:feat_strt_dim],
                   fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len],
                   fc_new_feat_1[feat_strt_dim+feat_len:]]
        fc_3_in = torch.cat(fc_3_in, dim=0)
        fc_new_feat_3 = self.fc_new_3(fc_3_in)
        attention_3, _ = self.forward_single_selsa(fc_new_feat_3, self.key_dim, 
                                self.nongt_dim, index=3, idx_output_cur_only=False, cur_range_s=[cur_range])       
        # if self.output_cur_only:
        #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
        #     fc_new_feat_3 = fc_new_feat_3[feat_strt_dim:feat_strt_dim+feat_len]
        fc_all_3 = fc_new_feat_3 + attention_3

        ############
        # if not self.output_cur_only:
        #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
        #     # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
        #     fc_all_3 = fc_all_3[feat_strt_dim:feat_strt_dim+feat_len]
        ############
        fc_all_3_relu = self.relu(fc_all_3)

        del fc_new_feat_3
        del attention_3

        fc_new_feat_4 = self.fc_new_4(fc_all_3_relu)
        attention_4, _ = self.forward_single_selsa(fc_new_feat_4, self.key_dim, 
                                self.nongt_dim, index=4, idx_output_cur_only=False, cur_range_s=[cur_range])       
        
        fc_all_4 = fc_new_feat_4 + attention_4
        fc_all_4_relu = self.relu(fc_all_4)

        # fc_all_4_branch_relu = fc_all_4_relu[feat_strt_dim:feat_strt_dim+feat_len]
        # cls_score = self.fc_cls_2(fc_all_4_branch_relu) if self.with_cls else None
        # bbox_pred = self.fc_reg_2(fc_all_4_branch_relu) if self.with_reg else None
        # # print('FOREARD-TEST')

        fc_5_in = [fc_new_feat_1[:feat_strt_dim],
                   fc_all_4_relu[feat_strt_dim:feat_strt_dim+feat_len],
                   fc_new_feat_1[feat_strt_dim+feat_len:]]
        fc_5_in = torch.cat(fc_5_in, dim=0)
        fc_new_feat_5 = self.fc_new_5(fc_5_in)
        attention_5, _ = self.forward_single_selsa(fc_new_feat_5, self.key_dim, 
                                self.nongt_dim, index=5, idx_output_cur_only=False, cur_range_s=[cur_range])       
    
        fc_all_5 = fc_new_feat_5 + attention_5
        fc_all_5_relu = self.relu(fc_all_5)

        del fc_new_feat_5
        del attention_5

        output_cur_only_6 = True
        fc_new_feat_6 = self.fc_new_6(fc_all_5_relu)
        attention_6, _ = self.forward_single_selsa(fc_new_feat_6, self.key_dim, 
                                self.nongt_dim, index=6, idx_output_cur_only=output_cur_only_6, cur_range_s=[cur_range])       
        if output_cur_only_6:
            # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            fc_new_feat_6 = fc_new_feat_6[feat_strt_dim:feat_strt_dim+feat_len]
        fc_all_6 = fc_new_feat_6 + attention_6

        ############
        if not output_cur_only_6:
            # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            fc_all_6 = fc_all_6[feat_strt_dim:feat_strt_dim+feat_len]
        ############
        fc_all_6_relu = self.relu(fc_all_6)

        cls_res_branch_2 = self.fc_cls_3(fc_all_6_relu) if self.with_cls else None
        reg_res_branch_2 = self.fc_reg_3(fc_all_6_relu) if self.with_reg else None
        # print('FOREARD-TEST')

        # return [cls_res_branch, cls_score, cls_res_branch_2], \
        #         [reg_res_branch, bbox_pred, reg_res_branch_2]
        return [cls_res_branch_2], \
                [reg_res_branch_2]

    #TODO: Rewrite this function, most of the functionality needs to be implemented here
    def forward_test_v2(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False):
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
        # del fc_new_feat_1
        del attention_1
        fc_all_1_relu = self.relu(fc_all_1)
        del fc_all_1

        fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
        attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                self.nongt_dim, index=2, idx_output_cur_only=False, cur_range_s=[cur_range])
        # attention_2, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_2, self.key_dim, 
        #                           self.nongt_dim, index=2, idx_output_cur_only=self.output_cur_only, test=True, k=150)
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
        # ! for branch prediction
        cls_res_branch_1 = self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None
        reg_res_branch_1 = self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None
        # cls_res_branch = []
        # reg_res_branch = []

        fc_new_feat_3 = self.fc_new_3(fc_all_2_relu)
        attention_3, _ = self.forward_single_selsa(fc_new_feat_3, self.key_dim, 
                                self.nongt_dim, index=3, idx_output_cur_only=False, cur_range_s=[cur_range])       
        # if output_cur_only_3:
        #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
        #     fc_new_feat_3 = fc_new_feat_3[feat_strt_dim:feat_strt_dim+feat_len]
        fc_all_3 = fc_new_feat_3 + attention_3

        ############
        # if not output_cur_only_3:
        #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
        #     # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
        #     fc_all_3 = fc_all_3[feat_strt_dim:feat_strt_dim+feat_len]
        ############
        fc_all_3_relu = self.relu(fc_all_3)

        fc_all_3_branch_relu = fc_all_3_relu[feat_strt_dim:feat_strt_dim+feat_len]
        cls_res_branch_2 = self.fc_cls_2(fc_all_3_branch_relu) if self.with_cls else None
        reg_res_branch_2 = self.fc_reg_2(fc_all_3_branch_relu) if self.with_reg else None

        output_cur_only_4 = True
        fc_4_in = [fc_new_feat_1[:feat_strt_dim],
                   fc_all_3_relu[feat_strt_dim:feat_strt_dim+feat_len],
                   fc_new_feat_1[feat_strt_dim+feat_len:]]
        fc_4_in = torch.cat(fc_4_in, dim=0)
        fc_new_feat_4 = self.fc_new_4(fc_4_in)
        attention_4, _ = self.forward_single_selsa(fc_new_feat_4, self.key_dim, 
                                self.nongt_dim, index=4, idx_output_cur_only=output_cur_only_4, cur_range_s=[cur_range])       
        if output_cur_only_4:
            # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            fc_new_feat_4 = fc_new_feat_4[feat_strt_dim:feat_strt_dim+feat_len]
        fc_all_4 = fc_new_feat_4 + attention_4

        ############
        if not output_cur_only_4:
            # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            fc_all_4 = fc_all_4[feat_strt_dim:feat_strt_dim+feat_len]
        ############
        fc_all_4_relu = self.relu(fc_all_4)

        cls_score = self.fc_cls_3(fc_all_4_relu) if self.with_cls else None
        bbox_pred = self.fc_reg_3(fc_all_4_relu) if self.with_reg else None
        # print('FOREARD-TEST')

        return [cls_res_branch_1, cls_res_branch_2, cls_score], [reg_res_branch_1, reg_res_branch_2, bbox_pred]

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
        # del fc_new_feat_1
        del attention_1
        fc_all_1_relu = self.relu(fc_all_1)
        del fc_all_1

        fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
        attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
                                self.nongt_dim, index=2, idx_output_cur_only=False, cur_range_s=[cur_range])
        # attention_2, _ = self.forward_single_selsa_with_mining_inplace(fc_new_feat_2, self.key_dim, 
        #                           self.nongt_dim, index=2, idx_output_cur_only=self.output_cur_only, test=True, k=150)
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
        # ! for branch prediction
        cls_res_branch = self.fc_cls(fc_all_2_branch_relu) if self.with_cls else None
        reg_res_branch = self.fc_reg(fc_all_2_branch_relu) if self.with_reg else None
        # cls_res_branch = []
        # reg_res_branch = []

        fc_3_in = [fc_new_feat_1[:feat_strt_dim],
                   fc_all_2_relu[feat_strt_dim:feat_strt_dim+feat_len],
                   fc_new_feat_1[feat_strt_dim+feat_len:]]
        fc_3_in = torch.cat(fc_3_in, dim=0)
        fc_new_feat_3 = self.fc_new_3(fc_3_in)
        attention_3, _ = self.forward_single_selsa(fc_new_feat_3, self.key_dim, 
                                self.nongt_dim, index=3, idx_output_cur_only=False, cur_range_s=[cur_range])       
        # if self.output_cur_only:
        #     # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
        #     fc_new_feat_3 = fc_new_feat_3[feat_strt_dim:feat_strt_dim+feat_len]
        fc_all_3 = fc_new_feat_3 + attention_3

        ############
        # if not self.output_cur_only:
        #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
        #     # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
        #     fc_all_3 = fc_all_3[feat_strt_dim:feat_strt_dim+feat_len]
        ############
        fc_all_3_relu = self.relu(fc_all_3)

        del fc_new_feat_3
        del attention_3

        output_cur_only_4 = True
        fc_new_feat_4 = self.fc_new_4(fc_all_3_relu)
        attention_4, _ = self.forward_single_selsa(fc_new_feat_4, self.key_dim, 
                                self.nongt_dim, index=4, idx_output_cur_only=output_cur_only_4, cur_range_s=[cur_range])       
        if output_cur_only_4:
            # fc_new_feat_2 = torch.index_select(fc_new_feat_2, dim=0, index=feat_select_tensor)
            fc_new_feat_4 = fc_new_feat_4[feat_strt_dim:feat_strt_dim+feat_len]
        fc_all_4 = fc_new_feat_4 + attention_4

        ############
        if not output_cur_only_4:
            # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            fc_all_4 = fc_all_4[feat_strt_dim:feat_strt_dim+feat_len]
        ############
        fc_all_4_relu = self.relu(fc_all_4)

        cls_score = self.fc_cls_2(fc_all_4_relu) if self.with_cls else None
        bbox_pred = self.fc_reg_2(fc_all_4_relu) if self.with_reg else None
        # print('FOREARD-TEST')

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
            for i,score in enumerate(cls_score):
                losses['loss_cls_{:d}'.format(i+1)] = self.loss_cls(
                    score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc_{:d}'.format(i+1)] = accuracy(score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            pos_bbox_pred = []
            for b_p in bbox_pred:
                if self.reg_class_agnostic:
                    pos_bbox_pred.append(b_p.view(b_p.size(0), 4)[pos_inds])
                else:
                    pos_bbox_pred.append(b_p.view(b_p.size(0), -1,
                                                4)[pos_inds, labels[pos_inds]])
            for i,bbox in enumerate(pos_bbox_pred):
                losses['loss_bbox_{:d}'.format(i+1)] = self.loss_bbox(
                    bbox,
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

            if cfg is None or not hasattr(cfg, 'nms'):
                bboxes_collect.append(bboxes)
                scores_collect.append(scores)
            else:
                det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
                bboxes_collect.append(det_bboxes)
                scores_collect.append(det_labels)

        return bboxes_collect, scores_collect
