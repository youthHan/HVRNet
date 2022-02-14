import math
import torch
import torch.nn as nn
from collections import OrderedDict

from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head import BBoxHead


@HEADS.register_module
class HNLBBoxHead(BBoxHead):
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
        super(HNLBBoxHead, self).__init__(*args, **kwargs)
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
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.dim[2], out_dim_reg)

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
        super(HNLBBoxHead, self).init_weights()
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
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.01)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward_single_selsa(self, 
                            roi_feat, 
                            key_dim,
                            nongt_dim, 
                            index, 
                            cur_range_s=None,
                            non_cur_space=False, 
                            idx_output_cur_only=False):
        '''
        args:
        '''
        # assert not non_cur_space and not idx_output_cur_only, "Under HNLBBoxhead, these two options haven't been carefully reviewed"
        assert not non_cur_space, "Under HNLBBoxhead, `non_cur_space` hasn't been carefully reviewed"
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

        # TODO: Fix this function under HNLBBoxhead
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

        return output, None

    #TODO: Rewrite this function, most of the functionality needs to be implemented here
    def forward(self, bbox_feat_s, cur_range_s=None, key_dim=0, all_res=False, others=None):
        """Main Selsa Functionality implementation

        Args:
            bbox_feat: Conved style features in shape [roi_nums, num_channel, roi_feat, roi_feat]

        Returns:

        """
        assert cur_range_s is not None, "Feature num range along axis needs specified, \
                                        as length of features for each frame could be different"
        self.key_dim = key_dim
        self.nongt_dim = self.sampler_num * self.t_dim
        
        num_videos = len(cur_range_s)
        video_feats = []
        bbox_num_per_video = []
        imgs_per_video = self.imgs_per_video
        sampler_num = self.sampler_num
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
            # ############
            # if not self.output_cur_only:
            #     # assert self.nongt_dim%self.t_dim == 0, "Error! self.nongt_dim should be divisible by self.t_dim"
            #     # fc_all_2 = torch.index_select(fc_all_2, dim=0, index=feat_select_tensor)
            #     fc_all_2 = fc_all_2[feat_strt_dim:feat_strt_dim+feat_len]
            # ############
            fc_all_2_relu = self.relu(fc_all_2)
            del fc_all_2
            video_feats.append(fc_all_2_relu)

        cur_only_for_3 = True
        video_feats = torch.cat(video_feats, dim=0)
        fc_new_feat_3 = self.fc_new_3(video_feats)
        target_ranges_3 = []
        for i, cur_range in enumerate(cur_range_s):
            feat_strt_dim = cur_range['start'] + sum(bbox_num_per_video[:i])
            feat_len = cur_range['length']
            target_ranges_3.append(dict(start=feat_strt_dim, length=feat_len))
        attention_3, _ = self.forward_single_selsa(fc_new_feat_3, None, 
                                  self.nongt_dim, index=3, idx_output_cur_only=cur_only_for_3, cur_range_s=target_ranges_3)
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

        ############
        fc_all_3_relu = self.relu(fc_all_3)

        cls_score = self.fc_cls(fc_all_3_relu) if self.with_cls else None
        bbox_pred = self.fc_reg(fc_all_3_relu) if self.with_reg else None

        return cls_score, bbox_pred

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

        cls_score = self.fc_cls(fc_all_3_relu) if self.with_cls else None
        bbox_pred = self.fc_reg(fc_all_3_relu) if self.with_reg else None

        return cls_score, bbox_pred

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
            attention_1, _ = self.forward_single_selsa(fc_new_feat_1, self.key_dim, 
                                    self.nongt_dim, index=1, idx_output_cur_only=False)
            fc_all_1 = fc_new_feat_1 + attention_1
            del fc_new_feat_1
            del attention_1
            fc_all_1_relu = self.relu(fc_all_1)
            del fc_all_1

            fc_new_feat_2 = self.fc_new_2(fc_all_1_relu)
            attention_2, _ = self.forward_single_selsa(fc_new_feat_2, self.key_dim, 
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

        cls_score = self.fc_cls(fc_all_3_relu) if self.with_cls else None
        bbox_pred = self.fc_reg(fc_all_3_relu) if self.with_reg else None

        return cls_score, bbox_pred