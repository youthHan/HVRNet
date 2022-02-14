import math
import collections
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d
from torch.nn.functional import softmax

from ..registry import DETECTORS
from .two_stage import TwoStageDetector

from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, merge_aug_proposals, multiclass_nms)

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
@DETECTORS.register_module
class HNMBRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 loss_frames=1):
        super(HNMBRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        # assert not (self.train_cfg and self.test_cfg), "Wrong! train_cfg and test_cfg exists at the same time in SelsaRCNN"
        if self.train_cfg is not None:
            self.key_dim = int(self.train_cfg.rcnn.key_dim)
        else:
            # self.key_dim = int(self.test_cfg.relation_setup.frame_interval)
            self.key_dim = int(self.test_cfg.bbox_head.key_dim)
            self.bbox_head.t_dim = int(test_cfg.bbox_head.t_dim)
            self.bbox_head.sampler_num = int(test_cfg.bbox_head.sampler_num)

    '''
    Extract c4 feat for RPN and c5 feat to fuse for video-wise coomparison.
    Notice! The process in this function performed under `torch.no_grad()`.
    '''
    def extract_c4_c5_feat(self, img, imgs_per_video=3,train_as_whole=False):
        c4_feats_all = []
        c5_feats_all = []
        img_keep = img.split(imgs_per_video, dim=0)
        if not train_as_whole:
            with torch.no_grad():
                for _, img in enumerate(img_keep):
                    x = self.extract_feat(img)
                    c5_feats = [self.shared_head(x[0])]
                    c4_feats_all.append(x)
                    c5_feats_all.append(c5_feats)
            return c4_feats_all, c5_feats_all
        else:
            with torch.no_grad():
                for _, img in enumerate(img_keep):
                    x = self.extract_feat(img)
                    c5_feats = [self.shared_head(x[0])]
                    c5_feats_all.append(c5_feats)
            return _, c5_feats_all

    def get_triplet_patches(self, c5_feats_all, key_video=0, imgs_per_video=3, extra_cls = 4, video_per_cls=3):
        video_feats = []
        for c5_feats in c5_feats_all:
            avg_pool_feat = adaptive_avg_pool2d(c5_feats[0], (1,1))
            avg_pool_feat_squeeze = avg_pool_feat.squeeze()
            video_feats.append(torch.unsqueeze(avg_pool_feat_squeeze.max(dim=0).values, dim=0))
        # By default, key_video index is 0
        key_cls_videos = video_feats[0: (0+1)*video_per_cls]
        key_cls_videos_cat = torch.cat(key_cls_videos, dim=0)
        key_dims = key_cls_videos_cat.shape[-1]
        key_cls_videos_cat_per = key_cls_videos_cat.permute(1,0)
        key_cls_sim = torch.mm(key_cls_videos[0], key_cls_videos_cat_per)
        key_cls_sim_scale = (1.0 / math.sqrt(float(key_dims))) * key_cls_sim
        key_cls_sim_scale = softmax(key_cls_sim_scale, dim=1)
        chosen_key_video_id = torch.argmin(key_cls_sim_scale[:, 1:], dim=1, keepdim=True)[0][0].tolist() + 1

        chosen_key_videos = torch.cat([key_cls_videos[key_video], key_cls_videos[chosen_key_video_id]], dim=0)
        extra_cls_videos = video_feats[(0+1)*video_per_cls:]
        extra_cls_videos_cat = torch.cat(extra_cls_videos, dim=0)
        extra_cls_videos_cat_per = extra_cls_videos_cat.permute(1,0)
        extra_cls_sim = torch.mm(chosen_key_videos, extra_cls_videos_cat_per)
        extra_cls_sim_scale = (1.0 / math.sqrt(float(key_dims))) * extra_cls_sim
        extra_cls_sim_scale = softmax(extra_cls_sim_scale, dim=1)
        extra_cls_sim_scale_sum = torch.unsqueeze(extra_cls_sim_scale.sum(dim=0), dim=0)
        # By default, the feats begin with key videos with number of `video_per_cls`
        chosen_extra_video_id = torch.argmax(extra_cls_sim_scale_sum, dim=1, keepdim=True)[0][0].tolist() + video_per_cls

        return [key_video, chosen_key_video_id, chosen_extra_video_id]

    
    def forward_feat_aug(self, x=None, img_meta=None, proposals=None, rescale=False):
        assert x is not None and img_meta is not None
        if isinstance(x, collections.Sequence):
            assert len(x)==len(img_meta)
            assert len(x[0])==len(img_meta[0])
            assert isinstance(x[0][0], torch.Tensor)

        feats = []
        img_metas = []
        scales_num = len(x[0])
        for i in range(scales_num):
            feats.append([torch.cat([x_[i] for x_ in x], dim=0)])
            img_metas.append([meta[i] for meta in img_meta])
        # x=[[x_ for x_ in x_scales ]for x_scales in x]

        bbox_feats_all_scales = []
        for feat_ in feats:
            if self.feat_from_shared_head:
                bbox_feats_all = [self.shared_head(feat_[0])]
            else:
                bbox_feats_all = feat_
            bbox_feats_all_scales.append(bbox_feats_all)

        imgs_per_gpu = len(img_metas[0])
        # imgs_per_gpu = 1
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)

        aug_img_metas = []
        for i in range(imgs_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)

        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, self.test_cfg.rpn)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]

        proposal_list = merged_proposals

        # x=[torch.cat(tuple(x),dim=0)]

        # if self.feat_from_shared_head:
        #     bbox_feats_all = [self.shared_head(x[0])]
        # else:
        #     bbox_feats_all = x

        # proposal_list = self.simple_test_rpn(
        #     x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        # assert isinstance(proposal_list, collections.Sequence)

        det_bboxes_collect, det_labels_collect = self.aug_test_bboxes(
            bbox_feats_all_scales, img_metas, proposal_list,
            self.test_cfg.rcnn)

        bbox_results_collect = []
        for det_bboxes,det_labels in zip(det_bboxes_collect, det_labels_collect):
            if rescale:
                _det_bboxes = det_bboxes
            else:
                _det_bboxes = det_bboxes.clone()
                _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']

            if hasattr(self.test_cfg.rcnn, 'nms'):
                bbox_results = bbox2result(_det_bboxes, det_labels,
                                        self.bbox_head.num_classes)
                bbox_results_collect.append(bbox_results)
            else:
                bbox_results_collect.append([_det_bboxes,det_labels])
                
        # det_bboxes always keep the original scale
        return bbox_results_collect


    def get_roi_feat(self, x, rois):
        if self.feat_from_shared_head:
            # print("enter feat_from_shared_head, type of x is {}".format(type(x)))
            bbox_feats = self.bbox_roi_extractor(
                        x[:self.bbox_roi_extractor.num_inputs], rois)
        else:
            bbox_feats = self.bbox_roi_extractor(
                        x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
        return bbox_feats

    def forward_feat(self, x=None, img_meta=None, proposals=None, rescale=False):
        assert x is not None and img_meta is not None
        if isinstance(x, collections.Sequence):
            assert len(x)==len(img_meta)
            assert isinstance(x[0], torch.Tensor)
        x=[torch.cat(tuple(x),dim=0)]

        if self.feat_from_shared_head:
            bbox_feats_all = [self.shared_head(x[0])]
        else:
            bbox_feats_all = x

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        assert isinstance(proposal_list, collections.Sequence)

        det_bboxes_collect, det_labels_collect = self.simple_test_bboxes(
            bbox_feats_all, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)

        bbox_results_collect = []
        for det_bboxes,det_labels in zip(det_bboxes_collect, det_labels_collect):
            bbox_results = bbox2result(det_bboxes, det_labels,
                                    self.bbox_head.num_classes)
            bbox_results_collect.append(bbox_results)

        assert not self.with_mask
        if not self.with_mask:
            return bbox_results_collect

    def forward_train(self,
                        img,
                        img_metas,
                        gt_bboxes,
                        gt_labels,
                        gt_bboxes_ignore=None,
                        gt_masks=None,
                        proposals=None):
            """
            Args:
                img (Tensor): of shape (N, C, H, W) encoding input images.
                    Typically these should be mean centered and std scaled.

                img_metas (list[dict]): list of image info dict where each dict has:
                    'img_shape', 'scale_factor', 'flip', and my also contain
                    'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                    For details on the values of these keys see
                    `mmdet/datasets/pipelines/formatting.py:Collect`.

                gt_bboxes (list[Tensor]): each item are the truth boxes for each
                    image in [tl_x, tl_y, br_x, br_y] format.

                gt_labels (list[Tensor]): class indices corresponding to each box

                gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                    boxes can be ignored when computing the loss.

                gt_masks (None | Tensor) : true segmentation masks for each box
                    used if the architecture supports a segmentation task.

                proposals : override rpn proposals with custom proposals. Use when
                    `with_rpn` is False.

            Returns:
                dict[str, Tensor]: a dictionary of loss components
            """
            assert gt_masks is None, "Masks functino has not been implemented"
            assert proposals is None, "RPN head is needed!"
            key_dim = self.key_dim
            assert key_dim == 0, "Key_dim has to be 0 in `HNMBRCNN.forward_train()`!"
            key_video = 0
            imgs_per_video = 3
            video_per_cls = 3
            extra_cls = 2 #4
            # TODO: change the name to match (video_level has been changed in VIDSeqDataset!)
            # video_levels = 2 + 1

            # with torch.no_grad():
            train_as_whole = False
            if not train_as_whole:
                c4_feats_all, c5_feats_all = self.extract_c4_c5_feat(img, imgs_per_video=imgs_per_video,train_as_whole=train_as_whole)
            else:
                _, c5_feats_all = self.extract_c4_c5_feat(img, imgs_per_video=imgs_per_video,train_as_whole=train_as_whole)
            chosen_video_ids = self.get_triplet_patches(c5_feats_all, key_video, imgs_per_video, extra_cls, video_per_cls)

            # del img
            del c5_feats_all

            cur_ranges = []
            if not train_as_whole:
                c4_feats_keep = [c4_feats_all[i] for i in chosen_video_ids]
                # gt_bboxes_ignore_keep = gt_labels.split(gt_bboxes_ignore, dim=0) if gt_bboxes_ignore is not None else None

                del c4_feats_all
                del img
                # del img_meta
            else:
                c4_feats_keep = []
                img_all_videos = img.split(imgs_per_video, dim=0)
                for i in chosen_video_ids:
                    x = self.extract_feat(img_all_videos[i])
                    c4_feats_keep.append(x)
                del img
                
            img_meta_keep = [img_metas[i*imgs_per_video : (i+1)*imgs_per_video] for i in chosen_video_ids]
            gt_bboxes_keep = [gt_bboxes[i*imgs_per_video : (i+1)*imgs_per_video] for i in chosen_video_ids]
            gt_labels_keep = [gt_labels[i*imgs_per_video : (i+1)*imgs_per_video] for i in chosen_video_ids]
            assert gt_bboxes_ignore is None, "gt_bbox_ignore is not None! Implement this function!"

            feats = []
            cur_ranges = []
            rois_key_all = []
            sampling_results_videos = []
            gt_bboxes_chosen_videos = []
            gt_labels_chosen_videos = []
            for idx, x in enumerate(c4_feats_keep):
                # Caution: gt_bboxes, gt_labels and img_meta have been updated!!!!
                gt_bboxes = gt_bboxes_keep[idx]
                gt_bboxes_chosen_videos.append(gt_bboxes)
                gt_labels = gt_labels_keep[idx]
                gt_labels_chosen_videos.append(gt_labels)
                img_meta = img_meta_keep[idx]
                losses = dict()
                num_imgs = len(img_meta)

                # RPN forward and loss
                #TODO: !!!!!!Caution: RPN Loss only with the reference frame!!!!!!
                if self.with_rpn:
                    with torch.no_grad():
                        rpn_outs = self.rpn_head(x)
                        # rpn_outs_split = [torch.split(r_out[0],1,dim=0)[key_dim:key_dim+1] for r_out in rpn_outs]
                        # ###########################
                        # rpn_loss_inputs = tuple(rpn_outs_split) + (gt_bboxes[key_dim:key_dim+1], img_meta[key_dim:key_dim+1], self.train_cfg.rpn)
                        # rpn_losses = self.rpn_head.loss(
                        #     *rpn_loss_inputs, gt_bboxes_ignore=None if gt_bboxes_ignore is None else gt_bboxes_ignore[key_dim])
                        # ############################
                        # losses.update(rpn_losses)

                        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                        self.test_cfg.rpn)
                        proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
                        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
                else:
                    proposal_list = proposals

                if self.feat_from_shared_head:
                    bbox_feats_all = [self.shared_head(x[0])]
                else:
                    bbox_feats_all = x
                del rpn_outs
                # del rpn_outs_split

                # assign gts and sample proposals
                bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
                if isinstance(self.train_cfg.rcnn.sampler, list):
                    bbox_sampler, post_bbox_sampler = build_sampler(
                                        self.train_cfg.rcnn.sampler, context=self)
                else:
                    bbox_sampler = build_sampler(
                        self.train_cfg.rcnn.sampler, context=self)
                    post_bbox_sampler = None
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []
                for i in range(num_imgs):
                    assign_result = bbox_assigner.assign(proposal_list[i],
                                                        gt_bboxes[key_dim],
                                                        gt_bboxes_ignore[key_dim],
                                                        gt_labels[key_dim])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[key_dim],
                        gt_labels[key_dim],
                        feats=[lvl_feat[i][None] for lvl_feat in bbox_feats_all])
                    sampling_results.append(sampling_result)
                sampling_results_videos.append(sampling_results)

                # bbox head forward and loss, below equals to
                # rois_all = bbox2roi([res.bboxes for res in sampling_results])
                rois_all = []
                rois_cur = bbox2roi([res.bboxes for res in sampling_results[key_dim:key_dim+1]])
                rois_all.append(rois_cur)
                for samp_res_ in sampling_results[1:]:
                    rois_ = bbox2roi([samp_res_.bboxes])
                    rois_all.append(rois_)
                bbox_feats = []
                bbox_feats_all_split = torch.split(bbox_feats_all[0], 1, dim=0)
                del bbox_feats_all
                # print(bbox_feats_all[0].shape)
                # print(bbox_feats_all_split[0].shape)
                for i in range(num_imgs):
                    bbox_feats_ = self.get_roi_feat(bbox_feats_all_split[i:i+1], rois_all[i])
                    bbox_feats.append(bbox_feats_)
                cur_range = dict(start=key_dim, length=bbox_feats[0].shape[0])
                #Not sure for concat operation!!!!
                # print("bbox_feats[0] {}".format(bbox_feats[0].shape))
                # if bbox_feats[0].shape[0] != 128:
                #     print("Debug here!")
                bbox_feats_cat = torch.cat(bbox_feats, dim=0)
                del bbox_feats
                #After cobncate all the seperated roi_pooled features
                #[num_rois, out_channel_shared_head,roi_feat,roi_feat], 
                #the conv styled features would be fed into selsa module.
                # print("bbox_feats_cat {}".format(bbox_feats_cat.shape))
                feats.append(bbox_feats_cat)
                rois_key_all.append(rois_cur)
                cur_ranges.append(cur_range)

            #################################
            # cls_score, bbox_pred = self.bbox_head(bbox_feats_cat, cur_range)
            # The value returned should be List[] of one key frame of each video
            # cls_scores, bbox_preds = self.bbox_head(feats, cur_ranges)

            # gt_bboxes = gt_bboxes[key_video]
            # gt_labels = gt_labels[key_video]
            # sampling_results = sampling_results_videos[key_video]
            # print("cls_score {}, bbox_pred {}".format(cls_score.shape, bbox_pred.shape))
            # Below, bbox target and losses are performed

            sampling_results_for_loss = [s_res[key_dim] for s_res in sampling_results_videos]
            gt_bboxes_for_loss = [gb_res[key_dim] for gb_res in gt_bboxes_chosen_videos]
            gt_labels_for_loss = [gl_res[key_dim] for gl_res in gt_labels_chosen_videos]
            bbox_targets_key = self.bbox_head.get_target(sampling_results_for_loss,
                                                    gt_bboxes_for_loss, gt_labels_for_loss,
                                                    self.train_cfg.rcnn)
            # ! whether use dynamic or not
            all_labels = []
            for i in range(len(sampling_results_videos)):
                s_res = sampling_results_videos[i]
                gb_res = gt_bboxes_chosen_videos[i]
                gl_res = gt_labels_chosen_videos[i]
                bbox_targets = self.bbox_head.get_target(s_res,
                                                        gb_res, gl_res,
                                                        self.train_cfg.rcnn)
                all_labels.append(bbox_targets[0])

            dynamic=False
            if dynamic:
                cls_scores, bbox_preds, loss_trip, similarity_ = self.bbox_head(feats, cur_range_s=cur_ranges, others=bbox_targets_key[0], all_labels=all_labels, dynamic=True)
                if loss_trip is not None:
                    losses.update(loss_trip)
            else:
                if post_bbox_sampler is None:
                    cls_scores, bbox_preds, loss_trip, similarity_ = self.bbox_head(feats, cur_range_s=cur_ranges, others=bbox_targets_key[0], all_labels=all_labels, dynamic=False)
                else:
                    cls_scores, bbox_preds, loss_trip, similarity_, ohem_inds = self.bbox_head(feats, cur_range_s=cur_ranges, others=bbox_targets_key[0], all_labels=all_labels, dynamic=False, post_sampler=post_bbox_sampler, bbox_targets_key=bbox_targets_key)
                if loss_trip is not None:
                    losses.update(loss_trip)
            # print(len(bbox_targets))
            
            # debug=False
            # if not debug and isinstance(self.train_cfg.rcnn.sampler, list) :
            #     for i,cls_score in enumerate(cls_scores):
            #         bbox_pred = bbox_preds[i]
            #         with torch.no_grad():
            #             labels = bbox_targets_key[0]
            #             loss_cls = self.bbox_head.loss(
            #                     cls_score=[cls_score], 
            #                     bbox_pred=None, 
            #                     labels=labels,
            #                     label_weights=cls_score.new_ones(cls_score.size(0)),
            #                     bbox_targets=None,
            #                     bbox_weights=None,
            #                     reduction_override='none')['loss_cls_1']
            #             label_weights, bbox_weights, pos_inds, neg_inds = \
            #                             post_bbox_sampler.get_ohem_weights(labels,
            #                                                             bbox_targets_key[1],
            #                                                             bbox_targets_key[3],
            #                                                             loss_cls)
            #         all_inds = torch.cat([pos_inds, neg_inds], dim=0)
            #         loss_bbox = self.bbox_head.loss(
            #                     cls_score=[cls_score[all_inds]], 
            #                     bbox_pred=[bbox_pred[all_inds]], 
            #                     labels=bbox_targets_key[0][all_inds],
            #                     label_weights=label_weights[all_inds],
            #                     bbox_targets=bbox_targets_key[2][all_inds],
            #                     bbox_weights=bbox_weights[all_inds],
            #                     reduction_override=None)

            #         # loss_bbox = self.bbox_head.loss(
            #         #             cls_score=cls_score, 
            #         #             bbox_pred=bbox_pred, 
            #         #             labels=bbox_targets[0],
            #         #             label_weights=label_weights,
            #         #             bbox_targets=bbox_targets[2],
            #         #             bbox_weights=bbox_weights,
            #         #             reduction_override=None)
            #         losses['loss_cls_{}'.format(i+1)] = loss_bbox['loss_cls_1']
            #         losses['loss_bbox_{}'.format(i+1)] = loss_bbox['loss_bbox_1']
            #         losses['acc_{}'.format(i+1)] = loss_bbox['acc_1']
            # else:
            #     loss_bbox = self.bbox_head.loss(cls_scores, bbox_preds,
            #                                 *bbox_targets_key)
            #     losses.update(loss_bbox)

            if post_bbox_sampler is not None:
                if isinstance(ohem_inds, list):
                    loss_bbox = self.bbox_head.loss(cls_scores, bbox_preds,
                            labels=bbox_targets_key[0][ohem_inds],
                            label_weights=bbox_targets_key[1][ohem_inds],
                            bbox_targets=bbox_targets_key[2][ohem_inds],
                            bbox_weights=bbox_targets_key[3][ohem_inds])
                    losses.update(loss_bbox)
                else:
                    for i,ohem_inds_ in enumerate(ohem_inds):
                        loss_bbox = self.bbox_head.loss(cls_scores[i:i+1], bbox_preds[i:i+1],
                                labels=bbox_targets_key[0][ohem_inds_],
                                label_weights=bbox_targets_key[1][ohem_inds_],
                                bbox_targets=bbox_targets_key[2][ohem_inds_],
                                bbox_weights=bbox_targets_key[3][ohem_inds_])
                        losses['loss_cls_{}'.format(i+1)] = loss_bbox['loss_cls_1']
                        losses['loss_bbox_{}'.format(i+1)] = loss_bbox['loss_bbox_1']
                        losses['acc_{}'.format(i+1)] = loss_bbox['acc_1']
            else:
                loss_bbox = self.bbox_head.loss(cls_scores, bbox_preds,
                                            *bbox_targets_key)
                losses.update(loss_bbox)


            if similarity_ is None:
                return losses
            else:
                bboxes_nms=dict(labels=[],bboxes=[])
                bboxes_multi=dict(labels=[],bboxes=[])
                proposal_multi=dict(labels=[],bboxes=[])
                lens = [r['length'] for r in cur_ranges]
                for i, roi_key in enumerate(rois_key_all):
                    cls_score_cur = [c[sum(lens[:i]):sum(lens[:i+1])] for c in cls_scores]
                    bbox_pred_cur = [b[sum(lens[:i]):sum(lens[:i+1])] for b in bbox_preds]
                    img_meta_cur = img_meta_keep[i][0]
                    img_shape = img_meta_cur['img_shape']
                    scale_factor = img_meta_cur['scale_factor']
                    det_bboxes_nms, det_labels_nms = self.bbox_head.get_det_bboxes(
                        roi_key,
                        cls_score_cur,
                        bbox_pred_cur,
                        img_shape,
                        scale_factor,
                        rescale=True,
                        cfg=self.test_cfg.rcnn)
                    det_bboxes_multi, det_labels_multi = self.bbox_head.get_det_bboxes(
                        roi_key,
                        cls_score_cur,
                        bbox_pred_cur,
                        img_shape,
                        scale_factor,
                        rescale=True,
                        cfg=None)
                    det_proposal_multi, det_labels_proposal = self.bbox_head.get_det_bboxes(
                        roi_key,
                        cls_score_cur,
                        [None]*len(cls_score_cur),
                        img_shape,
                        scale_factor,
                        rescale=True,
                        cfg=None)
                    bboxes_nms['labels'].append(det_labels_nms)
                    bboxes_nms['bboxes'].append(det_bboxes_nms)
                    bboxes_multi['labels'].append(det_labels_multi)
                    bboxes_multi['bboxes'].append(det_bboxes_multi)
                    proposal_multi['labels'].append(det_labels_proposal)
                    proposal_multi['bboxes'].append(det_proposal_multi)
                for key_bbox in [bboxes_nms, bboxes_multi, proposal_multi]:
                    key_bbox['labels'] = [torch.cat(k, dim=0).cpu().numpy() for k in zip(*key_bbox['labels'])]
                    key_bbox['bboxes'] = [torch.cat(k, dim=0).cpu().numpy() for k in zip(*key_bbox['bboxes'])]
                video_paths = [i_meta['filename'] for i_meta in img_metas]
                similarity_['similarity'].update(dict(range=cur_ranges,
                                                                    labels=bbox_targets_key[0].cpu().numpy(),
                                                                    bboxes_nms=bboxes_nms,
                                                                    bboxes_multi=bboxes_multi,
                                                                    proposal_multi=proposal_multi,
                                                                    video_inds=chosen_video_ids,
                                                                    video_paths=video_paths,
                                                                    ))
                return losses, similarity_

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        # TODO: Test the shape of every tensor below and conpare with forward_train
        # The logic below is that roi opration and roi-extract operation among imgs doesn't affect each other
        key_dim = self.key_dim

        rois_all = [[] for _ in range(len(proposals))]
        for i, con_rois in enumerate(proposals):
            rois_all[i] = bbox2roi([con_rois])

        feat_strt_dim = np.sum([r.shape[0] for r in rois_all[:key_dim]])
        cur_range = [dict(start=feat_strt_dim, length=rois_all[key_dim].shape[0])]
        x_split = torch.split(x[0], 1, dim=0)
        assert len(x_split)==len(proposals)

        gap = int(len(rois_all) / 3)
        cur_range_multi = []
        for i in range(3):
            cur_range_multi.append(np.sum([r.shape[0] for r in rois_all[i*gap: (i+1)*gap]]))

        roi_feats_list = []
        for i,_x in enumerate(x_split):
            roi_feats_list.append(self.get_roi_feat([_x], rois_all[i]))
        roi_feats = torch.cat(roi_feats_list, dim=0)
        
        # cls_score, bbox_pred = self.bbox_head.forward_test_multi_passes(roi_feats, cur_range_multi, cur_range, key_dim)
        cls_score, bbox_pred = self.bbox_head.forward_test(roi_feats, cur_range, key_dim=key_dim, all_res=False)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois_all[key_dim],
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        if hasattr(self.test_cfg, 'rcnn'):
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        else:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_meta, proposal_list, None, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[self.key_dim]['img_shape']
            scale_factor = img_meta[self.key_dim]['scale_factor']
            flip = [meta_['flip'] for meta_ in img_meta]
            # TODO more flexible

            proposals = [bbox_mapping(proposal_[:, :4], img_shape,
                        scale_factor, flip_) for flip_, proposal_ in zip(flip,proposal_list)]

            bboxes, scores = self.simple_test_bboxes(x, img_meta, proposals,
                                        None, rescale=False)
            # rois = bbox2roi([proposals])
            # # recompute feature maps to save GPU memory
            # if self.feat_from_shared_head:
            #     # print("enter feat_from_shared_head, type of x is {}".format(type(x)))
            #     shared_head_feat_ = [self.shared_head(x[0])]
            #     roi_feats = self.bbox_roi_extractor(shared_head_feat_[:1], rois)
            # else:
            #     roi_feats = self.bbox_roi_extractor(
            #         x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            #     if self.with_shared_head:
            #         roi_feats = self.shared_head(roi_feats)
            # cls_score, bbox_pred = self.bbox_head(roi_feats)
            # bboxes, scores = self.bbox_head.get_det_bboxes(
            #     rois,
            #     cls_score,
            #     bbox_pred,
            #     img_shape,
            #     scale_factor,
            #     rescale=False,
            #     cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        det_bboxes_collect = []
        det_labels_collect = []
        # after merging, bboxes will be rescaled to the original image size
        for i in range(len(aug_scores[0])):
            a_bboxes = [a_b[i] for a_b in aug_bboxes]
            a_scores = [a_s[i] for a_s in aug_scores]
            metas_ = [[meta_[self.key_dim]] for meta_ in img_metas]
            merged_bboxes, merged_scores = merge_aug_bboxes(
                a_bboxes, a_scores, metas_, rcnn_test_cfg)
            if hasattr(rcnn_test_cfg, 'nms'):
                det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                        rcnn_test_cfg.score_thr,
                                                        rcnn_test_cfg.nms,
                                                        rcnn_test_cfg.max_per_img)
                det_bboxes_collect.append(det_bboxes)
                det_labels_collect.append(det_labels)
            else:
                det_bboxes_collect.append(merged_bboxes)
                det_labels_collect.append(merged_scores)
                
        return det_bboxes_collect, det_labels_collect