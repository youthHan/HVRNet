import collections
import numpy as np

import torch
import torch.nn as nn

from ..registry import DETECTORS
from .two_stage import TwoStageDetector


from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
@DETECTORS.register_module
class SelsaRCNN(TwoStageDetector):

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
        super(SelsaRCNN, self).__init__(
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
            self.key_dim = int(self.test_cfg.relation_setup.frame_interval)
            self.bbox_head.t_dim = int(test_cfg.bbox_head.t_dim)
            self.bbox_head.sampler_num = int(test_cfg.bbox_head.sampler_num)

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
        # assert x is not None and img_meta is not None
        if isinstance(x, collections.Sequence):
            # assert len(x)==len(img_meta.data[0])
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

        det_bboxes, det_labels = self.simple_test_bboxes(
            bbox_feats_all, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def forward_train(self,
                        img,
                        img_meta,
                        gt_bboxes,
                        gt_labels,
                        gt_bboxes_ignore=None,
                        gt_masks=None,
                        proposals=None):
            """
            Args:
                img (Tensor): of shape (N, C, H, W) encoding input images.
                    Typically these should be mean centered and std scaled.

                img_meta (list[dict]): list of image info dict where each dict has:
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
            key_dim = self.key_dim
            x = self.extract_feat(img)

            losses = dict()

            # RPN forward and loss
            #TODO: !!!!!!Caution: RPN Loss only with the reference frame!!!!!!
            if self.with_rpn:
                rpn_outs = self.rpn_head(x)
                rpn_outs_split = [torch.split(r_out[0],1,dim=0)[key_dim:key_dim+1] for r_out in rpn_outs]
                ###########################
                rpn_loss_inputs = tuple(rpn_outs_split) + (gt_bboxes[key_dim:key_dim+1], img_meta[key_dim:key_dim+1], self.train_cfg.rpn)
                rpn_losses = self.rpn_head.loss(
                    *rpn_loss_inputs, gt_bboxes_ignore=None if gt_bboxes_ignore is None else gt_bboxes_ignore[key_dim])
                ############################
                losses.update(rpn_losses)

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

            # assign gts and sample proposals
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            if isinstance(self.train_cfg.rcnn.sampler, list):
                bbox_sampler, post_bbox_sampler = build_sampler(
                                    self.train_cfg.rcnn.sampler, context=self)
            else:
                bbox_sampler = build_sampler(
                    self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
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
            # print(bbox_feats_all[0].shape)
            # print(bbox_feats_all_split[0].shape)
            for i in range(img.size(0)):
                bbox_feats_ = self.get_roi_feat(bbox_feats_all_split[i:i+1], rois_all[i])
                bbox_feats.append(bbox_feats_)
            cur_range = dict(start=key_dim*bbox_feats[0].shape[0], length=bbox_feats[0].shape[0])
            #Not sure for concat operation!!!!
            # print("bbox_feats[0] {}".format(bbox_feats[0].shape))
            # if bbox_feats[0].shape[0] != 128:
            #     print("Debug here!")
            bbox_feats_cat = torch.cat(bbox_feats, dim=0)
            #After cobncate all the seperated roi_pooled features
            #[num_rois, out_channel_shared_head,roi_feat,roi_feat], 
            #the conv styled features would be fed into selsa module.
            # print("bbox_feats_cat {}".format(bbox_feats_cat.shape))
            #################################
            cls_score, bbox_pred, similarity_ = self.bbox_head(bbox_feats_cat, cur_range)
            # print("cls_score {}, bbox_pred {}".format(cls_score.shape, bbox_pred.shape))
            # Below, bbox target and losses are performed
            bbox_targets = self.bbox_head.get_target(sampling_results[key_dim:key_dim+1],
                                                    gt_bboxes[key_dim:key_dim+1], gt_labels[key_dim:key_dim+1],
                                                    self.train_cfg.rcnn)
            # print(len(bbox_targets))
            if isinstance(self.train_cfg.rcnn.sampler, list):
                with torch.no_grad():
                    labels = bbox_targets[0]
                    loss_cls = self.bbox_head.loss(
                            cls_score=cls_score, 
                            bbox_pred=None, 
                            labels=labels,
                            label_weights=cls_score.new_ones(cls_score.size(0)),
                            bbox_targets=None,
                            bbox_weights=None,
                            reduction_override='none')['loss_cls']
                    label_weights, bbox_weights, pos_inds, neg_inds = \
                                    post_bbox_sampler.get_ohem_weights(labels,
                                                                    bbox_targets[1],
                                                                    bbox_targets[3],
                                                                    loss_cls)
                all_inds = torch.cat([pos_inds, neg_inds], dim=0)
                loss_bbox = self.bbox_head.loss(
                            cls_score=cls_score[all_inds], 
                            bbox_pred=bbox_pred[all_inds], 
                            labels=bbox_targets[0][all_inds],
                            label_weights=label_weights[all_inds],
                            bbox_targets=bbox_targets[2][all_inds],
                            bbox_weights=bbox_weights[all_inds],
                            reduction_override=None)

                # loss_bbox = self.bbox_head.loss(
                #             cls_score=cls_score, 
                #             bbox_pred=bbox_pred, 
                #             labels=bbox_targets[0],
                #             label_weights=label_weights,
                #             bbox_targets=bbox_targets[2],
                #             bbox_weights=bbox_weights,
                #             reduction_override=None)
            else:
                
                loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

            if similarity_ is None:
                return losses
            else:
                for i, roi_ in enumerate(rois_all):
                    img_shape = img_meta['img_shape']
                    scale_factor = img_meta['scale_factor']
                    if i == self.key_dim:
                        det_bboxes_nms, det_labels_nms = self.bbox_head.get_det_bboxes(
                            roi_,
                            cls_score_cur,
                            bbox_pred_cur,
                            img_shape,
                            scale_factor,
                            rescale=True,
                            cfg=self.test_cfg.rcnn)
                        det_bboxes_multi, det_labels_multi = self.bbox_head.get_det_bboxes(
                            roi_,
                            cls_score_cur,
                            bbox_pred_cur,
                            img_shape,
                            scale_factor,
                            rescale=True,
                            cfg=None)
                    det_proposal_multi, det_labels_proposal = self.bbox_head.get_det_bboxes(
                        roi_,
                        cls_score_cur,
                        [None]*len(cls_score_cur),
                        img_shape,
                        scale_factor,
                        rescale=True,
                        cfg=None)
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
        cur_range = dict(start=feat_strt_dim, length=rois_all[key_dim].shape[0])
        x_split = torch.split(x[0], 1, dim=0)
        assert len(x_split)==len(proposals)

        roi_feats_list = []
        for i,_x in enumerate(x_split):
            roi_feats_list.append(self.get_roi_feat([_x], rois_all[i]))
        roi_feats = torch.cat(roi_feats_list, dim=0)

        cls_score, bbox_pred = self.bbox_head(roi_feats, cur_range, key_dim=key_dim, all_res=False)
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

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results
