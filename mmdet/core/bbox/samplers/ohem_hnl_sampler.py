import torch

from ..transforms import bbox2roi
from .base_sampler import BaseSampler


class OHEMHNLSampler(BaseSampler):
    """
    Online Hard Example Mining Sampler described in [1]_.

    References:
        .. [1] https://arxiv.org/pdf/1604.03540.pdf
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 context,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        super(OHEMHNLSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                          add_gt_as_proposals)
        if not hasattr(context, 'num_stages'):
            self.bbox_roi_extractor = context.bbox_roi_extractor
            self.bbox_head = context.bbox_head
        else:
            self.bbox_roi_extractor = context.bbox_roi_extractor[
                context.current_stage]
            self.bbox_head = context.bbox_head[context.current_stage]

    def hard_mining(self, inds, num_expected, bboxes, labels, feats):
        with torch.no_grad():
            rois = bbox2roi([bboxes])
            bbox_feats = self.bbox_roi_extractor(
                feats[:self.bbox_roi_extractor.num_inputs], rois)
            cur_range = dict(start=0, length=bbox_feats[0].shape[0])
            cls_score, _ = self.bbox_head(bbox_feats, cur_range)
            loss = self.bbox_head.loss(
                cls_score=cls_score,
                bbox_pred=None,
                labels=labels,
                label_weights=cls_score.new_ones(cls_score.size(0)),
                bbox_targets=None,
                bbox_weights=None,
                reduction_override='none')['loss_cls']
            _, topk_loss_inds = loss.topk(num_expected)
        return inds[topk_loss_inds]

    def hard_mining_loss(self, inds, num_expected, loss):
            with torch.no_grad():
                _, topk_loss_inds = loss[inds].topk(num_expected)
            return inds[topk_loss_inds]

    def _sample_pos(self,
                    labels,
                    num_expected,
                    loss,
                    **kwargs):
        # Sample some hard positive samples
        pos_inds = torch.nonzero(labels > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.hard_mining_loss(pos_inds, num_expected, loss)

    def _sample_neg(self,
                    labels,
                    num_expected,
                    loss,
                    **kwargs):
        # Sample some hard negative samples
        neg_inds = torch.nonzero(labels == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.hard_mining_loss(neg_inds, num_expected, loss)
    
    def get_ohem_weights(self,
                        labels,
                        label_weights,
                        bbox_weights,
                        loss):
        """

        """
        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
                        labels, num_expected_pos, loss)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
                    labels, num_expected_neg, loss)
        neg_inds = neg_inds.unique()

        label_weights[...]=0.
        label_weights[pos_inds]=1.0
        label_weights[neg_inds]=1.0

        bbox_weights[...]=0
        bbox_weights[pos_inds]=1.0

        return label_weights, bbox_weights, pos_inds, neg_inds