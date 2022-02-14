import mmcv
import numpy as np
from terminaltables import AsciiTable

from .bbox_overlaps import bbox_overlaps
from .class_names import get_classes


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_imagenet(det_bboxes,
                  gt_bboxes,
                  gt_ignore,
                  default_iou_thr,
                  area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): the detected bbox
        gt_bboxes (ndarray): ground truth bboxes of this image
        gt_ignore (ndarray): indicate if gts are ignored for evaluation or not
        default_iou_thr (float): the iou thresholds for medium and large bboxes
        area_ranges (list or None): gt bbox area ranges

    Returns:
        tuple: two arrays (tp, fp) whose elements are 0 and 1
    """
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp
    # of a certain scale.
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + 1) * (
                det_bboxes[:, 3] - det_bboxes[:, 1] + 1)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp
    ious = bbox_overlaps(det_bboxes, gt_bboxes - 1)
    gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1
    gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1
    iou_thrs = np.minimum((gt_w * gt_h) / ((gt_w + 10.0) * (gt_h + 10.0)),
                          default_iou_thr)
    # sort all detections by scores in descending order
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore, dtype=bool)
        else:
            gt_areas = gt_w * gt_h
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            max_iou = -1
            matched_gt = -1
            # find best overlapped available gt
            for j in range(num_gts):
                # different from PASCAL VOC: allow finding other gts if the
                # best overlaped ones are already matched by other det bboxes
                if gt_covered[j]:
                    continue
                elif ious[i, j] >= iou_thrs[j] and ious[i, j] > max_iou:
                    max_iou = ious[i, j]
                    matched_gt = j
            # there are 4 cases for a det bbox:
            # 1. it matches a gt, tp = 1, fp = 0
            # 2. it matches an ignored gt, tp = 0, fp = 0
            # 3. it matches no gt and within area range, tp = 0, fp = 1
            # 4. it matches no gt but is beyond area range, tp = 0, fp = 0
            if matched_gt >= 0:
                gt_covered[matched_gt] = 1
                if not (gt_ignore[matched_gt] or gt_area_ignore[matched_gt]):
                    tp[k, i] = 1
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp

def tpfp_analysis(det_bboxes, gt_bboxes, other_gt_bboxes, gt_ignore, iou_thr, area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): the detected bbox
        gt_bboxes (ndarray): ground truth bboxes of this image
        gt_ignore (ndarray): indicate if gts are ignored for evaluation or not
        iou_thr (float): the iou thresholds

    Returns:
        tuple: (tp, fp), two arrays whose elements are 0 and 1
    """
    assert area_ranges is None, "The analysis is only implemented for the results as a whole"
    # * For analysis, following variables are built
    # ! The analysis is performed only when area_ranges are none
    loc_error_1 = 0
    loc_error_2 = 0
    cls_error = 0
    # lc_error = 0
    bg_error = 0
    fn = 0

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    if len(det_bboxes) == 0:
        return tp, fp, [loc_error_1, loc_error_2, cls_error, bg_error, gt_bboxes.shape[0]]

    if other_gt_bboxes.shape[0] == 0:
        other_ious = None
    else:
        other_ious = bbox_overlaps(det_bboxes, other_gt_bboxes)
        other_ious_max = other_ious.max(axis=1)
        other_ious_argmax = other_ious.argmax(axis=1)
    other_sort_inds = np.argsort(-det_bboxes[:, -1])
    
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        fp[...] = 1
        
        for i in other_sort_inds:
            if other_ious is not None:
                if other_ious_max[i] >= 0.1:
                    cls_error += 1
                else:
                    bg_error += 1
            else:
                bg_error += 1
        return tp, fp, [loc_error_1, loc_error_2, cls_error, bg_error, 0]

    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        gt_area_ignore = np.zeros_like(gt_ignore, dtype=bool)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                        loc_error_1 += 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            else:
                fp[k, i] = 1
                if ious_max[i] >= 0.1:
                    loc_error_2 += 1
                else:
                    if other_ious is None:
                        bg_error += 1
                    else:
                        if other_ious_max[i] >= 0.1:
                            cls_error += 1
                        else:
                            bg_error += 1
    fn = num_gts - np.sum(gt_covered)
    return tp, fp, [loc_error_1, loc_error_2, cls_error, bg_error, fn]


def analysis_map(det_results,
             gt_bboxes,
             gt_labels,
             gt_ignore=None,
             scale_ranges=None,
             iou_thr=0.5,
             dataset=None,
             file_dir=None,
             print_summary=True):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list): a list of list, [[cls1_det, cls2_det, ...], ...]
        gt_bboxes (list): ground truth bboxes of each image, a list of K*4
            array.
        gt_labels (list): ground truth labels of each image, a list of K array
        gt_ignore (list): gt ignore indicators of each image, a list of K array
        scale_ranges (list, optional): [(min1, max1), (min2, max2), ...]
        iou_thr (float): IoU threshold
        dataset (None or str or list): dataset name or dataset classes, there
            are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc.
        print_summary (bool): whether to print the mAP summary

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(gt_bboxes) == len(gt_labels)
    assert type(dataset) in (tuple, list), \
                "Note: Parameter `dataset` should be a list containing all the class names"
    if gt_ignore is not None:
        assert len(gt_ignore) == len(gt_labels)
        for i in range(len(gt_ignore)):
            assert len(gt_labels[i]) == len(gt_ignore[i])
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    eval_results = []
    num_classes = len(det_results[0])  # positive class num
    gt_labels = [
        label if label.ndim == 1 else label[:, 0] for label in gt_labels
    ]
    all_analysis_res = []
    meta_analysis_res = []
    det_bbox_stack_res = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gt_ignore = get_cls_results(
            det_results, gt_bboxes, gt_labels, gt_ignore, i)
        other_cls_gts, _ = get_other_cls_results(
            det_results, gt_bboxes, gt_labels, gt_ignore, i)
        # calculate tp and fp for each image
        tpfp_func = (tpfp_analysis)
        tpfp = [
            tpfp_func(cls_dets[j], cls_gts[j], other_cls_gts[j], cls_gt_ignore[j], iou_thr,
                      area_ranges) for j in range(len(cls_dets))
        ]
        tp, fp, analysis = tuple(zip(*tpfp))
        analysis_res = tuple(zip(*analysis))

        # calculate gt number of each scale, gts ignored or beyond scale
        # are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += np.sum(np.logical_not(cls_gt_ignore[j]))
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (
                    bbox[:, 3] - bbox[:, 1] + 1)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum(
                        np.logical_not(cls_gt_ignore[j])
                        & (gt_areas >= min_area) & (gt_areas < max_area))
        
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        det_scores = cls_dets[sort_inds, -1]
        meta_analysis_res.append(dict(tp=tp, fp=fp, num_gts=num_gts, det_scores=det_scores, sort_inds=sort_inds))
        det_bbox_stack_res.append(cls_dets[sort_inds])

        all_analysis_res.append([num_gts.item(), cls_dets.shape[0], sum(tp[0]), sum(fp[0])])
        all_analysis_res[i].extend([sum(res) for res in analysis_res])

        assert sum([sum(a) for a in analysis_res[:4]]) == sum(fp[0])

        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if dataset != 'voc07' else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
        all_analysis_res[i].extend([ap])

    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0
    if print_summary:
        print_map_summary(mean_ap, eval_results, dataset, area_ranges)

    if file_dir is not None:
        import os.path as osp
        # mmcv.dump(all_analysis_res, osp.join(file_dir, 'hnl_epoch4_2018_analysis_res_meta.pkl'))
        mmcv.dump(meta_analysis_res, osp.join(file_dir, 'hnmb_mining_meta_analysis.pkl'))
        with open(osp.join(file_dir, 'hnmb_mining_format_analysis_res.txt'), 'w') as h:
            h.writelines(' ,num_gts,num_dets,tp,fp,loc_error_1,loc_error_2,cls_error,bg_error,fn,ap\n')
            # h.writelines(' ,tp,fp,num_gts,num_dets,\
            #                 loc_error_1,loc_error_2,cls_error,bg_error,fn,recall,precision,ap\n')
            [h.writelines(dataset[i]+','+','.join(list(map(str,res)))+'\n') for i,res in enumerate(all_analysis_res)]
            # last_line = 'mAP' + ','.join([' ']*(len(all_analysis_res[0])-1))+'{}'
            # h.writelines(last_line.format(mean_ap))
        

    return mean_ap, eval_results, all_analysis_res


def tpfp_default(det_bboxes, gt_bboxes, gt_ignore, iou_thr, area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): the detected bbox
        gt_bboxes (ndarray): ground truth bboxes of this image
        gt_ignore (ndarray): indicate if gts are ignored for evaluation or not
        iou_thr (float): the iou thresholds

    Returns:
        tuple: (tp, fp), two arrays whose elements are 0 and 1
    """
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + 1) * (
                det_bboxes[:, 3] - det_bboxes[:, 1] + 1)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp
    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def get_cls_results(det_results, gt_bboxes, gt_labels, gt_ignore, class_id):
    """Get det results and gt information of a certain class."""
    cls_dets = [det[class_id]
                for det in det_results]  # det bboxes of this class
    cls_gts = []  # gt bboxes of this class
    cls_gt_ignore = []
    for j in range(len(gt_bboxes)):
        gt_bbox = gt_bboxes[j]
        cls_inds = (gt_labels[j] == class_id + 1)
        cls_gt = gt_bbox[cls_inds, :] if gt_bbox.shape[0] > 0 else gt_bbox
        cls_gts.append(cls_gt)
        if gt_ignore is None:
            cls_gt_ignore.append(np.zeros(cls_gt.shape[0], dtype=np.int32))
        else:
            cls_gt_ignore.append(gt_ignore[j][cls_inds])
    return cls_dets, cls_gts, cls_gt_ignore


def get_other_cls_results(det_results, gt_bboxes, gt_labels, gt_ignore, class_id):
    """Get gt information of classes exclude a certain class."""
    cls_gts = []  # gt bboxes of this class
    cls_gt_ignore = []
    for j in range(len(gt_bboxes)):
        gt_bbox = gt_bboxes[j]
        cls_inds = (gt_labels[j] != class_id + 1)
        cls_gt = gt_bbox[cls_inds, :] if gt_bbox.shape[0] > 0 else gt_bbox
        cls_gts.append(cls_gt)
        if gt_ignore is None:
            cls_gt_ignore.append(np.zeros(cls_gt.shape[0], dtype=np.int32))
        else:
            cls_gt_ignore.append(gt_ignore[j][cls_inds])
    return cls_gts, cls_gt_ignore


def eval_map(det_results,
             gt_bboxes,
             gt_labels,
             gt_ignore=None,
             scale_ranges=None,
             iou_thr=0.5,
             dataset=None,
             print_summary=True):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list): a list of list, [[cls1_det, cls2_det, ...], ...]
        gt_bboxes (list): ground truth bboxes of each image, a list of K*4
            array.
        gt_labels (list): ground truth labels of each image, a list of K array
        gt_ignore (list): gt ignore indicators of each image, a list of K array
        scale_ranges (list, optional): [(min1, max1), (min2, max2), ...]
        iou_thr (float): IoU threshold
        dataset (None or str or list): dataset name or dataset classes, there
            are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc.
        print_summary (bool): whether to print the mAP summary

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(gt_bboxes) == len(gt_labels)
    if gt_ignore is not None:
        assert len(gt_ignore) == len(gt_labels)
        for i in range(len(gt_ignore)):
            assert len(gt_labels[i]) == len(gt_ignore[i])
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    eval_results = []
    num_classes = len(det_results[0])  # positive class num
    gt_labels = [
        label if label.ndim == 1 else label[:, 0] for label in gt_labels
    ]
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gt_ignore = get_cls_results(
            det_results, gt_bboxes, gt_labels, gt_ignore, i)
        # calculate tp and fp for each image
        tpfp_func = (
            tpfp_imagenet if dataset in ['det', 'vid'] else tpfp_default)
        tpfp = [
            tpfp_func(cls_dets[j], cls_gts[j], cls_gt_ignore[j], iou_thr,
                      area_ranges) for j in range(len(cls_dets))
        ]
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale, gts ignored or beyond scale
        # are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += np.sum(np.logical_not(cls_gt_ignore[j]))
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (
                    bbox[:, 3] - bbox[:, 1] + 1)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum(
                        np.logical_not(cls_gt_ignore[j])
                        & (gt_areas >= min_area) & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if dataset != 'voc07' else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0
    if print_summary:
        print_map_summary(mean_ap, eval_results, dataset, area_ranges)

    return mean_ap, eval_results


def print_map_summary(mean_ap, results, dataset=None, ranges=None):
    """Print mAP and results of each class.

    Args:
        mean_ap(float): calculated from `eval_map`
        results(list): calculated from `eval_map`
        dataset(None or str or list): dataset name or dataset classes.
        ranges(list or Tuple): ranges of areas
    """
    num_scales = len(results[0]['ap']) if isinstance(results[0]['ap'],
                                                     np.ndarray) else 1
    if ranges is not None:
        assert len(ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    precisions = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
            precisions[:, i] = np.array(
                cls_result['precision'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(1, num_classes + 1)]
    elif mmcv.is_str(dataset):
        label_names = get_classes(dataset)
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]
    header = ['class', 'gts', 'dets', 'recall', 'precision', 'ap']
    for i in range(num_scales):
        if ranges is not None:
            print("Area range ", ranges[i])
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                '{:.3f}'.format(recalls[i, j]),
                '{:.3f}'.format(precisions[i, j]), '{:.3f}'.format(aps[i, j])
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', '', '{:.3f}'.format(mean_ap[i])])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print(table.table)
