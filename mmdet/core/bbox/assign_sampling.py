import mmcv

from . import assigners, samplers


def build_assigner(cfg, **kwargs):
    if isinstance(cfg, assigners.BaseAssigner):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, assigners, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(
            type(cfg)))


def build_sampler(cfg, **kwargs):
    if isinstance(cfg, samplers.BaseSampler):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, samplers, default_args=kwargs)
    elif isinstance(cfg, list):
        flags = [isinstance(c, dict) for c in cfg]
        if sum(flags) == len(cfg):
            return [mmcv.runner.obj_from_dict(c, samplers, default_args=kwargs)
                                for c in cfg]
        else:
            raise TypeError('Invalid element in `list` type configs for sampler builder')
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(
            type(cfg)))


def assign_and_sample(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels, cfg):
    bbox_assigner = build_assigner(cfg.assigner)
    bbox_sampler = build_sampler(cfg.sampler)
    assign_result = bbox_assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore,
                                         gt_labels)
    sampling_result = bbox_sampler.sample(assign_result, bboxes, gt_bboxes,
                                          gt_labels)
    return assign_result, sampling_result
