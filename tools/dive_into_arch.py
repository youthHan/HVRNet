import mmcv
import torch

from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmcv.runner import obj_from_dict

def get_detector(config, device='cpu'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not instance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config obj,'
                        'but got {}'.format(type(config)))
    model = build_detector(config.model, train_cfg=config.train_cfg, test_cfg=config.test_cfg)
    model.cfg = config
    model.to(device)
    print(model)
    return model


def build_optimizer(model, optimizer_cfg, mod=False):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if mod:        
        base_lr = optimizer_cfg['lr']
        ft_lr = base_lr #/ 10.
        # frozen_params_list = [model.backbone.parameters(),
        #                model.neck.parameters(),
        #                model.rpn_head.parameters(),
        #                model.bbox_roi_extractor.parameters(),
        #                model.bbox_head.shared_fcs.parameters()]
        params = [{'params':model.backbone.parameters(), 'lr':ft_lr},
                  {'params':model.neck.parameters(), 'lr':ft_lr},
                  {'params':model.rpn_head.parameters(), 'lr':ft_lr},
                  {'params':model.bbox_roi_extractor.parameters(), 'lr':ft_lr},
                  {'params':model.bbox_head.shared_fcs.parameters(), 'lr':base_lr},
                  {'params':model.bbox_head.fc_cls.parameters(), 'lr':base_lr},
                  {'params':model.bbox_head.fc_reg.parameters(), 'lr':base_lr}
                    ]
        # for frozen_params in frozen_params_list:
        #     for p in frozen_params:
        #         p.requires_grad = False
        # params = [{'params':model.bbox_head.fc_cls.parameters(), 'lr':base_lr},
        #           {'params':model.bbox_head.fc_reg.parameters(), 'lr':base_lr}]
        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)

    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, torch.optim,
                            dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
                or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)

def vis_model(model, input_size=[1,3,224,224]):
    import tensorwatch as tw
    drawing = tw.draw_model(model, [1, 3, 224, 224])
    drawing.save('model_distillation.pdf')


if __name__ == '__main__':
    model = get_detector('./configs/faster_rcnn_r101_1x_vid_distillation_version.py')
    # datasets = [build_dataset(model.cfg.data.train)]
    # optimizer = build_optimizer(model, model.cfg.optimizer)
    # print(model)
    # vis_model(model,[1,3,1000,600])