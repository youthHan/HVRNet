import logging

import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmdet.core import auto_fp16
from ..backbones import Res2Net, make_res2_layer
from ..registry import SHARED_HEADS
from ..utils import ConvModule


@SHARED_HEADS.register_module
class Res2Layer(nn.Module):

    def __init__(self,
                 depth,
                 stage=3,
                 stride=2,
                 dilation=1,
                 style='pytorch',
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 external_conv=False,
                 dcn=None):
        super(Res2Layer, self).__init__()
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.stage = stage
        self.fp16_enabled = False
        self.external_conv = external_conv
        block, stage_blocks = Res2Net.arch_settings[depth]
        stage_block = stage_blocks[stage]
        planes = 64 * 2**stage
        inplanes = 64 * 2**(stage - 1) * block.expansion
        baseWidth = 26
        scale = 4
        
        res_layer = make_res2_layer(
            block,
            inplanes,
            planes,
            stage_block,
            baseWidth=baseWidth,
            scale=scale,
            stride=stride,
            dilation=dilation)
        self.add_module('layer{}'.format(stage + 1), res_layer)
        if external_conv:
            new_layer = ConvModule(2048,256,1)
            self.add_module('new_layer_1', new_layer)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    @auto_fp16()
    def forward(self, x):
        res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
        out = res_layer(x)
        if self.external_conv:
            new_layer_1 = getattr(self, 'new_layer_1')
            out = new_layer_1(out)
        return out

    def train(self, mode=True):
        super(Res2Layer, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
