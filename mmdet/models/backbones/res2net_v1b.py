import logging

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.runner import load_checkpoint
from ..registry import BACKBONES

__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b']


model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal',dilation=1):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()
        self.dilation = dilation

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, 
                                        padding=self.dilation, dilation=self.dilation, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def make_res2_layer(block, inplanes, planes, blocks, baseWidth, scale, stride=1, dilation=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=stride, stride=stride, 
                ceil_mode=True, count_include_pad=False),
            nn.Conv2d(inplanes, planes * block.expansion, 
                kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample=downsample, 
                    stype='stage', baseWidth = baseWidth, scale=scale))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, baseWidth = baseWidth, 
                                            scale=scale, dilation=dilation))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class Res2Net(nn.Module):
    
    arch_settings = {
        101: (Bottle2neck, (3, 4, 23, 3))
    }
    def __init__(self,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True):
        

        super(Res2Net, self).__init__()
        block = Bottle2neck
        layers = [3, 4, 23, 3]
        baseWidth = 26
        scale = 4
        
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        #### ! ############
        self.strides = strides
        #### ! ############
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        # ! make_stem_layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # self.res_layers = []
        self.layer1 = make_res2_layer(block, self.inplanes, 64, layers[0], 
                                        baseWidth=self.baseWidth, scale=self.scale)
        self.inplanes = 64 * block.expansion
        self.layer2 = make_res2_layer(block, self.inplanes, 128, layers[1], stride=2, 
                                        baseWidth=self.baseWidth, scale=self.scale)
        self.inplanes = 128 * block.expansion
        self.layer3 = make_res2_layer(block, self.inplanes, 256, layers[2], stride=2, 
                                        baseWidth=self.baseWidth, scale=self.scale)
        self.inplanes = 256 * block.expansion
        # ! omit conv5
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.res_layers.append(self.layer1)
        # self.res_layers.append(self.layer2)
        # self.res_layers.append(self.layer3)
        # self.res_layers.append(self.layer4)
        
        # self.res_layers.append(make_res2_layer(block, 64, layers[0]))
        # self.res_layers.append(make_res2_layer(block, 128, layers[1], stride=2))
        # self.res_layers.append(make_res2_layer(block, 256, layers[2], stride=2))
        # self.res_layers.append(make_res2_layer(block, 512, layers[3], stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._freeze_stages()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return tuple([x])

    def train(self, mode=True):
        super(Res2Net, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


def res2net101_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth = 26, scale = 4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
    return model

def res2net101_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth = 26, scale = 4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
    return model



# if __name__ == '__main__':
#     images = torch.rand(1, 3, 224, 224).cuda(0)
#     model = res2net50_v1b_26w_4s(pretrained=True)
#     model = model.cuda(0)
#     print(model(images).size())