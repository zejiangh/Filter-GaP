# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys

__all__ = ["build_resnet", "resnet_versions", "resnet_configs", "resnet_official"]

# ========================================= self-defined resnet50 =============================================#

'''
ResNet-B verision
'''

class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class resnet_official(nn.Module):
    def __init__(self, cfg=None):
        super(resnet_official, self).__init__()
        block = Bottleneck
        if cfg is None:
            cfg = [[64, 64, 64], [256, 64, 64] * 2, [256, 128, 128], [512, 128, 128] * 3, [512, 256, 256], [1024, 256, 256] * 5, [1024, 512, 512], [2048, 512, 512] * 2, [2048]]
            cfg = [item for sub_list in cfg for item in sub_list]
            assert len(cfg) == 49, "Length of cfg_official is not right"
        else:
            cfg = [[64, 128, 128], [256, 128, 128] * 2, [256, 256, 256], [512, 256, 256] * 3, [512, 512, 512], [1024, 512, 512] * 5, [1024, 1024, 1024], [2048, 1024, 1024] * 2, [2048]]
            cfg = [item for sub_list in cfg for item in sub_list]
            assert len(cfg) == 49, "Length of cfg_official is not right"
        self.cfg=cfg
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, cfg[0:9], 256, 3)
        self.layer2 = self._make_layer(block, cfg[9:21], 512, 4, stride=2)
        self.layer3 = self._make_layer(block, cfg[21:39], 1024, 6, stride=2)
        self.layer4 = self._make_layer(block, cfg[39:48], 2048, 3, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(cfg[-1], 1000)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, cfg, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i:3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
                
# ================================================================================================

resnet_configs = {
    "classic": {
        "conv": nn.Conv2d,
        "conv_init": "fan_out",
        "nonlinearity": "relu",
        "last_bn_0_init": False,
        "activation": lambda: nn.ReLU(inplace=True),
    },
    "fanin": {
        "conv": nn.Conv2d,
        "conv_init": "fan_in",
        "nonlinearity": "relu",
        "last_bn_0_init": False,
        "activation": lambda: nn.ReLU(inplace=True),
    },
    "grp-fanin": {
        "conv": nn.Conv2d,
        "conv_init": "fan_in",
        "nonlinearity": "relu",
        "last_bn_0_init": False,
        "activation": lambda: nn.ReLU(inplace=True),
    },
    "grp-fanout": {
        "conv": nn.Conv2d,
        "conv_init": "fan_out",
        "nonlinearity": "relu",
        "last_bn_0_init": False,
        "activation": lambda: nn.ReLU(inplace=True),
    },
}

resnet_versions = {
    "resnet50": {
        "net": resnet_official,
        "block": Bottleneck,
        "layers": [3, 4, 6, 3],
        "widths": [64, 128, 256, 512],
        "expansion": 4,
    },
}

def build_resnet(version, config, num_classes, cfg=None, verbose=True):
    print('>============================== SELF-DEFINED MODELS ====================================<')
    if version == "resnet50":
        model = resnet_official(cfg=cfg)
    else:
        print('no model found')
        sys.exit()
    version = resnet_versions[version]
    config = resnet_configs[config]
    if verbose:
        print("Version: {}".format(version))
        print("Config: {}".format(config))
        print("Num classes: {}".format(num_classes))

    return model
