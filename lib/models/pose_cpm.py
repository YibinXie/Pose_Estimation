# ------------------------------------------------------------------------------
# https://zhuanlan.zhihu.com/p/45002720 Hourglass network structure
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride,
        padding=0, bias=False
    )


def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=5, stride=stride,
        padding=2, bias=False
    )


def conv9x9(in_planes, out_planes, stride=1):
    """9x9 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=9, stride=stride,
        padding=4, bias=False
    )


def conv11x11(in_planes, out_planes, stride=1):
    """11x11 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=11, stride=stride,
        padding=5, bias=False
    )


def maxpool3x3(stride=2):
    """3x3 maxpooling with padding"""
    return nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)


class FirstStage(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, out_channels, stride=1):
        super(FirstStage, self).__init__()
        self.out_channels = out_channels

        # planes = 128
        self.conv1 = conv9x9(3, planes)
        self.pool1 = maxpool3x3()
        self.conv2 = conv9x9(planes, planes)
        self.pool2 = maxpool3x3()
        self.conv3 = conv9x9(planes, planes)
        self.pool3 = maxpool3x3()
        self.conv4 = conv9x9(planes, planes)
        self.pool4 = maxpool3x3()
        self.conv5 = conv9x9(planes, planes)
        self.pool5 = maxpool3x3()
        self.deconv1 = nn.ConvTranspose2d(planes, planes, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(planes, planes, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(planes, planes, 4, stride=2, padding=1)
        self.conv6 = conv5x5(planes, planes // self.expansion)  # Unusual operations, get less channels in process
        self.conv7 = conv9x9(planes // self.expansion, planes * self.expansion)  # 32 -> 512 big step
        self.conv8 = conv1x1(planes * self.expansion, planes * self.expansion)
        self.conv9 = conv1x1(planes * self.expansion, self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool3(out)  # 32

        out = self.conv4(out)
        out = self.relu(out)
        out = self.pool4(out)  # 16

        out = self.conv5(out)
        out = self.relu(out)
        out = self.pool5(out)  # 8

        out = self.deconv1(out)  # 16
        out = self.relu(out)

        out = self.deconv2(out)  # 32
        out = self.relu(out)

        out = self.deconv3(out)  # 64
        out = self.relu(out)

        out = self.conv6(out)
        out = self.relu(out)

        out = self.conv7(out)
        out = self.relu(out)

        out = self.conv8(out)
        out = self.relu(out)

        out = self.conv9(out)

        return out


class BasicStage(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, out_channels, stride=1):
        super(BasicStage, self).__init__()
        self.inplanes = inplanes  # 128
        self.planes = planes  # 32
        self.out_channels = out_channels

        self.img_conv1 = conv5x5(self.inplanes, self.planes)
        self.img_pool1 = maxpool3x3()
        self.img_conv2 = conv5x5(self.planes, self.planes)
        self.img_pool2 = maxpool3x3()
        self.img_deconv1 = nn.ConvTranspose2d(planes, planes, 4, stride=2, padding=1)
        self.img_deconv2 = nn.ConvTranspose2d(planes, planes, 4, stride=2, padding=1)
        self.img_deconv3 = nn.ConvTranspose2d(planes, planes, 4, stride=2, padding=1)

        self.map_conv1 = conv11x11(self.planes + self.out_channels,
                                   self.planes * self.expansion)
        self.map_pool1 = maxpool3x3()
        self.map_conv2 = conv11x11(self.planes * self.expansion,
                                   self.planes * self.expansion)
        self.map_pool2 = maxpool3x3()
        self.map_conv3 = conv11x11(self.planes * self.expansion,
                                   self.planes * self.expansion)
        self.map_pool3 = maxpool3x3()
        self.map_deconv1 = nn.ConvTranspose2d(
            self.planes * self.expansion, self.planes * self.expansion,
            4, stride=2, padding=1)
        self.map_deconv2 = nn.ConvTranspose2d(
            self.planes * self.expansion, self.planes * self.expansion,
            4, stride=2, padding=1)
        self.map_deconv3 = nn.ConvTranspose2d(
            self.planes * self.expansion, self.planes * self.expansion,
            4, stride=2, padding=1)
        self.map_conv4 = conv1x1(self.planes * self.expansion,
                                 self.planes * self.expansion)
        self.map_conv5 = conv1x1(self.planes * self.expansion,
                                 self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_img, x_map):
        out_feat = self.img_conv1(x_img)
        out_feat = self.relu(out_feat)
        out_feat = self.img_pool1(out_feat)  # 16

        out_feat = self.img_conv2(out_feat)
        out_feat = self.relu(out_feat)
        out_feat = self.img_pool2(out_feat)  # 8

        out_feat = self.img_deconv1(out_feat)  # 16
        out_feat = self.relu(out_feat)

        out_feat = self.img_deconv2(out_feat)  # 32
        out_feat = self.relu(out_feat)

        out_feat = self.img_deconv3(out_feat)  # 64
        out_feat = self.relu(out_feat)

        out = torch.cat([out_feat, x_map], dim=1)  # dim = 1 channel

        out = self.map_conv1(out)
        out = self.relu(out)
        out = self.map_pool1(out)  # 32

        out = self.map_conv2(out)
        out = self.relu(out)
        out = self.map_pool2(out)  # 16

        out = self.map_conv3(out)
        out = self.relu(out)
        out = self.map_pool3(out)  # 8

        out = self.map_deconv1(out)  # 16
        out = self.relu(out)

        out = self.map_deconv2(out)  # 32
        out = self.relu(out)

        out = self.map_deconv3(out)  # 64
        out = self.relu(out)

        out = self.map_conv4(out)
        out = self.relu(out)

        out = self.map_conv5(out)

        return out


class Pose_CPM(nn.Module):
    def __init__(self, block, cfg):
        super(Pose_CPM, self).__init__()
        self.inplanes = 128  # Only used for the first conv in each stage, should be equal to planes
        self.planes = 128  # Median number in channel sizes, 128/4 128 128*4
        # self.num_stages = cfg.MODEL.EXTRA.NUM_STAGES
        self.num_stages = 6
        # self.out_channels = cfg.MODEL.NUM_JOINTS
        self.out_channels = 16

        self.stage = self._make_stage(BasicStage, self.planes, self.num_stages)
        self.middle = self._make_middle(self.planes)


    def _make_stage(self, block, planes, num_stages):
        # planes = 128
        layers = []
        layers.append(FirstStage(3, planes, self.out_channels))
        planes = planes // block.expansion  # 128 / 4 = 32
        for i in range(num_stages - 1):
            layers.append(block(self.inplanes, planes, self.out_channels))  # inplanes = 128

        return nn.ModuleList(layers)

    def _make_middle(self, planes):
        # planes = 128
        layers = []
        layers.append(conv9x9(3, planes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(maxpool3x3())

        layers.append(conv9x9(planes, planes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(maxpool3x3())

        layers.append(conv9x9(planes, planes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(maxpool3x3())

        return nn.Sequential(*layers)

    def forward(self, x_img):
        out_stage = []
        for i in range(self.num_stages):
            if i == 0:
                out = self.stage[i](x_img)
                out_middle = self.middle(x_img)
                out_stage.append(out)
            else:
                out = self.stage[i](out_middle, out_stage[i - 1])
                out_stage.append(out)

        return out_stage

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)


def get_pose_net(cfg, is_train, **kwargs):
    model = Poes_CPM(BasicStage, cfg)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model


if __name__ == '__main__':
    from torchsummary import summary
    from utils.utils import get_model_summary
    from utils.receptive_field import receptive_field, receptive_field_for_unit
    from utils.visual import Visualizer
    from utils.ops import *

    x_img = torch.randn((1, 3, 256, 256)).cuda()
    x_center_map = torch.randn((1, 1, 256, 256))
    # print(x_img.size())
    model = Pose_CPM(BasicStage, '').cuda()
    # out = model(x_img)
    # print(model)
    # print(out[0].size())
    # print(summary(model, (3, 256, 256)))
    print(get_model_summary(model, x_img, verbose=True))
    # rec = receptive_field(model, (3, 256, 256))
    # print(rec)

    
