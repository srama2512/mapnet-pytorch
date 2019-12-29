import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

class DilatedResNet(nn.Module):

    def __init__(self, block, layers, strides=[1,2,2,2], dilations=[1,1,1,1]):
        self.inplanes = 64
        super(DilatedResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def train(self, mode):
        return super(DilatedResNet, self).train(False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DilatedResNetBackbone(nn.Module):
    def __init__(self, block, layers, strides=[1,2,2,2], dilations=[1,1,1,1], pretrained=None):
        super(DilatedResNetBackbone, self).__init__()
        rnet = DilatedResNet(block, layers, strides, dilations)

        pretrained = {k:v for k,v in pretrained.items() if 'fc' not in k}
        rnet.load_state_dict(pretrained)
        self.rnet = rnet
        for param in self.rnet.parameters():
            param.requires_grad = False

        self.spatial_dim = int(56//np.prod(strides))
        self.feat_dim = 2048

    def forward(self, x):
        with torch.no_grad():
            x = self.rnet(x)
        return x

class R8(nn.Module):
    def __init__(self, block, layers, strides=[1,2,2,2], dilations=[1,1,1,1]):
        super(DilatedResNetBackbone, self).__init__()
        rnet = DilatedResNet(block, layers, strides, dilations)

        pretrained = {k:v for k,v in pretrained.items() if 'fc' not in k}
        rnet.load_state_dict(pretrained)
        self.rnet = rnet
        for param in self.rnet.parameters():
            param.requires_grad = False

        self.spatial_dim = int(56//np.prod(strides))
        self.feat_dim = 2048

    def forward(self, x):
        with torch.no_grad():
            x = self.rnet(x)
        return x

def r50_n7():
    wts = tmodels.resnet50(pretrained=True).state_dict()
    net = DilatedResNetBackbone(Bottleneck, [3, 4, 6, 3], strides=[1,2,2,2], dilations=[1,1,1,1], pretrained=wts)
    return net

def dr50_n28():
    wts = tmodels.resnet50(pretrained=True).state_dict()
    net = DilatedResNetBackbone(Bottleneck, [3, 4, 6, 3], strides=[1,2,1,1], dilations=[1,1,2,4], pretrained=wts)
    return net

def dr50_n56(pretrained=True):
    wts = tmodels.resnet50(pretrained=pretrained).state_dict()
    net = DilatedResNetBackbone(Bottleneck, [3, 4, 6, 3], strides=[1,1,1,1], dilations=[1,2,4,8], pretrained=wts)
    return net

def r8():
    net = DilatedResNet(Bottleneck, [1,1,1,1], strides=[1,2,2,2], dilations=[1,1,1,1])
    net.load_state_dict(tmodels.resnet101(pretrained=True).state_dict(), strict=False)
    net.feat_dim = 2048
    net.spatial_dim = 7
    return net

class AVDResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet50 = dr50_n56(pretrained=pretrained)
        self.main = nn.Sequential(
                                resnet50.rnet.conv1,
                                resnet50.rnet.bn1,
                                resnet50.rnet.relu,
                                resnet50.rnet.maxpool,
                                resnet50.rnet.layer1,
                                resnet50.rnet.layer2,
                                nn.Conv2d(512, 32, 1, 1)
                           )

    def forward(self, x):
        out = self.main(x)
        return out
