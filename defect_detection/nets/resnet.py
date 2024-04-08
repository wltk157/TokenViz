import torch
import torchvision
from torch import nn
from typing import List, Optional


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample: Optional[nn.Module] = None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample == None:
            identity = x
        else:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, init_channel, layers: List[int], num_classess=1000, dropout=False):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.bottleneck = Bottleneck

        self.conv1 = nn.Conv2d(init_channel, self.in_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.bottleneck.expansion, num_classess)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=0.2)

    def _make_layer(self, channels, num_bottleneck, stride=1):

        downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, channels * self.bottleneck.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(channels * self.bottleneck.expansion))
        layers = []

        layers.append(self.bottleneck(
            self.in_channels, channels, stride, downsample))

        self.in_channels *= self.bottleneck.expansion

        if channels != 64:
            self.in_channels = int(self.in_channels / 2)
        for _ in range(1, num_bottleneck):
            layers.append(self.bottleneck(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training and self.dropout:

            out = self.maxpool(self.relu(self.bn1(self.conv1(x))))

            out = self.layer1(out)
            out = self.dropout_layer(out)
            out = self.layer2(out)
            out = self.dropout_layer(out)
            out = self.layer3(out)
            out = self.dropout_layer(out)

        else:

            out = self.maxpool(self.relu(self.bn1(self.conv1(x))))

            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)

        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def ResNet50(init_channel, num_classess, dropout=False):
    return ResNet(init_channel, [3, 4, 6, 3], num_classess, dropout)


def ResNet101(init_channel, num_classess, dropout=False):
    return ResNet(init_channel, [3, 4, 23, 3], num_classess, dropout)


def ResNet152(init_channel, num_classess, dropout=False):
    return ResNet(init_channel, [3, 8, 36, 3], num_classess, dropout)
