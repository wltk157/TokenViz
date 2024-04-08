import torch
import torch.nn as nn

from .vgg import VGG_with_attn_without_class, VGG_without_classfication
from .resnet import ResNet50


def get_img_output_length(width, height):
    def get_output_length(input_length):
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length

    return get_output_length(width) * get_output_length(height)


class Siamese_VGG(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese_VGG, self).__init__()
        self.vgg = VGG_without_classfication(input_shape[0])

        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[2])
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x

        x1 = self.vgg(x1)
        x2 = self.vgg(x2)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)

        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x


class Siamese_VGG_attn(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese_VGG_attn, self).__init__()
        self.vgg = VGG_with_attn_without_class(input_shape[0])

        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[2])
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x

        x1 = self.vgg(x1)
        x2 = self.vgg(x2)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)

        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x


class Siamese_Resnet(nn.Module):
    def __init__(self, input_shape, dropout=False):
        super(Siamese_Resnet, self).__init__()
        self.resnet = ResNet50(input_shape[0], dropout, [input_shape[0], input_shape[1]])

        self.fully_connect1 = torch.nn.Linear(2048, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x

        x1 = self.resnet(x1)
        x2 = self.resnet(x2)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)

        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x
