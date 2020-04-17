import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.cuda.comm as comm
from torch.autograd.function import once_differentiable
import time
import functools


class ResNet(nn.Module):
    def __init__(self, name, model_size=18, pretrained=True):
        super(ResNet, self).__init__()

        self.name = name
        if model_size == 18:
            self.last_input_size = 512
        elif model_size == 50:
            self.last_input_size = 2048

        pretrained_model = torchvision.models.__dict__['resnet{}'.format(model_size)](pretrained=pretrained)
        # if pretrained:
        #     for param in pretrained_model.parameters():
        #         param.requires_grad = False

        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.dropout = nn.Dropout(p=0.5)
        self.classify = nn.Linear(self.last_input_size, 1)
        self.relu = nn.ReLU()

        # self.dropout2 = nn.Dropout(p=0.3)
        # self.classify2 = nn.Linear(10, 1)
        # self.relu2 = nn.ReLU()

        del pretrained_model

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
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classify(x)
        x = self.relu(x)
        # x = self.dropout2(x)
        # x = self.classify2(x)
        # x = self.relu2(x)

        return x

if __name__ == '__main__':
    resnet = ResNet('resnet18', model_size=18)
    # for param in resnet.parameters():
    #     print(param)

    print(resnet)
