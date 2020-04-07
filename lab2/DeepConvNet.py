import torch
import torch.nn as nn
from util import *
from dataloader import *

class DeepConvNet(nn.Module):
    def __init__(self, actvt='ReLU'):
        C = 2
        T = 750
        N = 2
        super(DeepConvNet, self).__init__()
        self.activate_func = which_activate(actvt)
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 5))
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(C, 1)),
            nn.BatchNorm2d(num_features=25, eps=1e-5, momentum=0.1),
            self.activate_func,
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=50, eps=1e-5, momentum=0.1),
            self.activate_func,
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=100, eps=1e-5, momentum=0.1),
            self.activate_func,
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.Conv5 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=200, eps=1e-5, momentum=0.1),
            self.activate_func,
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.Dense = nn.Sequential(
            nn.Linear(
                in_features=8600,
                out_features=N,
                bias=True
            )
            )
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = self.Dense(x)
        return x
