import torch
import torch.nn as nn
from util import *

class EEGNet(nn.Module):
    def __init__(self, actvt='ReLU'):
        super(EEGNet, self).__init__()
        self.activate_func = which_activate(actvt)
        self.firstConv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,                # input height
                out_channels=16,            # n filters
                kernel_size=(1, 51),    # filter size
                stride=(1, 1),                # filter movement/step
                padding=(0, 25),
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=16,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            )
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(2, 1),
                stride=(1, 1),
                groups=16,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=32,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            self.activate_func,
            nn.AvgPool2d(
                kernel_size=(1, 4),
                stride=(1, 4),
                padding=0
            ),
            nn.Dropout(p=0.6)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 15),
                stride=(1, 1),
                padding=(0, 7),
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=32,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            self.activate_func,
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=0
            ),
            nn.Dropout(p=0.6)
        )
        self.classify = nn.Sequential(
            nn.Linear(
                in_features=736,
                out_features=2,
                bias=True
            )
        )
    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.reshape(x.shape[0], 736)
        x = self.classify(x)
        return x
