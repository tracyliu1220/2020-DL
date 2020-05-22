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
from config import *

# input: 4 x 240 x 320
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.condition2Hidden = nn.Linear(nz+ncond, nz) 
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.Upsample((4, 4)),
            nn.Conv2d(nz, ngf * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            # nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.Upsample((8, 8)),
            nn.Conv2d(ngf * 8, ngf * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            # nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.Upsample((16, 16)),
            nn.Conv2d(ngf * 4, ngf * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.Upsample((32, 32)),
            nn.Conv2d(ngf * 2, ngf, 3, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Upsample((64, 64)),
            nn.Conv2d(ngf, nc, 3, padding=1, bias=False),
            nn.Tanh()
        )
    def forward(self, z, condition):
        x = self.condition2Hidden(torch.cat((z, condition), dim=-1))
        x = x.view(-1, nz, 1, 1)
        x = self.main(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.sizeFC = 40 * 4 * 4
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        self.real = nn.Sequential(
            nn.Linear(self.sizeFC, 1),
            nn.Sigmoid()
        )
        self.classify = nn.Sequential(
            nn.Linear(self.sizeFC, 24),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, self.sizeFC)
        real = self.real(x)
        classify = self.classify(x)
        return real, classify


if __name__ == '__main__':
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    # z = torch.randn(batch_size, nz, 1, 1, device=device)
    z = torch.randn(batch_size * nz, device=device)
    cond = torch.zeros(24, device=device)
    cond[3] = 1
    x = netG(z, cond)
    y = netD(x)
    print(y)




