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
from loader import *
from models import *
from evaluator import *
import pyprind
import json
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Train(netG, netD, optimG, optimD, trainloader):
    netG.train()
    netD.train()

    # real_criterion = nn.CrossEntropyLoss()
    label_criterion = nn.BCELoss()
    
    # netD
    # print('netD')
    length = 5 # len(trainloader)

    bar = pyprind.ProgPercent(len(trainloader))
    for _iter, _data in enumerate(trainloader, 0):

        if _iter % length != 0: # netD
            img, condition = _data 
            batch = len(condition)
            img = img.to(device)
            condition = condition.to(device)
    
            # fake_tensor = torch.from_numpy(np.array([0] * batch)).to(device)
            # real_tensor = torch.from_numpy(np.array([1] * batch)).to(device)

            loss = 0
            optimD.zero_grad()
            
            d_real, d_real_cond = netD(img)
            # real_loss = d_real
            real_label_loss = label_w * label_criterion(d_real_cond, condition)

            z = torch.randn(batch, nz, device=device)
            x = netG(z, condition)
            ep = torch.rand(batch, 1, 1, 1, device=device)

            d_fake, d_fake_cond = netD(x)
            # fake_loss = d_fake
            fake_label_loss = label_w * label_criterion(d_fake_cond, condition)

            # compute gradient
            interpolated = ep * img + (1 - ep) * x
            interpolated = autograd.Variable(interpolated, requires_grad=True).to(device)
            d_inter_fake, d_inter_fake_cond = netD(interpolated)
            gradients = autograd.grad(outputs=d_inter_fake, inputs=interpolated,
                    grad_outputs=torch.ones(d_inter_fake.size(), device=device),
                    create_graph=True, retain_graph=True)[0]
            gradients = gradients.view(batch, -1)
            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
            gradient_loss = gp_w * ((gradients_norm - 1) ** 2).mean()

            loss = -d_real + d_fake + gradient_loss
            loss += real_label_loss + fake_label_loss

            loss.backward()
            optimD.step()
            bar.update(1)

        else:
            img, condition = _data # netG
            batch = len(condition)
            img = img.to(device)
            condition = condition.to(device)
                
            real_tensor = torch.from_numpy(np.array([1] * batch)).to(device)
            
            optimG.zero_grad()
            loss = 0

            z = torch.randn(batch, nz, device=device)
            x = netG(z, condition)
            d_fake, d_fake_cond = netD(x)
            fake_label_loss = label_w * label_criterion(d_fake_cond, condition)
            loss = -d_fake + fake_label_loss

            loss.backward()
            optimG.step()
            bar.update(1)

def Test(netG, netE):
    acc = 0
    cnt = 0
    length = 200 # len(trainloader)
    bar = pyprind.ProgPercent(length, title='Test')
    for i, _data in enumerate(trainloader, 0):
        img, condition = _data
        batch = len(condition)
        img = img.to(device)
        condition = condition.to(device)
        z = torch.randn(batch, nz, device=device)
        x = netG(z, condition)
        acc += netE.eval(x, condition)
        cnt += 1
        bar.update(1)
        if cnt == length:
            break
    img = trans_save(x[0].cpu()) # .convert('RGB')
    img.save('sample.png')
    print('acc: ', acc / cnt)

def TrainIter(netG, netD, netE, trainloader):
    optimD = torch.optim.Adam(netD.parameters(), lr=lr)
    optimG = torch.optim.Adam(netG.parameters(), lr=lr)
    for epoch in range(epochs):
        print('\033[38;5;11mepoch', epoch, '\033[0m')
        Train(netG, netD, optimG, optimD, trainloader)
        with torch.no_grad():
            Test(netG, netE)


if __name__ == '__main__':
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netE = evaluation_model()
    traindata = TrainData()
    trainloader = data.DataLoader(traindata, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
    TrainIter(netG, netD, netE, trainloader)
