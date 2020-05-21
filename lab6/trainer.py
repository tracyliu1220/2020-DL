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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Train(netG, netD, optimG, optimD, trainloader):
    netG.train()
    netD.train()

    real_criterion = nn.CrossEntropyLoss()
    label_criterion = nn.BCELoss()
    
    # netD
    # print('netD')
    for _d_iter in range(d_iter):
        bar = pyprind.ProgPercent(len(trainloader), title='netD iter '+str(_d_iter))
        for i, _data in enumerate(trainloader, 0):
            img, condition = _data 
            batch = len(condition)
            img = img.to(device)
            condition = condition.to(device)
    
            fake_tensor = torch.from_numpy(np.array([0] * batch)).to(device)
            real_tensor = torch.from_numpy(np.array([1] * batch)).to(device)

            loss = 0
            optimD.zero_grad()
            
            d_real, d_real_cond = netD(img)
            loss += real_criterion(d_real, real_tensor)
            loss += label_w * label_criterion(d_real_cond, condition)

            z = torch.randn(batch, nz, device=device)
            latent = netG(z, condition)
            d_fake, d_fake_cond = netD(latent)
            loss += real_criterion(d_fake, fake_tensor)
            loss += label_w * label_criterion(d_fake_cond, condition)

            loss.backward()
            optimD.step()
            bar.update(1)
    
    # netG
    # print('netG')
    bar = pyprind.ProgPercent(len(trainloader), title='netG')
    for i, _data in enumerate(trainloader, 0):
        img, condition = _data
        batch = len(condition)
        img = img.to(device)
        condition = condition.to(device)
            
        real_tensor = torch.from_numpy(np.array([1] * batch)).to(device)
        
        optimG.zero_grad()
        loss = 0

        z = torch.randn(batch, nz, device=device)
        latent = netG(z, condition)
        d_fake, d_fake_cond = netD(latent)
        loss += real_criterion(d_fake, real_tensor)
        loss += label_w * label_criterion(d_fake_cond, condition)

        loss.backward()
        optimG.zero_grad()
        bar.update(1)

def Test(netG, netE):
    acc = 0
    cnt = 0
    bar = pyprind.ProgPercent(len(trainloader), title='Test')
    for i, _data in enumerate(trainloader, 0):
        img, condition = _data
        batch = len(condition)
        img = img.to(device)
        condition = condition.to(device)
        z = torch.randn(batch, nz, device=device)
        latent = netG(z, condition)
        acc += netE.eval(latent, condition)
        cnt += 1
        bar.update(1)
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
