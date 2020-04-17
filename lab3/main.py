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
from dataloader import *
from util import *
from models import *
import pyprind
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Train(net, trainloader, loss_func, learning_rate):
    bar = pyprind.ProgPercent(len(trainloader), title='training...')
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    net.train()
    running_loss = 0
    acc = 0
    cnt = 0
    for i, _data in enumerate(trainloader, 0):
        inputs, labels = _data
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        acc += accuracy(outputs, labels)
        bar.update(1)
    acc = acc / len(trainloader)
    running_loss = running_loss / len(trainloader)
    return acc * 100

def Test(net, testloader):
    net.eval()
    bar = pyprind.ProgPercent(len(testloader), title='testing...')
    with torch.no_grad():
        running_loss = 0
        cnt = 0
        acc = 0
        for i, _data in enumerate(testloader, 0):
            cnt += 1
            inputs, labels = _data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            acc += accuracy(outputs, labels)
            bar.update(1)
        acc = acc / cnt
    return acc * 100

def ChangeState(net, train_acc, test_acc, target_acc, results):
    # results
    results[net.name]['train_acc'].append(train_acc)
    results[net.name]['test_acc'].append(test_acc)

    # target
    if test_acc > target_acc[net.name]:
        # torch.save(net.state_dict(), 'results/weights/'+net.name+'-'+'{:.2f}'.format(test_acc)+'.pth')
        # torch.save(net, 'results/weights/model-'+net.name+'-'+'{:.2f}'.format(test_acc)+'.pth')
        target_acc[net.name] = test_acc

def TrainIter(net, epoches, trainloader, testloader, target_acc, results):
    print('\033[38;5;011m'+net.name+'\033[0m')
    # hyper
    learning_rate = 0.001
    # epoches = 10
    loss_func = nn.MSELoss()
    highest_acc = 0

    results[net.name]['train_acc'] = []
    results[net.name]['test_acc'] = []

    for epoch in range(epoches):
        print('\033[38;5;014mepoch', epoch, '\033[0m')
        train_acc = Train(net, trainloader, loss_func, learning_rate)
        test_acc = Test(net, testloader)
        highest_acc = max(highest_acc, test_acc)
        ChangeState(net, train_acc, test_acc, target_acc, results)
        print('train acc:', '{:.2f}'.format(train_acc), ' ', 'test acc:', '{:.2f}'.format(test_acc), ' ', '\033[38;5;010mhighest:', '{:.2f}'.format(target_acc[net.name]), '\033[0m')

def ReadState():
    # target acc
    with open('results/target_acc.json', 'r') as f:
        target_acc = json.load(f)
    
    # results
    with open('results/results.json', 'r') as f:
        results = json.load(f)
    return target_acc, results


def WriteState(target_acc, results):
    with open('results/target_acc.json', 'w') as f:
        f.write(json.dumps(target_acc))
    with open('results/results.json', 'w') as f:
        f.write(json.dumps(results))

def main():
    batch_size = 10
    trainloader = data.DataLoader(RetinopathyLoader('data/imgs/', 'train'), batch_size=batch_size, num_workers=8, pin_memory=True)
    testloader = data.DataLoader(RetinopathyLoader('data/imgs/', 'test'), batch_size=batch_size, num_workers=8, pin_memory=True)

    resnet18_pre = ResNet('resnet18_pretrained', 18, pretrained=True).to(device)
    resnet50_pre = ResNet('resnet50_pretrained', 50, pretrained=True).to(device)
    resnet18 = ResNet('resnet18', 18, pretrained=False).to(device)
    resnet50 = ResNet('resnet50', 50, pretrained=False).to(device)

    target_acc, results = ReadState()


    # Test(resnet18_pre, testloader)

    # TrainIter(resnet18_pre, 10, trainloader, testloader, target_acc, results)
    # WriteState(target_acc, results)
    # del resnet18_pre
    TrainIter(resnet50_pre, 10, trainloader, testloader, target_acc, results)
    # WriteState(target_acc, results)
    del resnet50_pre
    TrainIter(resnet18, 10, trainloader, testloader, target_acc, results)
    # WriteState(target_acc, results)
    del resnet18
    TrainIter(resnet50, 10, trainloader, testloader, target_acc, results)
    # WriteState(target_acc, results)
    del resnet50


if __name__ == '__main__':
    main()
