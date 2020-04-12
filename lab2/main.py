from util import *
import torch
import torch.nn as nn
import torch.utils.data as data
from EEGNet import *
from DeepConvNet import *
import torch.multiprocessing as mp
import json

# mp.set_start_method('spawn')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
torch.set_default_dtype(torch.double)

def Train(net, trainloader, loss_func, learning_rate):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    running_loss = 0
    cnt = 0
    acc = 0
    for i, _data in enumerate(trainloader, 0):
        cnt += 1
        inputs, labels = _data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        acc += accuracy(outputs, labels)
    acc = acc / cnt
    running_loss = running_loss / cnt
    return acc * 100


def Test(net, net_name, testloader, loss_func):
    net.eval()
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
            loss = loss_func(outputs, labels)
            running_loss += loss.item()
            acc += accuracy(outputs, labels)
        acc = acc / cnt
    return acc * 100

def ChangeState(net, net_name, train_acc, test_acc, target_acc, results):
    # results
    results[net_name]['train_acc'].append(train_acc)
    results[net_name]['test_acc'].append(test_acc)

    # target
    if test_acc > target_acc[net_name]:
        torch.save(net.state_dict(), 'results/weights/'+net_name+'-'+'{:.2f}'.format(test_acc)+'.pth')
        target_acc[net_name] = test_acc


def TrainIter(net_id, loader, hyper, target_acc, results):
    print('\033[38;5;011m'+net_id['name']+'\033[0m')
    for epoch in range(hyper['epoches']):
        print('\033[38;5;014mepoch', epoch, '\033[0m')
        train_acc = Train(net_id['net'], loader['train'], hyper['loss_func'], hyper['learning_rate'])
        test_acc  = Test(net_id['net'], net_id['name'], loader['test'], hyper['loss_func'])
        ChangeState(net_id['net'], net_id['name'], train_acc, test_acc, target_acc, results)
        print('train acc:', '{:.2f}'.format(train_acc), ' ', 'test acc:', '{:.2f}'.format(test_acc), ' ', '\033[38;5;010mhighest:', '{:.2f}'.format(target_acc[net_id['name']]), '\033[0m')

def Demo(net_id, testloader, loss_func):
    pth = {
        'DeepConvNet_ELU': 'DeepConvNet_ELU-79.35.pth',
        'DeepConvNet_LeakyReLU': 'DeepConvNet_LeakyReLU-80.46.pth',
        'DeepConvNet_ReLU': 'DeepConvNet_ReLU-80.09.pth',
        'EEGNet_ELU': 'EEGNet_ELU-84.63.pth',
        'EEGNet_LeakyReLU': 'EEGNet_LeakyReLU-88.80.pth',
        'EEGNet_ReLU': 'EEGNet_ReLU-88.43.pth'
    }
    net_id['net'].load_state_dict(torch.load('results/weights/'+pth[net_id['name']]))
    test_acc = Test(net_id['net'], net_id['name'], testloader, loss_func)
    print(('\033[38;5;011m'+net_id['name']).ljust(30, ' '), '\t', '\033[0mtest acc:', '{:.2f}'.format(test_acc), ' ', )

def main():
    # control: train, demo
    action = 'train'

    # net: DeepConvNet, EEGNet
    EEGNet_ReLU      = EEGNet('ReLU').to(device)
    EEGNet_LeakyReLU = EEGNet('LeakyReLU').to(device)
    EEGNet_ELU       = EEGNet('ELU').to(device)
    DeepConvNet_ReLU      = DeepConvNet('ReLU').to(device)
    DeepConvNet_LeakyReLU = DeepConvNet('LeakyReLU').to(device)
    DeepConvNet_ELU       = DeepConvNet('ELU').to(device)
    nets = [ {'net': EEGNet_ReLU,           'name': 'EEGNet_ReLU'},
             {'net': EEGNet_LeakyReLU,      'name': 'EEGNet_LeakyReLU'},
             {'net': EEGNet_ELU,            'name': 'EEGNet_ELU'},
             {'net': DeepConvNet_ReLU,      'name': 'DeepConvNet_ReLU'},
             {'net': DeepConvNet_LeakyReLU, 'name': 'DeepConvNet_LeakyReLU'},
             {'net': DeepConvNet_ELU,       'name': 'DeepConvNet_ELU'} ]

    # hyper parameters
    hyper = {'batch_size'   : 1080,
             'learning_rate': 0.001,
             'epoches'      : 2000,
             'loss_func'    : nn.CrossEntropyLoss()}

    # data
    trainloader = data.DataLoader(TrainSet(), batch_size=hyper['batch_size'], num_workers=0)
    testloader = data.DataLoader(TestSet(), batch_size=1080, num_workers=0)
    loader = {'train': trainloader, 'test': testloader}

    # target acc
    with open('results/target_acc.json', 'r') as f:
        target_acc = json.load(f)
    
    # results
    with open('results/results_initial.json', 'r') as f:
        results = json.load(f)

    # action
    for net_id in nets:
        if action == 'train':
            TrainIter(net_id, loader, hyper, target_acc, results)
        if action == 'demo':
            Demo(net_id, testloader, hyper['loss_func'])
    
    with open('results/target_acc.json', 'w') as f:
        f.write(json.dumps(target_acc))
    with open('results/results.json', 'w') as f:
        f.write(json.dumps(results))

if __name__ == '__main__':
    main()
