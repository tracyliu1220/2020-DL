from util import *
import torch
import torch.nn as nn
import torch.utils.data as data
from EEGNet import *
from DeepConvNet import *
import torch.multiprocessing as mp

# mp.set_start_method('spawn')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.set_default_dtype(torch.double)

def Train(net, epoches, train_data, test_data, optimizer, loss_func, batch_size=64, print_steps=5):
    testloader = data.DataLoader(test_data, batch_size=120, shuffle=False, num_workers=0)
    
    for epoch in range(epoches):
        trainloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        print('\033[38;5;014mepoch', epoch, '\033[0m')
        running_loss = 0
        cnt = 0
        acc = 0
        for i, _data in enumerate(trainloader, 0):
            cnt += 1
            inputs, labels = _data
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            # print('loss:', loss.item())
            # print('acc :', accuracy(outputs, labels))
            running_loss += loss.item()
            acc += accuracy(outputs, labels)

        print('loss:', running_loss / cnt)
        print('acc :', acc / cnt)
        if (epoch + 1) % print_steps == 0:
            Test(net, testloader, loss_func)

def Test(net, testloader, loss_func):
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
    print('\033[38;5;011m---')
    print('test loss:', running_loss / cnt)
    print('test acc :', acc / cnt)
    print('---\033[0m')



def main():
    # net: DeepConvNet, EEGNet
    net = EEGNet().to(device)

    # hyper parameters
    batch_size = 64
    learning_rate = 0.001
    epoches = 300
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    # data
    train_data = TrainSet()
    test_data = TestSet()
    
    # train
    Train(net, epoches, train_data, test_data, optimizer, loss_func, batch_size)

    print('done')

if __name__ == '__main__':
    main()
