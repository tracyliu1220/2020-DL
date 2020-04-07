from dataloader import *
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.multiprocessing as mp

mp.set_start_method('spawn')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.set_default_dtype(torch.double)

def which_activate(choice):
    activate_functions = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
        'ELU': nn.ELU(alpha=1.0)
    }
    activate_func = activate_functions[choice]
    return activate_func

def accuracy(y, t):
    cnt = 0
    for i in range(y.shape[0]):
        if (y[i][0] > y[i][1]) and t[i] == 0:
            cnt = cnt + 1
        if (y[i][0] < y[i][1]) and t[i] == 1:
            cnt = cnt + 1
    return cnt / y.shape[0]

class TrainSet(data.Dataset):
    def __init__(self):
        self.train_data, self.train_label = read_bci_train_data()
        self.train_data = self.train_data.to(device)
        self.train_label = self.train_label.long().to(device)
    def __getitem__(self, idx):
        return self.train_data[idx], self.train_label[idx]
    def __len__(self):
        return len(self.train_data)

class TestSet(data.Dataset):
    def __init__(self):
        self.test_data, self.test_label = read_bci_test_data()
        self.test_label = self.test_label.long()
    def __getitem__(self, idx):
        return self.test_data[idx], self.test_label[idx]
    def __len__(self):
        return len(self.test_data)
