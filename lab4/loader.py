import torch
import torch.utils.data as data
import json
import numpy as np
import random

SOS_token = 0
EOS_token = 1
UNK_token = 29

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def charToIndex(c):
    if c == 'SOS':
        return SOS_token
    elif c == 'EOS':
        return EOS_token
    return int(ord(c) - ord('a')) + 2

def indexToChar(idx):
    if idx == SOS_token or idx == EOS_token or idx == UNK_token:
        return ''
    return chr(ord('a') + idx - 2)

def stringToTorch(str, is_tar=False):
    ret = []
    for c in str:
        idx = charToIndex(c)
        ret.append([idx])
    if is_tar:
        ret.append([charToIndex('EOS')])

    return torch.from_numpy(np.array(ret)) # .to(device)

def torchToString(t):
    ret = ''
    for num in t:
        num = num[0]
        # num = torch.max(num[0], 0)[1]
        # print(num)
        if num == SOS_token or num == EOS_token:
            continue
        ret += indexToChar(num)
    return ret

class DataSet(data.Dataset):
    def __init__(self, mode):
        path = 'data/'+mode+'.json'

        f = open(path)
        f_dict = json.loads(f.read())
        f.close()

        self.input = []
        self.target = []
        for voc in f_dict:
            for _in in voc['input']:
                self.input.append(_in)
                self.target.append(voc['target'])

        self.rand = [ i for i in range(len(self.input)) ]
        random.shuffle(self.rand)

    def __getitem__(self, idx):
        idx = idx % len(self)
        return self.input[self.rand[idx]], self.target[self.rand[idx]]

    def __len__(self):
        return len(self.input)


if __name__ == '__main__':
    train_set = DataSet('train')
    test_set = DataSet('test')
    print(len(train_set))
    print(len(test_set))
    test_in, test_tar = train_set[3]
    print(test_in)
    print(test_tar)
    print(stringToTorch(test_in, is_tar=True))
    print(torchToString(stringToTorch(test_in)))
