import torch
import torch.utils.data as data
import json
import numpy as np
import random

'''
SOS_token = 0
EOS_token = 1
'''

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def charToIndex(c):
    if c == 'SOS':
        return 0
    elif c == 'EOS':
        return 1
    return int(ord(c) - ord('a')) + 2

def indexToChar(idx):
    if idx == 0 or idx == 1:
        return ''
    return chr(ord('a') + idx - 2)

def stringToTorch(str, is_tar=False):
    ret = []
    for c in str:
        idx = charToIndex(c)
        # one_hot = [ 0 for i in range(29) ]
        # one_hot[idx] = 1
        # ret.append([one_hot])
        ret.append([idx])
    if is_tar:
        # tar = [ 0 for i in range(29) ]
        # tar[charToIndex('EOS') + 2] = 1;
        # ret.append([tar])
        ret.append([charToIndex('EOS')])
    return torch.from_numpy(np.array(ret)) # .to(device)

def torchToString(t):
    ret = ''
    for num in t:
        # num = num[0]
        num = torch.max(num[0], 0)[1]
        # print(num)
        if num == 0 or num == 1:
            continue
        ret += indexToChar(num)
    return ret

class DataSet(data.Dataset):
    def __init__(self, mode):
        if mode == 'train':
            path = 'data/train.json'
        else:
            path = 'data/test.json'

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
    test_in, test_tar = train_set[3]
    print(test_in)
    print(test_tar)
    print(stringToTorch(test_in))
    print(torchToString(stringToTorch(test_in)))
