import torch
import torch.utils.data as data
import json
import numpy as np

'''
SOS_token = -1
EOS_token = -2
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def charToIndex(c):
    if c == 'SOS':
        return -1
    elif c == 'EOS':
        return -2
    return int(ord(c) - ord('a'))

def indexToChar(idx):
    if idx == -1 or idx == -2:
        return ''
    return chr(ord('a') + idx)

def stringToTorch(str, is_tar=False):
    ret = []
    for c in str:
        ret.append([charToIndex(c)])
    if is_tar:
        ret.append([charToIndex('EOS')])
    return torch.from_numpy(np.array(ret)).to(device)

def torchToString(t):
    ret = ''
    for num in t:
        if num[0] == -1 or num[0] == -2:
            continue
        ret += indexToChar(num[0])
    return ret

class TrainSet(data.Dataset):
    def __init__(self):
        f = open('train.json')
        f_dict = json.loads(f.read())
        f.close()

        self.input = []
        self.target = []
        for voc in f_dict:
            for _in in voc['input']:
                self.input.append(_in)
                self.target.append(voc['target'])

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

    def __len__(self):
        return self.input.length


if __name__ == '__main__':
    train_set = TrainSet()
    test_in, test_tar = train_set[3]
    print(test_in)
    print(stringToTorch(test_in))
    print(torchToString(stringToTorch(test_in)))
