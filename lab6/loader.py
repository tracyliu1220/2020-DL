import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np
import sys

trans1 = transforms.Compose([transforms.Resize((64, 64)),
                             transforms.ToTensor()])
trans2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
trans_save = transforms.Compose([
        transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
        transforms.ToPILImage()
        ])


def getImageTensor(path):
    img = Image.open(path)
    img = trans1(img)
    img = img[0:3]
    img = trans2(img)
    return img

class TrainData(data.Dataset):
    def __init__(self):
        with open('data/train.json', 'r') as f:
            _dict = json.load(f)
        with open('data/objects.json') as f:
            _ob = json.load(f)

        self.root = 'data/imgs/'
        self.file = []
        self.label = []
        for key in _dict:
            self.file.append(key)
            _label = torch.zeros(24)
            for label in _dict[key]:
                _label[_ob[label]] = 1
            self.label.append(_label)


    def __getitem__(self, idx):
        return getImageTensor(self.root+self.file[idx]), self.label[idx]

    def __len__(self):
        return len(self.file)


if __name__ == '__main__':
    print(torch.zeros(3))
    traindata = TrainData()
    img, label = traindata[3]
    img = trans_save(img)
    img.save('sample_real.png')
    # print(img.shape)
    # print(img)
    # print(label.shape)
    # print(label)
    # print(len(traindata))
    trainloader = data.DataLoader(traindata, batch_size=1, num_workers=8, pin_memory=True, shuffle=True)
    img = getImageTensor('sample.png')
    img = trans_save(img)
    img.save('gen.png')
    
