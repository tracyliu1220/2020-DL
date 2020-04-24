import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np
import sys

def accuracy(y, t):
    return (torch.max(y, 1)[1] == t.long()).sum().item() / len(y)
    # tmp = (torch.round(y).long() == t.long())
    # return (torch.round(y).int() == t.int()).sum().item() / len(y)
