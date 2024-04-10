#!/usr/bin/env python3

import torch
from torchinfo import summary
import sys
sys.path.append('..')
from utils.functions import *
from models.AlexNet import AlexNetWithExits
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

glob = '2016_01'

model = AlexNetWithExits().to(device)
print(summary(model))
