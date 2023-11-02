#!/usr/bin/env python3

import torch
from torchinfo import summary
import sys
sys.path.append('..')
from utils.functions import *
from models.MobileNet import MobileNetV2WithExits
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

glob = '2016_01'

model = MobileNetV2WithExits(ch_in=1, n_classes=2).to(device)
print(summary(model))
