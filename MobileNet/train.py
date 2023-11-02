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

batch_size = 1000
train_data   = CustomDataset(glob=glob, as_matrix=True, directory='/home/ubuntu/datasets/MOORE')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

epochs = 5
model = MobileNetV2WithExits(ch_in=1, n_classes=2).to(device)
train_2exits(model, train_loader=train_loader, device=device, epochs=epochs)