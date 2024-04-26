#!/usr/bin/env python3

import torch
from torchinfo import summary
import sys
sys.path.append('..')
from utils.functions import *
from models.AlexNet import AlexNetWithExits
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(device)

glob = '2016_01'

batch_size = 1000
# train_data   = CustomDataset(glob=glob, as_matrix=True, directory='/home/jasimioni/ppgia/datasets/MOORE')
train_data   = CustomDataset(glob=glob, as_expanded_matrix=True, directory='/home/jasimioni/ppgia/datasets/MOORE')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

epochs = 5
model = AlexNetWithExits().to(device)
train_2exits(model, train_loader=train_loader, device=device, epochs=epochs)
