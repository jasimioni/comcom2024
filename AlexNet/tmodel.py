#!/usr/bin/env python3

import torch
from torchinfo import summary
import sys
sys.path.append('..')
from utils.functions import *
from models.AlexNet import AlexNet
from torch.utils.data import DataLoader

try:
    savefile = sys.argv[1]
    assert os.path.isfile(savefile), f'{savefile} is not a valid file'
except Exception as e:
    print(f'Failed to read model state dict: {e}')
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

glob = '2016_01'

model = AlexNet(num_classes=2).to(device)
model.load_state_dict(torch.load(savefile))
model.eval()

model(torch.rand(1, 1, 8, 8).to(device))
