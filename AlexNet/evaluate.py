#!/usr/bin/env python3

import torch
from torchinfo import summary
import sys
sys.path.append('..')
from utils.functions import *
from models.MobileNet import MobileNetV2WithExits
from torch.utils.data import DataLoader

try:
    savefile = sys.argv[1]
    assert os.path.isfile(savefile), f'{savefile} is not a valid file'
except Exception as e:
    print(f'Failed to read model state dict: {e}')
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

glob = '2016_01'

model = MobileNetV2WithExits(ch_in=1, n_classes=2).to(device)
model.load_state_dict(torch.load(savefile))
model.eval()

path = os.path.join('evaluations', savefile)
Path(path).mkdir(parents=True, exist_ok=True)


for year in range(2016, 2020):
    for month in range(1, 13):
        glob = f'{year:04d}_{month:02d}'
        csv = os.path.join(path, f'{glob}.csv')
        print(f'Processing {glob} and saving to {csv}')
        dump_2exits(model=model, device=device, glob=glob, savefile=csv)
