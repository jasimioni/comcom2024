#!/usr/bin/env python3

import torch
import sys
sys.path.append('..')
from utils.functions import *
import argparse
import os

from models.AlexNet import AlexNetWithExits
from models.MobileNet import MobileNetV2WithExits

parser = argparse.ArgumentParser()

parser.add_argument('--trainedmodel',
                    help='.pth file to open',
                    type=argparse.FileType('r'))

parser.add_argument('--model',
                    help='Model to choose - [alexnet | mobilenet]',
                    default='alexnet')

parser.add_argument('--savefolder',
                    help='Folder to save to',
                    required=True)

parser.add_argument('--dataset-folder',
                    help='Folder to read from',
                    required=True)

parser.add_argument('--batch-size',
                    help='Batch size',
                    type=int,
                    default=1000)  

args = parser.parse_args()

if not os.path.isdir(args.savefolder):
    os.makedirs(args.savefolder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if args.model == 'alexnet':
    model = AlexNetWithExits().to(device)
elif args.model == 'mobilenet':
    model = MobileNetV2WithExits(ch_in=1, n_classes=2).to(device)
else:
    raise ValueError(f'Unknown model {args.model}')

model.load_state_dict(torch.load(args.trainedmodel.name))
model.eval()

for month in range(1, 13):
    glob = f'2016_{month:02d}'
    dst_csv = os.path.join(args.savefolder, f'{glob}.csv')
    print(f'Processing {glob} and saving to {dst_csv}')
    data   = CustomDataset(glob=glob, as_matrix=True, directory=args.dataset_folder)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    dump_2exits(model=model, device=device, loader=loader, savefile=dst_csv, batch_size=args.batch_size)