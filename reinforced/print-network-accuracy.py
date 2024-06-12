#!/usr/bin/env python3

import torch
import sys
sys.path.append('..')
from utils.functions import *
from models.AlexNet import AlexNetWithExits
from models.MobileNet import MobileNetV2WithExits
from torch.utils.data import DataLoader
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--model',
                    help='Model to train (AlexNet or MobileNetV2)',
                    default='alexnet')

parser.add_argument('--trained-model',
                    help='Trained model to start training from',
                    default=None,
                    required=True)

parser.add_argument('--data',
                    help='CSV File with data to test the model',
                    required=True)

parser.add_argument('--batch-size',
                    help='Batch size to train the model',
                    default=1000)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if args.model == 'alexnet':
    model = AlexNetWithExits().to(device)
elif args.model == 'mobilenetv2':
    model = MobileNetV2WithExits().to(device)
else:
    raise ValueError('Model not supported')

directory = os.path.join(*args.data.split('/')[0:-1])
glob = args.data.split('/')[-1]

model.load_state_dict(torch.load(args.trained_model))

data   = CustomDataset(glob=glob, as_matrix=True, directory=directory)
loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

evaluate_2exits(model, loader=loader, device=device)
