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

parser.add_argument('--train-data',
                    help='CSV File with data to train the model',
                    required=True)

parser.add_argument('--batch-size',
                    help='Batch size to train the model',
                    default=1000)

parser.add_argument('--epochs',
                    help='Number of epochs to train the model',
                    default=15)  

parser.add_argument('--save-path',
                    help='Path to save the trained model',
                    required=True)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

if args.model == 'alexnet':
    model = AlexNetWithExits().to(device)
elif args.model == 'mobilenetv2':
    model = MobileNetV2WithExits().to(device)
else:
    raise ValueError('Model not supported')

directory = os.path.join(*args.train_data.split('/')[0:-1])
glob = args.train_data.split('/')[-1]

model.load_state_dict(torch.load(args.trained_model))

train_data   = CustomDataset(glob=glob, as_matrix=True, directory=directory)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

train_2exits(model, train_loader=train_loader, device=device, epochs=int(args.epochs), save_path=args.save_path)