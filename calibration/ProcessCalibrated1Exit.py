#!/usr/bin/env python3

import torch
import torch.nn as nn
from torchinfo import summary
import sys
import os
sys.path.append('..')
from utils.functions import *
from models.AlexNet import AlexNet
from models.MobileNet import MobileNetV2
from torch.utils.data import DataLoader
from temperature_scaling import ModelWithTemperature
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('trainedmodel',
                    help='.pth file to open',
                    type=argparse.FileType('r'))

parser.add_argument('--model',
                    help='Model to choose - [alexnet | mobilenet]',
                    default='alexnet')

parser.add_argument('--savefolder',
                    help='Folder to save to',
                    required=True)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if args.model == 'alexnet':
    model = AlexNet().to(device)
else:
    model = MobileNetV2(ch_in=1, n_classes=2).to(device)

model.load_state_dict(torch.load(args.trainedmodel.name))

model = ModelWithTemperature(model)

directory = '../MOORE'
glob = '2016_02'
batch_size = 1000
    
data   = CustomDataset(glob=glob, as_matrix=True, directory=directory)
loader = DataLoader(data, batch_size=batch_size, shuffle=False)

model.set_temperature(loader)

calibrated_name = args.trainedmodel.name[0:-4] + '_calibrated.pth'
torch.save(model.state_dict(), calibrated_name)

for month in range(12):
    glob = f'2016_{month+1:02d}'

    data   = CustomDataset(glob=glob, as_matrix=True, directory=directory)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    df = pd.DataFrame()

    for b, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)
        b += 1

        print(f'{b:05d}: {len(y)}')

        results = model(X) 

        line = y.view(-1, 1).cpu().numpy().tolist()

        count = len(y)

        y_pred = results

        certainty, predicted = torch.max(nn.functional.softmax(y_pred, dim=-1), 1)

        print(f"Count: {count}")

        for n in range(count):
            line[n].extend([ predicted[n].item(), certainty[n].item() ])

        line_df = pd.DataFrame(line)#, columns = [ 'y', 'y_exit_1', 'cnf_exit_1', 'bb_time_exit_1', 'exit_time_exit_1',
                                        #                 'y_exit_2', 'cnf_exit_2', 'bb_time_exit_2', 'exit_time_exit_2' ])
        
        df = pd.concat([ df, line_df ], ignore_index=True)

    df.columns = [ 'y', 'y_pred', 'cnf' ]

    savefile = os.path.join(args.savefolder, f"{glob}.csv")
    df.to_csv(savefile, index=False)

    del df, data, loader