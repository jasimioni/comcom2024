#!/usr/bin/env python3

import torch
import torch.nn as nn
from torchinfo import summary
import sys
import os
sys.path.append('..')
from utils.functions import *
from models.AlexNet import AlexNetWithExits
from models.MobileNet import MobileNetV2WithExits
from torch.utils.data import DataLoader
from temperature_scaling_2exits import ModelWithTemperature
import pandas as pd
import argparse

'''
./Calibrate2exits.py --model alexnet ../trained_models/AlexNetWithExits_epoch_19_90.1_91.1.pth --savefolder calibrated/AlexNetWithExits/
./Calibrate2exits.py --model mobilenet ../trained_models/MobileNetV2WithExits_epoch_19_89.7_90.9.pth --savefolder calibrated/MobileNetWithExits/
'''

parser = argparse.ArgumentParser()

parser.add_argument('--trained-model',
                    help='.pth file to open',
                    required=True)

parser.add_argument('--calibrated-model-savefile',
                    help='.pth file to save',
                    required=True)

parser.add_argument('--model',
                    help='Model to choose - [alexnet | mobilenet]',
                    default='alexnet')

parser.add_argument('--savefolder',
                    help='Folder to save to',
                    required=False)

parser.add_argument('--batch-size',
                    help='Batch size',
                    default=1000,
                    type=int)

parser.add_argument('--max-iter', 
                    help='Max iterations for temperature scaling',
                    default=10,
                    type=int)

parser.add_argument('--epochs',
                    help='Number of epochs for training',
                    default=80,
                    type=int)

parser.add_argument('--dataset',
                    help='Dataset to use',
                    required=True)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)

if args.model == 'alexnet':
    model = AlexNetWithExits().to(device)
else:
    model = MobileNetV2WithExits(ch_in=1, n_classes=2).to(device)

model.load_state_dict(torch.load(args.trained_model))

print(f"Epochs: {args.epochs}, Max Iterations: {args.max_iter}")
model_t = ModelWithTemperature(model, device=device, max_iter=args.max_iter, epochs=args.epochs)

data   = CustomDataset(as_matrix=True, file=args.dataset)
loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

model_t.set_temperature(loader)

torch.save(model_t.state_dict(), args.calibrated_model_savefile)

print(model_t.temperature)
print(model_t.temperature[0])
print(model_t.temperature[1])

model_t = ModelWithTemperature(model, device=device, max_iter=args.max_iter, epochs=args.epochs)
model_t.load_state_dict(torch.load(args.calibrated_model_savefile))

print(model_t.temperature)
print(model_t.temperature[0])
print(model_t.temperature[1])

# Evaluate the model - need to remove from here

if args.savefolder:
    os.makedirs(args.savefolder, exist_ok=True)
    directory = '../MOORE'

    for month in range(1, 13):
        torch.cuda.empty_cache()
        glob = f'2016_{month:02d}'

        data   = CustomDataset(glob=glob, as_matrix=True, directory=directory)
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

        df = pd.DataFrame()

        for b, (X, y) in enumerate(loader):
            X = X.to(device)
            y = y.to(device)
            b += 1

            print(f'{b:05d}: {len(y)}', end="")

            y_pred = model_t(X) 

            line = y.view(-1, 1).cpu().numpy().tolist()

            for exit, results in enumerate(y_pred):   
                count = len(y)

                y_pred_exit = results

                certainty, predicted = torch.max(nn.functional.softmax(y_pred_exit, dim=-1), 1)
                accuracy = (predicted == y).sum().item() / len(y)
                avg_certainty = certainty.mean().item()

                print(f' | Exit {exit + 1}: Accuracy: {accuracy:.3f}, Avg Certainty: {avg_certainty:.3f}' , end="")

                avg_bb_time = 0
                avg_exit_time = 0

                for n in range(count):
                    line[n].extend([ predicted[n].item(), certainty[n].item(), avg_bb_time, avg_exit_time ])

            print("")
            line_df = pd.DataFrame(line)#, columns = [ 'y', 'y_exit_1', 'cnf_exit_1', 'bb_time_exit_1', 'exit_time_exit_1',
                                         #                 'y_exit_2', 'cnf_exit_2', 'bb_time_exit_2', 'exit_time_exit_2' ])
            
            df = pd.concat([ df, line_df ], ignore_index=True)

        df.columns = [ 'y', 'y_exit_1', 'cnf_exit_1', 'bb_time_exit_1', 'exit_time_exit_1',
                            'y_exit_2', 'cnf_exit_2', 'bb_time_exit_2', 'exit_time_exit_2' ]

        savefile = os.path.join(args.savefolder, f"{glob}.csv")
        df.to_csv(savefile, index=False)

        del df, data, loader
