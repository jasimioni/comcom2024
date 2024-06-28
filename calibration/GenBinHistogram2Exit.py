#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import os
from sklearn.linear_model import Ridge
import argparse
from helper import *


plt.rc('font', family='serif', size=16)


def genHistogram(file, savefolder, description):
    df_total = pd.read_csv(file)

    os.makedirs(savefolder, exist_ok=True)

    n_bin = 10

    print(f"## Network: {description}")

    for exit in [ 1, 2 ]:

        dfs = {
            'total': df_total,
            'normal': df_total.query("y == 0"),
            'attack': df_total.query("y == 1")
        }

        for df_name in [ 'total' ]: #, 'normal', 'attack' ]:
            df = dfs[df_name]

            max = df[f"cnf_exit_{exit}"].max()
            min = df[f"cnf_exit_{exit}"].min()
            
            max = 1
            min = 0.5

            step = ( max - min ) / n_bin

            print(f"### Exit {exit} | {df_name}")

            bin_acc = []
            bins = []
            bin_labels = []

            print("```")
            for bin in range(n_bin):
                # y  y_pred cnf
                bin_min = min + bin * step
                bin_max = min + ( bin + 1 ) * step

                df_bin = df.query(f"cnf_exit_{exit} > {bin_min} and cnf_exit_{exit} <= {bin_max}")
                total = df_bin['y'].count()
                correct = df_bin.query(f"y == y_exit_{exit}")['y'].count() 

                accuracy = correct / total

                bin_acc.append(accuracy)
                bin_labels.append(f"{bin_min:.2f}-")
                bins.append(bin)
                
                print(f"{bin_min:.4f} < x <= {bin_max:.4f}: {total} | {correct} | {accuracy:.4f}")

            fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))
            print("```")
            ax.bar(bin_labels, bin_acc)

            lr = Ridge()

            np_bins = np.array(bins).reshape(-1, 1)

            print(np_bins)
            print(bin_acc)

            lr.fit(np_bins, bin_acc)
            plt.plot(bin_labels, lr.coef_*np_bins+lr.intercept_, color='orange')

            ax.tick_params(axis='x', rotation=35)
            ax.set_title(f"{description} exit {exit} - {df_name}")

            image_name = f'{description}_exit_{exit}_{df_name}'
            filename = os.path.join(savefolder, image_name + '.png')
            fig.savefig(filename)

            print(f"![{image_name}]({image_name + '.png'})")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datafile', type=str, required=True)
    
    
    for month in range(12):
        glob = f'2016_{month + 1:02d}'
        #genHistogram('../evaluations/AlexNet/cuda/saves/AlexNetWithExits/2023-10-31-01-01-09/epoch_19_90.1_91.1.pth/',
        #             glob, 'non_calibrated/AlexNetEE', f'AlexNetEE_{glob}')

        #genHistogram('../evaluations/MobileNet/cuda/saves/MobileNetV2WithExits/2023-08-20-05-20-25/epoch_19_89.7_90.9.pth/',
        #             glob, 'non_calibrated/MobileNetEE', f'MobileNetEE_{glob}')

        genHistogram(f'calibrated/AlexNetWithExits/01/{glob}.csv',
                     'calibrated/AlexNetWithExits/01/', f'Calibrated_AlexNetEE_{glob}')

        genHistogram(f'calibrated/AlexNetWithExits/02/{glob}.csv',
                     'calibrated/AlexNetWithExits/02/', f'Calibrated_AlexNetEE_{glob}')

        genHistogram(f'calibrated/MobileNetWithExits/01/{glob}.csv',
                     'calibrated/MobileNetWithExits/01/', f'Calibrated_MobileNetEE_{glob}')

        genHistogram(f'calibrated/MobileNetWithExits/02/{glob}.csv',
                     'calibrated/MobileNetWithExits/02/', f'Calibrated_MobileNetEE_{glob}')
        