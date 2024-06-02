#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import os
from sklearn.linear_model import Ridge


plt.rc('font', family='serif', size=16)

def genHistogram(directory, glob, savefolder, description):
    files = Path(directory).glob(f'*{glob}*')
    dfs = []
    for file in sorted(files):
        dfs.append(pd.read_csv(file))        

    df_total = pd.concat(dfs, ignore_index=True)

    os.makedirs(savefolder, exist_ok=True)

    n_bin = 10

    print(f"## Network: {description}")

    dfs = {
        'total': df_total,
        'normal': df_total.query("y == 0"),
        'attack': df_total.query("y == 1")
    }

    for df_name in [ 'total', 'normal', 'attack' ]:
        df = dfs[df_name]

        max = df["cnf"].max()
        min = df["cnf"].min()

        step = ( max - min ) / n_bin

        print(f"### {df_name}")

        bin_acc = []
        bins = []
        bin_labels = []

        print("```")
        for bin in range(n_bin):
            # y  y_pred cnf
            bin_min = min + bin * step
            bin_max = min + ( bin + 1 ) * step

            df_bin = df.query(f"cnf > {bin_min} and cnf <= {bin_max}")
            total = df_bin['y'].count()
            correct = df_bin.query("y == y_pred")['y'].count() 

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
        ax.set_title(f"{description} - {df_name}")

        image_name = f'{description}_1exit_{df_name}'
        filename = os.path.join(savefolder, image_name + '.png')
        fig.savefig(filename)

        print(f"![{image_name}]({image_name + '.png'})")

if __name__ == '__main__':
    for month in range(12):
        glob = f'2016_{month + 1:02d}'
        genHistogram('../evaluations/AlexNet/cuda/saves/AlexNet/2023-10-31-01-48-09/epoch_16_91.2.pth/',
                     glob, 'pre_calibration/AlexNet', f'AlexNet_{glob}')

        genHistogram('../evaluations/MobileNet/cuda/saves/MobileNetV2/2023-10-26-04-42-32/epoch_17_90.9.pth/',
                     glob, 'pre_calibration/MobileNet', f'MobileNet_{glob}')

        genHistogram('calibrated/AlexNet',
                     glob, 'calibrated/AlexNet', f'Calibrated_AlexNet_{glob}')

        genHistogram('calibrated/MobileNet',
                     glob, 'calibrated/MobileNet', f'Calibrated_MobileNet_{glob}')
        