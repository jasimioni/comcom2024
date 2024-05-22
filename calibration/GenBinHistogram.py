#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

folder = {
    'AlexNet': '../evaluations/AlexNet/cuda/saves/AlexNetWithExits/2023-10-31-01-01-09/epoch_19_90.1_91.1.pth/',
    'MobileNet': '../evaluations/MobileNet/cuda/saves/MobileNetV2WithExits/2023-08-20-05-20-25/epoch_19_89.7_90.9.pth/'
}

glob = '2016_02'

plt.rc('font', family='serif', size=26)

for network in [ 'AlexNet', 'MobileNet' ]:
    directory = folder[network]
    files = Path(directory).glob(f'*{glob}*')
    dfs = []
    for file in sorted(files):
        dfs.append(pd.read_csv(file))        

    df_total = pd.concat(dfs, ignore_index=True)

    n_bin = 10

    print(f"## Network: {network}")

    dfs

    for exit in [ 1, 2 ]:

        dfs = {
            'total': df_total,
            'normal': df_total.query("y == 0"),
            'attack': df_total.query("y == 1")
        }

        for df_name in [ 'total', 'normal', 'attack' ]:
            df = dfs[df_name]

            max = df[f"cnf_exit_{exit}"].max()
            min = df[f"cnf_exit_{exit}"].min()

            step = ( max - min ) / n_bin

            print(f"### Exit {exit} | {df_name}")

            bin_acc = []
            bins = []

            print("```")
            for bin in range(n_bin):
                # y  y_exit_1  cnf_exit_1  y_exit_2  cnf_exit_2
                bin_min = min + bin * step
                bin_max = min + ( bin + 1 ) * step

                df_bin = df.query(f"cnf_exit_{exit} > {bin_min} and cnf_exit_{exit} <= {bin_max}")
                total = df_bin['y'].count()
                correct = df_bin.query(f"y == y_exit_{exit}")['y'].count() 

                accuracy = correct / total

                bin_acc.append(accuracy)
                bins.append(f"{bin_min:.2f}-")
                
                print(f"{bin_min:.4f} < x <= {bin_max:.4f}: {total} | {correct} | {accuracy:.4f}")

            fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))
            print(bins)
            print(bin_acc)
            print("```")
            ax.bar(bins, bin_acc)
            ax.tick_params(axis='x', rotation=35)
            ax.set_title(f"{network} - exit {exit} - {df_name}")

            image_name = f'{network}_{exit}_{df_name}'
            fig.savefig(image_name + '.png')

            print(f"![{image_name}]({image_name + '.png'})")