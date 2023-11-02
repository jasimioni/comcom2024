#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

basedir = 'icc2024/evaluations'

list = [
    [ 'an_rpi_short', 'rpi_evals/AlexNet/AlexNetWithExits_epoch_19_90.1_91.1.pth' ],
    [ 'an_cpu_short', 'AlexNet/short_cpu/saves/AlexNetWithExits/2023-10-31-01-01-09/epoch_19_90.1_91.1.pth' ],
    [ 'an_gpu_short', 'AlexNet/short_cuda/saves/AlexNetWithExits/2023-10-31-01-01-09/epoch_19_90.1_91.1.pth' ],
    [ 'an_gpu', 'AlexNet/cuda/saves/AlexNetWithExits/2023-10-31-01-01-09/epoch_19_90.1_91.1.pth' ],
    [ 'mn_rpi_short', 'rpi_evals/MobileNet/MobileNetV2WithExits_epoch_19_89.7_90.9.pth' ],
    [ 'mn_cpu_short', 'MobileNet/short_cpu/saves/MobileNetV2WithExits/2023-08-20-05-20-25/epoch_19_89.7_90.9.pth' ],
    [ 'mn_gpu_short', 'MobileNet/short_cuda/saves/MobileNetV2WithExits/2023-08-20-05-20-25/epoch_19_89.7_90.9.pth' ],
    [ 'mn_gpu', 'MobileNet/cuda/saves/MobileNetV2WithExits/2023-08-20-05-20-25/epoch_19_89.7_90.9.pth' ]
]

def get_time(directory):
    directory = os.path.join(basedir, directory)
    dfs = []
    glob = f'2016'
    files = Path(directory).glob(f'*{glob}*')
    for file in sorted(files):
        dfs.append(pd.read_csv(file))        
        
    df = pd.concat(dfs, ignore_index=True)

    count = df['y'].count()

    exit_1_total_time = df['bb_time_exit_1'].sum() + df['exit_time_exit_1'].sum()
    exit_2_total_time = df['bb_time_exit_1'].sum() + df['bb_time_exit_2'].sum() + df['exit_time_exit_2'].sum()

    exit_1_avg_time = exit_1_total_time / count
    exit_2_avg_time = exit_2_total_time / count

    return exit_1_avg_time, exit_2_avg_time

for name, folder in list:
    exit_1_avg_time, exit_2_avg_time = get_time(folder)
    rate_exit_1 = 1 / exit_1_avg_time
    rate_exit_2 = 1 / exit_2_avg_time

    print(f"{name}|{rate_exit_1:.2f}|{rate_exit_2:.2f}|({rate_exit_1 / rate_exit_2:.2f})")