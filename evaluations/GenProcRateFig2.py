#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

list = [
    [ 'an_rpi', 'icc2024/evaluations/rpi_evals/AlexNet/AlexNet_epoch_16_91.2.pth' ],
    [ 'an_cpu', 'icc2024/evaluations/AlexNet/short_cpu/saves/AlexNet/2023-10-31-01-48-09/epoch_16_91.2.pth' ],
    [ 'an_gpu', 'icc2024/evaluations/AlexNet/short_cuda/saves/AlexNet/2023-10-31-01-48-09/epoch_16_91.2.pth' ],
    [ 'mn_rpi', 'icc2024/evaluations/rpi_evals/MobileNet/MobileNetV2_epoch_17_90.9.pth' ],
    [ 'mn_cpu', 'icc2024/evaluations/MobileNet/short_cpu/saves/MobileNetV2/2023-10-26-04-42-32/epoch_17_90.9.pth' ],
    [ 'mn_gpu', 'icc2024/evaluations/MobileNet/short_cuda/saves/MobileNetV2/2023-10-26-04-42-32/epoch_17_90.9.pth' ],
    [ 'an_gpu_full', 'icc2024/evaluations/AlexNet/cuda/saves/AlexNet/2023-10-31-01-48-09/epoch_16_91.2.pth' ],
    [ 'mn_gpu_full', 'icc2024/evaluations/MobileNet/cuda/saves/MobileNetV2/2023-10-26-04-42-32/epoch_17_90.9.pth' ]
]

def get_time(directory):
    year = '2016'
    dfs = []
    for month in range(1, 13):
        glob = f'{year}_{month:02d}'
        files = Path(directory).glob(f'*{glob}*')
        for file in sorted(files):
            dfs.append(pd.read_csv(file))        
        
    df = pd.concat(dfs, ignore_index=True)

    total_time = df['avg_time'].sum()
    count = df['avg_time'].count()

    return total_time / count

for name, folder in list:
    avg_time = get_time(folder)
    rate = 1 / avg_time
    print(f"{name}: {rate:.2f}")


'''
plt.rc('font', family='serif', size=16)

y_max = max([ an_rpi_rate, mn_rpi_rate ])
y_max += 1
fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))
ax.set(ylim=(-2.5, y_max), ylabel='Inference Rate (sample/s)')
ax.tick_params(axis='x', rotation=45)

ax.bar([ 'AlexNet', 'MobileNet' ], [ an_rpi_rate, mn_rpi_rate ])

fig.savefig(f'icc2024/evaluations/rpi_inference_rate.pdf')

y_max = max([ an_cpu_rate, mn_cpu_rate, mn_gpu_rate, an_gpu_rate ])
y_max = int(y_max / 1000) * 1000 + 1000

fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))
ax.set(ylim=(-2.5, y_max), ylabel='Inference Rate (sample/s)')
ax.tick_params(axis='x', rotation=45)

ax.bar([ 'AlexNet CPU', 'AlexNet GPU', 'MobileNet CPU', 'MobileNet GPU' ], [ an_cpu_rate, an_gpu_rate, mn_cpu_rate, mn_gpu_rate ])

fig.savefig(f'icc2024/evaluations/computer_inference_rate.pdf')
'''