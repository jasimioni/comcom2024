#!/usr/bin/env python3

import pandas as pd
import os
import sys

# Adiciona os tempos do raspberry para os resultados calibrados
# para poder fazer as avaliações posteriorment

BASEDIR = os.environ.get('BASEDIR') or '/home/jasimioni/ppgia/greencomputing2024/'

rpi_times = {
    'alexnet': 'evaluations/rpi_evals/AlexNet/AlexNetWithExits_epoch_19_90.1_91.1.pth',
    'mobilenet': 'evaluations/rpi_evals/MobileNet/MobileNetV2WithExits_epoch_19_89.7_90.9.pth',
}

target = {
    'alexnet': 'calibration/calibrated/AlexNetWithExits',
    'mobilenet': 'calibration/calibrated/MobileNetWithExits',
} 

for network in ['alexnet', 'mobilenet']:
    for calibration in [ '01', '02', '03' ]:
        for m in range(12):
            month = f'{m+1:02d}'
            filename = f"2016_{month}.csv"

            target_filename = os.path.join(BASEDIR, target[network], calibration, filename)

            df_target = pd.read_csv(target_filename)
            df_rpi_times = pd.read_csv(os.path.join(BASEDIR, rpi_times[network], f'short_{filename}'))

            bb_time_exit_1 = df_rpi_times['bb_time_exit_1'].mean()
            exit_time_exit_1 = df_rpi_times['exit_time_exit_1'].mean()
            bb_time_exit_2 = df_rpi_times['bb_time_exit_2'].mean()
            exit_time_exit_2 = df_rpi_times['exit_time_exit_2'].mean()

            df_target['bb_time_exit_1'] = bb_time_exit_1
            df_target['exit_time_exit_1'] = exit_time_exit_1
            df_target['bb_time_exit_2'] = bb_time_exit_2
            df_target['exit_time_exit_2'] = exit_time_exit_2

            df_target.to_csv(target_filename, index=False)
    
