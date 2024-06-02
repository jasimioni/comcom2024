#!/usr/bin/env python3

import pandas as pd
import os
import sys

BASEDIR = '/home/jasimioni/ppgia/globecom2024/'

calibrated = {
    'alexnet': 'calibration/calibrated/AlexNetWithExits',
    'mobilenet': 'calibration/calibrated/MobileNetWithExits',
} 

non_calibrated = {
    'alexnet': 'evaluations/rpi_evals/AlexNet/AlexNetWithExits_epoch_19_90.1_91.1.pth',
    'mobilenet': 'evaluations/rpi_evals/MobileNet/MobileNetV2WithExits_epoch_19_89.7_90.9.pth',
}

for network in ['alexnet', 'mobilenet']:
    for m in range(12):
        month = f'{m+1:02d}'
        filename = f"2016_{month}.csv"

        calibrated_filename = os.path.join(BASEDIR, calibrated[network], filename)

        df_calibrated = pd.read_csv(calibrated_filename)
        df_non_calibrated = pd.read_csv(os.path.join(BASEDIR, non_calibrated[network], f'short_{filename}'))

        bb_time_exit_1 = df_non_calibrated['bb_time_exit_1'].mean()
        exit_time_exit_1 = df_non_calibrated['exit_time_exit_1'].mean()
        bb_time_exit_2 = df_non_calibrated['bb_time_exit_2'].mean()
        exit_time_exit_2 = df_non_calibrated['exit_time_exit_2'].mean()

        df_calibrated['bb_time_exit_1'] = bb_time_exit_1
        df_calibrated['exit_time_exit_1'] = exit_time_exit_1
        df_calibrated['bb_time_exit_2'] = bb_time_exit_2
        df_calibrated['exit_time_exit_2'] = exit_time_exit_2

        df_calibrated.to_csv(calibrated_filename, index=False)
    