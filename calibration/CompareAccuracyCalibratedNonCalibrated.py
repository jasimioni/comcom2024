#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

BASEDIR = '/home/jasimioni/ppgia/globecom2024/'

def compare_accuracy_calibrated_non_calibrated(calibrated, non_calibrated):
    df_calibrated = pd.read_csv(calibrated)
    df_non_calibrated = pd.read_csv(non_calibrated)

    acc_calibrated_exit_1 = (df_calibrated['y_exit_1'] == df_calibrated['y']).sum() / df_calibrated['y_exit_1'].count()
    acc_calibrated_exit_2 = (df_calibrated['y_exit_2'] == df_calibrated['y']).sum() / df_calibrated['y_exit_2'].count()

    acc_non_calibrated_exit_1 = (df_non_calibrated['y_exit_1'] == df_non_calibrated['y']).sum() / df_non_calibrated['y_exit_1'].count()
    acc_non_calibrated_exit_2 = (df_non_calibrated['y_exit_2'] == df_non_calibrated['y']).sum() / df_non_calibrated['y_exit_2'].count()

    return acc_calibrated_exit_1, acc_calibrated_exit_2, acc_non_calibrated_exit_1, acc_non_calibrated_exit_2

if __name__ == '__main__':
    calibrated = {
        'alexnet': 'calibration/calibrated/AlexNetWithExits',
        'mobilenet': 'calibration/calibrated/MobileNetWithExits',
    } 

    non_calibrated = {
        'alexnet': 'evaluations/AlexNet/cuda/saves/AlexNetWithExits/2023-10-31-01-01-09/epoch_19_90.1_91.1.pth',
        'mobilenet': 'evaluations/MobileNet/cuda/saves/MobileNetV2WithExits/2023-08-20-05-20-25/epoch_19_89.7_90.9.pth',
    }

    for network in ['alexnet', 'mobilenet']:
        for m in range(12):
            month = f'{m+1:02d}'
            name = f"2016_{month}.csv"
            calibrated_file = os.path.join(BASEDIR, calibrated[network], name)
            non_calibrated_file = os.path.join(BASEDIR, non_calibrated[network], name)

            acc_calibrated_exit_1, acc_calibrated_exit_2, acc_non_calibrated_exit_1, acc_non_calibrated_exit_2 = \
                compare_accuracy_calibrated_non_calibrated(calibrated_file, non_calibrated_file)

            print(f'{network} - {month}')
            print(f'\tExit 1 Calibrated | Non Calibrated: {100*acc_calibrated_exit_1:.2f} | {100*acc_non_calibrated_exit_1:.2f}')
            print(f'\tExit 2 Calibrated | Non Calibrated: {100*acc_calibrated_exit_2:.2f} | {100*acc_non_calibrated_exit_2:.2f}')



