#!/usr/bin/env python3

import pandas as pd
import os

dataset_folder = '../MOORE/'
results_folder = 'original/MobileNetV2WithExits/'

for month in range(1, 13):
    ds_file = os.path.join(dataset_folder, f'2016_{month:02d}.csv')
    rs_file = os.path.join(results_folder, f'2016_{month:02d}.csv')

    df_ds = pd.read_csv(ds_file)
    df_rs = pd.read_csv(rs_file)

    # Extract only column 'class' from df_ds and 'y' from df_rs

    df_ds = df_ds[['class']]
    df_ds['y'] = df_rs[['y']]

    # Count the number of rows and the number of equal rows

    n_rows = len(df_ds)
    n_equal = len(df_ds[df_ds['class'] == df_ds['y']])

    print(f'2016_{month:02d}: {n_equal}/{n_rows} ({100 * n_equal/n_rows:.2f}%)')



