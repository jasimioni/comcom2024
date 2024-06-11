#!/usr/bin/env python3

import pandas as pd
import os
import argparse

'''
./generate-datasets-for-retraining.py --dataset-folder ../MOORE/ --input-results-folder original/AlexNetWithExits/ \
                                      --output-results-folder-exit-2 filtered/AlexNetWithExits/exit2/              \
                                      --output-results-folder-rejected filtered/AlexNetWithExits/rejected/         \
                                      --threshold-attack-exit-1 0.9 \
                                      --threshold-attack-exit-2 0.9 \
                                      --threshold-normal-exit-1 0.9 \
                                      --threshold-normal-exit-2 0.9

./generate-datasets-for-retraining.py --dataset-folder ../MOORE/ --input-results-folder original/MobileNetV2WithExits/ \
                                      --output-results-folder-exit-2 filtered/MobileNetV2WithExits/exit2/              \
                                      --output-results-folder-rejected filtered/MobileNetV2WithExits/rejected/         \
                                      --threshold-attack-exit-1 0.9 \
                                      --threshold-attack-exit-2 0.9 \
                                      --threshold-normal-exit-1 0.9 \
                                      --threshold-normal-exit-2 0.9

'''

parser = argparse.ArgumentParser()

parser.add_argument('--dataset-folder',
                    help='Raw dataset folder to read from',
                    required=True)  

parser.add_argument('--input-results-folder',
                    help='Folder to read from where the confidence values are stored',
                    required=True)

parser.add_argument('--output-results-folder-exit-2',
                    help='Folder where the filtered inputs will be stored for exit 2',
                    required=True)

parser.add_argument('--output-results-folder-rejected',
                    help='Folder where the filtered inputs will be stored for rejected inputs',
                    required=True)

parser.add_argument('--threshold-attack-exit-1',
                    help='Threshold for attack exit 1',
                    type=float,
                    required=True)

parser.add_argument('--threshold-attack-exit-2',
                    help='Threshold for attack exit 2',
                    type=float,
                    required=True)  

parser.add_argument('--threshold-normal-exit-1',
                    help='Threshold for normal exit 1',
                    type=float,
                    required=True)

parser.add_argument('--threshold-normal-exit-2',
                    help='Threshold for normal exit 2',
                    type=float,
                    required=True)

args = parser.parse_args()

if not os.path.isdir(args.output_results_folder_exit_2):
    os.makedirs(args.output_results_folder_exit_2)

if not os.path.isdir(args.output_results_folder_rejected):
    os.makedirs(args.output_results_folder_rejected)

for month in range(1, 13):
    ds_file = os.path.join(args.dataset_folder, f'2016_{month:02d}.csv')
    df = pd.read_csv(ds_file)

    input_file = os.path.join(args.input_results_folder, f'2016_{month:02d}.csv')
    df_input = pd.read_csv(input_file)

    df = df.rename(columns={'class': 'clazz'})

    df['cnf_exit_1'] = df_input['cnf_exit_1']
    df['cnf_exit_2'] = df_input['cnf_exit_2']

    # Filter entries where y == 1 and cnf_exit1 < threshold_attack_exit_1
    # or y = 0 and cnf_exit1 < threshold_normal_exit_1
    
    exit2 = df.query('(clazz == 0 and cnf_exit_1 < @args.threshold_attack_exit_1) or (clazz == 0 and cnf_exit_1 < @args.threshold_normal_exit_1)')

    # Save the filtered entries to output_results_folder_exit_2
    dst_filename = os.path.join(args.output_results_folder_exit_2, f'2016_{month:02d}.csv')
    print(f'Saving to {dst_filename} - Total rows before {len(df)} and after {len(exit2)}')
    exit2.drop(columns=['cnf_exit_1', 'cnf_exit_2']).rename(columns={'clazz': 'class'}).to_csv(dst_filename, index=False)    

    # Filter entries where y == 1 and cnf_exit2 < threshold_attack_exit_2
    # or y = 0 and cnf_exit2 < threshold_normal_exit_2

    rejected = exit2.query('(clazz == 1 and cnf_exit_2 < @args.threshold_attack_exit_2) or (clazz == 0 and cnf_exit_2 < @args.threshold_normal_exit_2)')

    # Save the filtered entries to output_results_folder_rejected
    dst_filename = os.path.join(args.output_results_folder_rejected, f'2016_{month:02d}.csv')
    print(f'Saving to {dst_filename} - Total rows before {len(df)} and after {len(rejected)}')
    rejected.drop(columns=['cnf_exit_1', 'cnf_exit_2']).rename(columns={'clazz': 'class'}).to_csv(dst_filename, index=False)