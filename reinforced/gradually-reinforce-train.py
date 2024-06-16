#!/usr/bin/env python3

# Initial Parameters:
# Trained model: ../trained-models/AlexNetWithExits_epoch_19_90.1_91.1.pth
# Initial data: ../MOORE/2016_01.csv

"""
Step 1: Run the inference using the model and the data 
Step 2: Pick the reject list as a new dataset
Step 3: Balance the dataset
Step 4: Train the model with the balanced dataset
Step 5: Pick the best re-trained model
Step 6: Delete all the other models
Step 7: Set the picked model as the new model
Step 8: Set the dataset to the next month
Step 9: Repeat from step 1
"""

import torch
import sys
sys.path.append('..')
from utils.functions import *
import argparse
import os
from models.AlexNet import AlexNetWithExits

def balance_dataset(df):
    normal = df[df['class'] == 0]
    attack = df[df['class'] == 1]
    attack_count = len(attack)
    normal_count = len(normal)
    if normal_count > attack_count:
        normal = normal.sample(n=attack_count)
    else:
        attack = attack.sample(n=normal_count)
    return pd.concat([normal, attack])

def create_rejected_datasets(threshold_attack_exit_1, threshold_normal_exit_1, threshold_attack_exit_2, threshold_normal_exit_2,
                             dataset_file, results_file, exit2_output_file, rejected_output_file):
    
    df = pd.read_csv(dataset_file)
    df = df.rename(columns={'class': 'clazz'})
    
    df_results = pd.read_csv(results_file)

    df['cnf_exit_1'] = df_results['cnf_exit_1']
    df['cnf_exit_2'] = df_results['cnf_exit_2']

    exit2 = df.query('(clazz == 1 and cnf_exit_1 < @threshold_attack_exit_1) or (clazz == 0 and cnf_exit_1 < @threshold_normal_exit_1)')
    exit2_save = exit2.drop(columns=['cnf_exit_1', 'cnf_exit_2']).rename(columns={'clazz': 'class'})
    normal_count = (exit2_save['class'] == 0).sum()
    attack_count = (exit2_save['class'] == 1).sum()
    print(f"Saving to {exit2_output_file} - Total rows before {len(df)} and after {len(exit2)}")
    print(f'\tNormal count: {normal_count} and Attack count: {attack_count}')
    
    exit2_save = balance_dataset(exit2_save)
    print(f"\tAfter balancing - Total rows before {len(exit2)} and after {len(exit2_save)}")
    exit2_save.to_csv(exit2_output_file, index=False)    
    
    rejected = exit2.query('(clazz == 1 and cnf_exit_2 < @threshold_attack_exit_2) or (clazz == 0 and cnf_exit_2 < @threshold_normal_exit_2)')
    rejected_save = rejected.drop(columns=['cnf_exit_1', 'cnf_exit_2']).rename(columns={'clazz': 'class'})
    normal_count = (rejected_save['class'] == 0).sum()
    attack_count = (rejected_save['class'] == 1).sum()
    print(f"Saving to {rejected_output_file} - Total rows before {len(df)} and after {len(rejected)}")
    print(f'\tNormal count: {normal_count} and Attack count: {attack_count}')
    
    rejected_save = balance_dataset(rejected_save)
    print(f"\tAfter balancing - Total rows before {len(rejected)} and after {len(rejected_save)}")
    rejected_save.to_csv(rejected_output_file, index=False)
    
                                                                                                  
if __name__ == '__main__':
    threshold_attack_exit_1 = 0.8374
    threshold_attack_exit_2 = 0.9028
    threshold_normal_exit_1 = 0.8776
    threshold_normal_exit_2 = 0.9214
    
    # get from args
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder', help='Output folder', required=True)
    parser.add_argument('--input-folder', help='Input folder', required=True)
    parser.add_argument('--epochs', help='Number of epochs to train the model', default=15, type=int)
    parser.add_argument('--level', help='If processing all exit2 or only rejected [exit2 | rejected]', default='exit2')
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    batch_size = 1000
    trained_model = '../trained_models/AlexNetWithExits_epoch_19_90.1_91.1.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for month in range(1, 13):
        month = f'{month:02d}'
        dataset_file = os.path.join(args.input_folder, f'2016_{month}.csv')

        # Create and load model
        print(f"Loading {trained_model}")
        model = AlexNetWithExits().to(device)
        model.load_state_dict(torch.load(trained_model))
        model.eval()

        # Process files and generate the result file
        results_csv = os.path.join(args.output_folder, f'results_2016_{month}.csv')

        print(f"Processing {dataset_file} with {trained_model} and saving to {results_csv}")
        
        data   = CustomDataset(as_matrix=True, file=dataset_file)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        dump_2exits(model=model, device=device, loader=loader, savefile=results_csv)
        
        # Create the filtered datasets
        exit2_output_file = os.path.join(args.output_folder, f'exit2_2016_{month}.csv')
        rejected_output_file = os.path.join(args.output_folder, f'rejected_2016_{month}.csv')
        create_rejected_datasets(threshold_attack_exit_1, threshold_normal_exit_1, 
                                 threshold_attack_exit_2, threshold_normal_exit_2, 
                                 dataset_file, results_csv, exit2_output_file, rejected_output_file)
        
        model = AlexNetWithExits().to(device)
        model.load_state_dict(torch.load(trained_model))
        model.train()
        
        train_file = rejected_output_file if args.level == 'rejected' else exit2_output_file
        
        train_data   = CustomDataset(as_matrix=True, file=train_file)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        save_path = os.path.join(args.output_folder, 'trained_models', month)
        train_2exits(model, train_loader=train_loader, device=device, epochs=args.epochs, save_path=save_path)
        
        # list files in save_path
        trained_models = os.listdir(save_path)
        best_model = None
        best_accuracy = 0
        for model_file in trained_models:
            if model_file[-4:] != '.pth':
                continue
            # remove .pth extension
            acc1, acc2 = model_file[:-4].split('_')[-2:]
            acc = (float(acc1) + float(acc2))
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model_file

        print(f'Best model: {best_model} with accuracy {best_accuracy}')
        
        for model_file in trained_models:
            if model_file[-4:] != '.pth':
                continue
            
            if model_file != best_model:
                os.remove(os.path.join(save_path, model_file))
                        
        trained_model = os.path.join(save_path, best_model)
        print(f'Setting model to {trained_model}')