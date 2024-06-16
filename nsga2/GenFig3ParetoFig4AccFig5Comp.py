#!/usr/bin/env python3

import os
import sys
sys.path.append('..')
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import scipy

import matplotlib.pyplot as plt

print(os.getcwd())

BASEDIR = os.environ.get('BASEDIR') or '/home/jasimioni/ppgia/'
sys.path.append(f'{BASEDIR}')
from utils.eval_functions import *

MONTHS_NAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

calibration = [ 'non_calibrated', 93, 66 ]
# calibration = [ 'calibrated', 85, 27 ]

net_data = {
    'alexnet' : {
        'rpi_folder' : f'{BASEDIR}/evaluations/rpi_evals/AlexNet/AlexNetWithExits_epoch_19_90.1_91.1.pth',
        'cuda_folder' : f'{BASEDIR}/calibration/{calibration[0]}/AlexNetWithExits',
        'nsga_result_file' : f'{BASEDIR}/nsga2/{calibration[0]}/alexnet_x_f_0.9_2016_23.sav',
        'rpi_single_exit_folder' : f'{BASEDIR}/evaluations/rpi_evals/AlexNet/AlexNet_epoch_16_91.2.pth/',
        'cuda_single_exit_folder' : f'{BASEDIR}/calibration/{calibration[0]}/AlexNet',
        'operation_point' : calibration[1],
        'label' : 'AlexNet'
    },
    'mobilenet' : {
        'rpi_folder' : f'{BASEDIR}/evaluations/rpi_evals/MobileNet/MobileNetV2WithExits_epoch_19_89.7_90.9.pth',
        'cuda_folder' : f'{BASEDIR}/calibration/{calibration[0]}/MobileNetWithExits',
        'nsga_result_file' : f'{BASEDIR}/nsga2/{calibration[0]}/mobilenet_x_f_0.9_2016_23.sav',
        'rpi_single_exit_folder' : f'{BASEDIR}/evaluations/rpi_evals/MobileNet/MobileNetV2_epoch_17_90.9.pth/',
        'cuda_single_exit_folder' : f'{BASEDIR}/calibration/{calibration[0]}/MobileNet',
        'operation_point' : calibration[2],
        'label' : 'MobileNetV2'
    }
}

def remove_dominated(results):
    to_delete = []
    for i in range(len(results)):
        for f in range(len(results)):
            if f == i or ( len(to_delete) and to_delete[-1] == i ):
                continue
            if results[f][1] <= results[i][1] and results[f][2] <= results[i][2]:
                to_delete.append(i)

    # print(f'to_delete: {to_delete}')
    to_delete.reverse()
    for i in to_delete:
        results.pop(i)
    return len(to_delete)
    
for network in net_data.keys():
    glob = f'2016_0[23]'
    # Single exit time

    files = Path(net_data[network]['rpi_single_exit_folder']).glob(f'*{glob}*')
    dfs = []
    for e_file in sorted(files):
        dfs.append(pd.read_csv(e_file))
    
    df = pd.concat(dfs, ignore_index=True)

    count = df['y'].count()
    total_time = df['avg_time'].sum()
    avg_time = total_time / count

    correct = df.query('y == y_pred')['y'].count()

    error = ( count - correct ) / count

    net_data[network]['single_exit_time'] = avg_time
    net_data[network]['single_exit_error'] = error

    # Multi-exit calculations
    files = Path(net_data[network]['rpi_folder']).glob(f'*{glob}*')
    dfs = []
    for e_file in sorted(files):
        dfs.append(pd.read_csv(e_file))
    
    df = pd.concat(dfs, ignore_index=True)

    count = df['y'].count()

    exit_1_total_time = df['bb_time_exit_1'].sum() + df['exit_time_exit_1'].sum()
    exit_2_total_time = df['bb_time_exit_1'].sum() + df['bb_time_exit_2'].sum() + df['exit_time_exit_2'].sum()

    exit_1_avg_time = exit_1_total_time / count
    exit_2_avg_time = exit_2_total_time / count    

    net_data[network]['rpi_times'] = [ exit_1_avg_time, exit_2_avg_time ]

    with open(net_data[network]['nsga_result_file'], 'rb') as f:
        X, F, min_time, max_time, accuracy_e1, acceptance_e1, accuracy_e2, acceptance_e2 = pickle.load(f)

    min_time = net_data[network]['rpi_times'][0]
    max_time = net_data[network]['rpi_times'][1]

    net_data[network]['nsga_data'] = {
        'X' : X,
        'F' : F,
        'min_time' : min_time, 
        'max_time' : max_time,
        'accuracy_e1' : accuracy_e1,
        'acceptance_e1' : acceptance_e1,
        'accuracy_e2' : accuracy_e2,
        'acceptance_e2' : acceptance_e2,
    }

    print(f'{network} e1: Accuracy: {100 * accuracy_e1:.2f}% - Acceptance: {100 * acceptance_e1:.2f}% - Cost: {min_time:.4f}s')
    print(f'{network} e2: Accuracy: {100 * accuracy_e2:.2f}% - Acceptance: {100 * acceptance_e2:.2f}% - Cost: {max_time:.4f}s\n')

    results = []

    for i in range(len(F)):
        f = F[i]
        x = X[i]
        quality = 100 * (1 - sum(f) / len(f))

        acc  = 100 * ( 1 - f[0] )
        time = min_time + f[1] * (max_time - min_time) 

        string = f'Score: {quality:.2f}% | Acc: {acc:.2f}% Time: {time:.2f}s | {x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}, {x[3]:.4f}'
        results.append([ quality, *f, *x, string ])

        # print(f'{network}: {i:02d}: {quality:.2f} {100 * (1-f[0]):.2f} {100*(1-f[1]):.2f} => {string}')

    while remove_dominated(results):
        pass

    net_data[network]['results'] = results

    df = pd.DataFrame(results, columns = ['Quality', 'Accuracy', 'Time', 'n_1', 'a_1', 'n_2', 'a_2', 'String'])
    df['Score'] = ( df['Accuracy'] + df['Time'] ) / 2
    df['Distance'] = df['Accuracy'] ** 2 + df['Time'] ** 2

    df = df.sort_values(by='Time')

    net_data[network]['df'] = df

    seq = 0
    for i, row in df.iterrows():
        print(f"{network} - {seq:02d}: {row['Quality']:.2f} {100 * (1-row['Accuracy']):.2f} {100*(1-row['Time']):.2f} => {row['String']}")
        seq += 1

if __name__ == '__main__':
    plt.rc('font', family='serif', size=30)

    y_max = 11
    y_min = 6
    
    x_max = 30
    x_min = -0.5

    colors = [ 'black', 'red', 'blues' ]
    markers = [ 'o', '^' ]

    fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 6.5))
    ax.set(ylim=(y_min, y_max), ylabel='Error Rate (%)')
    # ax.tick_params(axis='x', rotation=45)

    # plt.grid(linestyle = '--', linewidth = 0.5)

    arr_x = 18
    arr_y = 9.3

    for i, network in enumerate(sorted(net_data.keys())):
    # for i, network in enumerate(['mobilenet']):

        min_time = net_data[network]['nsga_data']['min_time']
        max_time = net_data[network]['nsga_data']['max_time']

        df = net_data[network]['df']
        y = 1 - df['Accuracy'].to_numpy()
        x = df['Time'].to_numpy()
        score = df['Score'].to_numpy()
        names = df['String'].to_numpy()    

        y1 = 100 * ( 1 - y )
        # x1 = min_time + x * (max_time - min_time)
        x1 = x * 100

        operation_point = df.iloc[net_data[network]['operation_point']]
        print(operation_point)
        y_op = operation_point['Accuracy']
        y1_op = 100 * y_op
        x_op = operation_point['Time']
        x1_op = x_op * 100


        ax.arrow(arr_x, arr_y, x1_op - arr_x, y1_op - arr_y, head_width=0.05, head_length=0.2, 
                 fc='k', ec='k', length_includes_head=True, zorder=15)
        # ax.scatter(x1, y1, s=70, marker=markers[i], label=net_data[network]['label'], alpha=0.5)
        ax.plot(x1, y1, linewidth=2, label=net_data[network]['label'])

        # ax.scatter(x1_op, y1_op, s=40)

        avg_time = min_time + x_op * ( max_time + min_time )
        print(f'{network} - Error rate: {y1_op}% | Time: {1000 * avg_time:.2f}ms')
        net_data[network]['multi_exit_time'] = avg_time
        net_data[network]['multi_exit_error'] = y_op
    
    ax.text(arr_x - 3, arr_y + 0.1, 'Operation Point', fontsize=18)
    ax.legend(loc='upper right', frameon=False, fontsize=18)
    ax.set(xlim=(x_min, x_max), xlabel='Normalized Processing Time')
    fig.savefig(f'{BASEDIR}/nsga2/nsga_pareto_{calibration[0]}.pdf')


    fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 6.5))

    y_max = 11
    y_min = 6
    
    x_max = 0.15 * 1000
    x_min = 15

    ax.set(xlim=(x_min, x_max), xlabel='Processing time (ms)')
    ax.set(ylim=(y_min, y_max), ylabel='Error rate (%)')
    # ax.tick_params(axis='x', rotation=45)

    for i, network in enumerate(sorted(net_data.keys())):
        single_exit_time = 1000 * net_data[network]['single_exit_time']
        single_exit_error = 100 * net_data[network]['single_exit_error']
        multi_exit_time = 1000 * net_data[network]['multi_exit_time']
        multi_exit_error = 100 * net_data[network]['multi_exit_error']
        print(f"{network}: {single_exit_time} {single_exit_error} {multi_exit_time} {multi_exit_error}")
        op = net_data[network]['df'].iloc[net_data[network]['operation_point']]
        print(f"{network} Operation points: {op['n_1']} | {op['a_1']} | {op['n_2']} | {op['a_2']}")

        x = [ single_exit_time, multi_exit_time ]
        y = [ single_exit_error, multi_exit_error ]

        error_improvement = single_exit_error - multi_exit_error
        time_improvement = 100 * (single_exit_time - multi_exit_time) / single_exit_time

        text = f'- {error_improvement:.1f} Error\n-{time_improvement:.0f}% Time'
        
        ax.scatter(single_exit_time, single_exit_error, s=300, marker=markers[i], label=f"Trad. {net_data[network]['label']}", alpha=1, color=colors[i])
        ax.scatter(multi_exit_time, multi_exit_error, s=300, marker=markers[i], label=f"Ours {net_data[network]['label']}", alpha=0.3, color=colors[i])
        ax.text(multi_exit_time + 3.2, multi_exit_error - 0.5, text, fontsize=14)
        ax.plot(x, y, linestyle='dashed', linewidth=2, color=colors[i])

    ax.legend(loc='upper left', frameon=False, fontsize=18)
    # plt.grid(linestyle = '--', linewidth = 0.5)
    
    fig.savefig(f'{BASEDIR}/nsga2/paretocomp_{calibration[0]}.pdf')

    ### GENERATING FIG4 - ACCURACY

    for i, network in enumerate(sorted(net_data.keys())):
        fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))

        x_max = 0.15 * 1000
        x_min = 15

        #ax.set(xlim=(x_min, x_max))
        #ax.set(ylim=(y_min, y_max), ylabel='Error rate (%)')
        ax.set(ylabel='Accuracy (%)')
        ax.tick_params(axis='x', rotation=50)

        op = net_data[network]['df'].iloc[net_data[network]['operation_point']]
        print(f"{network} Operation points: {op['n_1']} | {op['a_1']} | {op['n_2']} | {op['a_2']}")

        op_results = {
            'acc' : [],
            'f1' : [],
            'exit_rate' : []
        }

        for month in range(1, 13):
            glob = f'2016_{month:02d}.csv'
            files = Path(net_data[network]['cuda_folder']).glob(f'*{glob}*')

            print(net_data[network]['cuda_folder'])
            dfs = []
            for e_file in sorted(files):
                print(e_file)
                dfs.append(pd.read_csv(e_file))
            
            df = pd.concat(dfs, ignore_index=True)

            accuracy, fpr, fnr, precision, recall, f1, acceptance_rate, exit1_rate = get_model_results(df, op['n_1'], op['a_1'], op['n_2'], op['a_2'])

            print(' | '.join([ f'{val:.2f}' for val in [ accuracy, fpr, fnr, precision, recall, f1, acceptance_rate, exit1_rate ] ]))

            op_results['acc'].append(accuracy)
            op_results['f1'].append(f1)
            op_results['exit_rate'].append(exit1_rate)

        net_data[network]['op_results'] = op_results

        print(op_results['acc'])
        ax.plot(MONTHS_NAME, [ 100 * acc for acc in op_results['acc'] ], label='Accuracy', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='black')

        # ax.legend(loc='upper right', frameon=False, fontsize=18)
        
        fig.savefig(f'{BASEDIR}/nsga2/accuracyprop{network}_{calibration[0]}.pdf')

    ### FIG 5 
    network = 'alexnet'
    op_results = net_data[network]['op_results']
    f1_single = []
    for month in range(1, 13):
        glob = f'2016_{month:02d}.csv'
        files = Path(net_data[network]['cuda_single_exit_folder']).glob(f'*{glob}*')
        dfs = []
        for e_file in sorted(files):
            dfs.append(pd.read_csv(e_file))
        
        df = pd.concat(dfs, ignore_index=True)

        tn = df.query('y == 0 and y_pred == 0')['y'].count()
        tp = df.query('y == 1 and y_pred == 1')['y'].count()
        fn = df.query('y == 1 and y_pred == 0')['y'].count()
        fp = df.query('y == 0 and y_pred == 1')['y'].count()
        total = df['y'].count()

        accuracy = (tn + tp) / total
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1 = 2 * precision * recall / (precision + recall)

        f1_single.append(f1)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))

    x_max = 0.15 * 1000
    x_min = 15

    #ax.set(xlim=(x_min, x_max))
    #ax.set(ylim=(y_min, y_max), ylabel='F1')
    ax.set(ylabel='F1')
    ax.tick_params(axis='x', rotation=50)    
    ax.plot(MONTHS_NAME, op_results['f1'], label='Ours', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='black')
    ax.plot(MONTHS_NAME, f1_single, label='Trad.', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='red')

    ax.legend(loc='upper right', frameon=False, fontsize=18)
    fig.savefig(f'{BASEDIR}/nsga2/f1comp_{calibration[0]}.pdf')
    
    # FIG 6
    network = 'alexnet'
    files = Path(net_data[network]['rpi_single_exit_folder']).glob(f'*')
    dfs = []
    for e_file in sorted(files):
        dfs.append(pd.read_csv(e_file))
    df = pd.concat(dfs, ignore_index=True)
    avg_time_single_exit = df['avg_time'].sum() / df['avg_time'].count()
    print(f"Average Time: {avg_time_single_exit}")

    times = []

    for month in range(1, 13):
        glob = f'2016_{month:02d}'
        files = Path(net_data[network]['rpi_folder']).glob(f'*{glob}*')
        dfs = []
        for e_file in sorted(files):
            dfs.append(pd.read_csv(e_file))
        df = pd.concat(dfs, ignore_index=True)

        count = df['y'].count()

        bb_time_exit_1 = df['bb_time_exit_1'].sum() / count
        exit_time_exit_1 = df['exit_time_exit_1'].sum() / count
        bb_time_exit_2 = df['bb_time_exit_2'].sum() / count
        exit_time_exit_2 = df['exit_time_exit_2'].sum() / count

        min_time = bb_time_exit_1 + exit_time_exit_1
        max_time = bb_time_exit_1 + bb_time_exit_2 + exit_time_exit_2

        print(f'{month:02d}: Min time: {min_time * 1000:.2f}, Max time: {max_time * 1000:.2f}')
        time = min_time + ( max_time - min_time ) * ( 1 - net_data[network]['op_results']['exit_rate'][month - 1] )
        times.append(time * 1000)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))

    y_min = 0
    y_max = 160

    #ax.set(xlim=(x_min, x_max))
    ax.set(ylim=(y_min, y_max), ylabel='Error rate (%)')
    ax.set(ylabel='Inf. Time (ms)')
    ax.tick_params(axis='x', rotation=50)

    ax.plot(MONTHS_NAME, times, label='Ours', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='black')
    ax.plot(MONTHS_NAME, [ 1000 * avg_time_single_exit for x in range(0, 12) ], label='Trad.', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='red')
    ax.legend(loc='center right', frameon=False, fontsize=18)
    fig.savefig(f'{BASEDIR}/nsga2/proccomp_{calibration[0]}.pdf')
    


