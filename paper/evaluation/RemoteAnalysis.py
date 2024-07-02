#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def gen_list(df_local, df_remote, network, datacenter, input_size, column):
    results = []
    print(f"Input size: {input_size}")
    for batch_size in [1, 5, 10]:
        results.append(df_local[(df_local['network'] == network) &
                                (df_local['input_size'] == input_size) &
                                (df_local['batch_size'] == batch_size) & 
                                (df_local['datacenter'] == datacenter)][column].values[0])
        results.append(df_remote[(df_remote['network'] == network) &
                                (df_remote['input_size'] == input_size) &
                                (df_remote['batch_size'] == batch_size) & 
                                (df_remote['datacenter'] == datacenter)][column].values[0])
    print(results)
    return np.array(results)

# Get script real dir
script_dir = os.path.dirname(__file__)
file = os.path.join(script_dir, 'remote_results.tsv')


'''
network: AlexNet or MobileNet
datacenter: SP or DC
input_size: Tests with 8x8 and 48x48
batch_size: 1, 5 and 10
e1_local_time: local time to process e1
e1_process_time: local process time (energy) e1
e2_local_time: local time to process e2
e2_process_time: local process time (energy) e2
request_local_time: time spent to do the request. It's e2_remote_time + network_time
request_process_time: local process time (energy) to remotely request
e2_remote_time: time to run e2 remotely
network_time: time to send the data to the network (request_local_time - e2_remote_time)
data_size: size of the pickle data sent
'''

df_local = pd.read_csv(file, sep='\t')
df_remote = df_local.copy()

print(df_local)

# For local processing
# No request_time, so it is 0
# No e2_remote_time, so it is 0
df_local['total_time'] = df_local['e1_local_time'] + df_local['e2_local_time']
df_local['total_process_time'] = df_local['e1_process_time'] + df_local['e2_process_time']
df_local['request_local_time'] = 0
df_local['request_process_time'] = 0
df_local['e2_remote_time'] = 0
df_local['network_time'] = 0

# For offload procesing
# total time is e1_local_time + e2_remote_time + network_time
# No e2_local_time, so it is 0
df_remote['total_time'] = df_remote['e1_local_time'] + df_remote['e2_remote_time'] + df_remote['network_time']
df_remote['total_process_time'] = df_remote['e1_process_time'] + df_remote['request_process_time']
df_remote['e2_local_time'] = 0
df_remote['e2_process_time'] = 0


plt.rc('font', family='serif', size=14)

labels = [
        'B:1 Local', 
        'B:1 Remote',
        'B:5 Local',
        'B:5 Remote',
        'B:10 Local',
        'B:10 Remote'
       ]


for network in ['AlexNet', 'MobileNet']:
    for datacenter in 'ibm-sp', 'ibm-dc':
        fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))

        ax.set_ylabel("Time (ms)")
        ax.set_ylim(0, 1500)

        e1_local_time = gen_list(df_local, df_remote, network, datacenter, 8, 'e1_local_time')
        e2_local_time = gen_list(df_local, df_remote, network, datacenter, 8, 'e2_local_time')
        e2_remote_time = gen_list(df_local, df_remote, network, datacenter, 8, 'e2_remote_time')
        network_time = gen_list(df_local, df_remote, network, datacenter, 8, 'network_time')

        ax.bar(labels, e1_local_time, label="E1 Local Time")
        ax.bar(labels, e2_local_time, bottom=e1_local_time, label="E2 Local Time")
        ax.bar(labels, e2_remote_time, bottom=e1_local_time + e2_local_time, label="E2 Remote Time")
        ax.bar(labels, network_time, bottom=e1_local_time + e2_local_time + e2_remote_time, label="Network Time")

        ax.legend(loc="upper left")

        ax.tick_params(axis='x', rotation=45)
        ax.set_title(f'{network} {datacenter} - Elapsed Time')

        filename = os.path.join(script_dir, 'mq', f'elapsed_time_{network}_{datacenter}.png')

        if filename:
            fig.savefig(filename)
        else:
            plt.show()
        
        fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))

        ax.set_ylabel("Process Time (ms)")
        ax.set_ylim(0, 5500)

        e1_process_time = gen_list(df_local, df_remote, network, datacenter, 8, 'e1_process_time')
        e2_process_time = gen_list(df_local, df_remote, network, datacenter, 8, 'e2_process_time')
        request_process_time = gen_list(df_local, df_remote, network, datacenter, 8, 'request_process_time')

        ax.bar(labels, e1_process_time, label="E1 Process Time")
        ax.bar(labels, e2_process_time, bottom=e1_process_time, label="E2 Process Time")
        ax.bar(labels, request_process_time, bottom=e1_process_time + e2_process_time, label="Request Process Time")

        ax.legend(loc="upper left")

        ax.tick_params(axis='x', rotation=45)
        ax.set_title(f'{network} {datacenter} - Processed Time (Energy)')

        filename = os.path.join(script_dir, 'mq', f'processed_time_{network}_{datacenter}.png')

        if filename:
            fig.savefig(filename)
        else:
            plt.show()




