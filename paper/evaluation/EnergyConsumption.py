#!/usr/bin/env python3

# https://docs.google.com/spreadsheets/d/1mz9FeGGzptKERec5ZN3vaJsfVdjiKhgSIJhfVWJSnDo/edit?gid=0#gid=0

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import tabulate

data = {
    'MobileNet': {
        '1exit': {
            'mAh_minute': 8.83,
            'uAh_per_request_total': 88.33,
            'uAh_per_request_net': 35.976,
            'requests': 3000,
        },
        '2exit_local': {
            'mAh_minute': 8.77,
            'uAh_per_request_total': 102.33,
            'uAh_per_request_net': 41.250,
            'requests': 3000,  
        },
        '2exit_remote': {
            'mAh_minute': 8.19,
            'uAh_per_request_total': 43.67,
            'uAh_per_request_net': 15.743,
            'requests': 3000,
        }
    },
    'AlexNet': {
        '1exit': {
            'mAh_minute': 9.17,
            'uAh_per_request_total': 220,
            'uAh_per_request_net': 94.343,
            'requests': 1500,
        },
        '2exit_local': {
            'mAh_minute': 8.92,
            'uAh_per_request_total': 214,
            'uAh_per_request_net': 88.343,
            'requests': 1500,  
        },
        '2exit_remote': {
            'mAh_minute': 5.73,
            'uAh_per_request_total': 9.56,
            'uAh_per_request_net': 0.829,
            'requests': 9000,
        }
    }
}

if __name__ == '__main__':
    lines = []
    for model, values in data.items():
        for exit, value in values.items():
            mAh_minute = value['mAh_minute']
            uAh_per_request_total = value['uAh_per_request_total']
            uAh_per_request_net = value['uAh_per_request_net']
            requests = value['requests']
            lines.append([model, exit, mAh_minute, uAh_per_request_total, uAh_per_request_net, requests])
    print(tabulate.tabulate(lines, 
                            headers=[
                                     'Model', 
                                     'Exit', 
                                     'mAh_minute', 
                                     'uAh_per_request_total', 
                                     'uAh_per_request_net', 
                                     'requests'
                                    ], 
                            tablefmt='grid'))
            
            
            