#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse

"""
./GenChartErrorRate.py --folder ../../calibration/non_calibrated/AlexNet --output ChartErrorRate/AlexNetErrorRate.png
./GenChartErrorRate.py --folder ../../calibration/non_calibrated/MobileNet --output ChartErrorRate/MobileNetErrorRate.png
./GenChartErrorRate.py --folder ../../calibration/calibrated/AlexNet --output ChartErrorRate/AlexNetCalibratedErrorRate.png
./GenChartErrorRate.py --folder ../../calibration/calibrated/MobileNet --output ChartErrorRate/MobileNetCalibratedErrorRate.png
"""

MONTHS_NAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def consolidate(directory):
    year = '2016'
    results = pd.DataFrame(columns=['month', 'total', 'correct', 'fp', 'fn', 'tp', 'tn', 'err', 'fpr', 'fnr'])
    for month in range(1, 13):
        glob = f'{year}_{month:02d}.csv'
        files = Path(directory).glob(f'*{glob}*')
        dfs = []
        for file in sorted(files):
            print(f"Reading {file}")
            dfs.append(pd.read_csv(file))        
        
        df = pd.concat(dfs, ignore_index=True)
        
        total = df['y'].count()
        correct = df.query('y == y_pred')['y'].count()
        fp = df.query('y == 0 and y_pred == 1')['y'].count()
        fn = df.query('y == 1 and y_pred == 0')['y'].count()
        tn = df.query('y == 0 and y_pred == 0')['y'].count()
        tp = df.query('y == 1 and y_pred == 1')['y'].count()

        err = 100 * ( total - correct ) / total
        fpr = 100 * fp / (fp + tn)
        fnr = 100 * fn / (fn + tp)

        results.loc[len(results)] = {
            'month' : f'{month:02d}',
            'total': total,
            'correct': correct,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'tn': tn,
            'err': err,
            'fpr': fpr,
            'fnr': fnr
        }

        # print(f"{glob}: Total: {total}, Correct: {correct}, Error Rate: {err:.2f}, FPR: {fpr:.2f}, FNR: {fnr:.2f}")

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', help='Data folder with the evaluations from 2016_01 to 2016_12', required=True)
    parser.add_argument('--output', help='Output file', required=True)
    
    args = parser.parse_args()
    data = consolidate(args.folder)
    
    plt.rc('font', family='serif', size=26)

    y_max = max([ data['err'].max(), data['fpr'].max(), data['fnr'].max() ])
    y_max = int(y_max / 10) * 10 + 10

    fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))
    ax.set(ylim=(-2.5, y_max), ylabel='Error Rate (%)')
    ax.tick_params(axis='x', rotation=45)

    ax.plot(MONTHS_NAME, data['err'], label='ERROR RATE', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='blue')
    ax.plot(MONTHS_NAME, data['fpr'], label='FPR', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='black')
    ax.plot(MONTHS_NAME, data['fnr'], label='FNR', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='red')

    ax.legend(loc='upper left', frameon=False)

    fig.savefig(args.output)