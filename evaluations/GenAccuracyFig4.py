#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

alexnet_folder = 'icc2024/evaluations/AlexNet/cuda/saves/AlexNet/2023-10-31-01-48-09/epoch_16_91.2.pth/'
mobilenet_folder = 'icc2024/evaluations/MobileNet/cuda/saves/MobileNetV2/2023-10-26-04-42-32/epoch_17_90.9.pth/'

MONTHS_NAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def consolidate(directory):
    year = '2016'
    results = pd.DataFrame(columns=['month', 'total', 'correct', 'fp', 'fn', 'tp', 'tn', 'err', 'fpr', 'fnr'])
    for month in range(1, 13):
        glob = f'{year}_{month:02d}'
        files = Path(directory).glob(f'*{glob}*')
        dfs = []
        for file in sorted(files):
            dfs.append(pd.read_csv(file))        
        
        df = pd.concat(dfs, ignore_index=True)
        
        total = df['y'].count()
        correct = df.query('y == y_pred')['y'].count()
        fp = df.query('y == 0 and y_pred == 1')['y'].count()
        fn = df.query('y == 1 and y_pred == 0')['y'].count()
        tn = df.query('y == 0')['y'].count()
        tp = df.query('y == 1')['y'].count()

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

an = consolidate(alexnet_folder)
mn = consolidate(mobilenet_folder)

print(an)
print(mn)

plt.rc('font', family='serif', size=26)

y_max = max([ an['err'].max(), an['fpr'].max(), an['fnr'].max(),
              mn['err'].max(), mn['fpr'].max(), mn['fnr'].max() ])

y_max = int(y_max / 10) * 10 + 10

fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))
ax.set(ylim=(-2.5, y_max), ylabel='Error Rate (%)')
ax.tick_params(axis='x', rotation=45)

# ax.plot(MONTHS_NAME, an['err'], label='ERROR RATE', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='blue')
ax.plot(MONTHS_NAME, an['fpr'], label='FPR', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='black')
ax.plot(MONTHS_NAME, an['fnr'], label='FNR', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='red')

ax.legend(loc='upper left', frameon=False)

fig.savefig(f'icc2024/evaluations/alexnet_error_rate.pdf')

fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))
ax.set(ylim=(-2.5, y_max), ylabel='Error Rate (%)')
ax.tick_params(axis='x', rotation=45)

# ax.plot(MONTHS_NAME, mn['err'], label='ERROR RATE', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='blue')
ax.plot(MONTHS_NAME, mn['fpr'], label='FPR', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='black')
ax.plot(MONTHS_NAME, mn['fnr'], label='FNR', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='red')

ax.legend(loc='upper left', frameon=False)

fig.savefig(f'icc2024/evaluations/mobilenet_error_rate.pdf')