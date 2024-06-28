#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

plt.rc('font', family='serif', size=14)

def getBins(y, y_pred, cnf, n_bins=10, lower=0.5, upper=1):
    ece = 0
    step = ( upper - lower ) / n_bins
    bins = []
    for bin in range(n_bins):
        bin_min = lower + bin * step
        bin_max = lower + ( bin + 1 ) * step

        mask = (cnf > bin_min) & (cnf <= bin_max)
        bin_acc = (y[mask] == y_pred[mask]).sum() / mask.sum()
        bin_cnf = cnf[mask].mean()
        
        bins.append({
            'min': bin_min,
            'max': bin_max,
            'acc': bin_acc,
            'cnf': bin_cnf,
            'prop': mask.sum() / len(y)
        })

        ece += np.abs(bin_acc - bin_cnf) * mask.sum()

    ece /= len(y)

    data = { 'ece': ece, 'bins': bins }
    # print(data)
    return data

def genHistogram(y, y_pred, cnf, n_bins=10, lower=0.5, upper=1, title="", filename=""):
    data = getBins(y, y_pred, cnf, n_bins, lower, upper)
    
    bin_labels = [ f"{bin['min']:.2f}-{bin['max']:.2f} ({100 * bin['prop']:.2f}%)" for bin in data['bins'] ]
    # bin_labels = [ f"{bin['prop']:.2f}" for bin in data['bins'] ]
    bin_acc = [ bin['acc'] for bin in data['bins'] ]
    bin_missing = [ max(0, bin['cnf'] - bin['acc']) for bin in data['bins'] ]
    bin_overflow = [ min(0, bin['cnf'] - bin['acc']) for bin in data['bins'] ]
    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 7))
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 1)
    
    print(f"Accuracy: {list(bin_acc)}")
    print(f"Missing bin: {bin_missing}")

    ax.bar(bin_labels, bin_acc, label="Accuracy")
    ax.bar(bin_labels, bin_missing, label="Missing", bottom=bin_acc, color='red')
    ax.bar(bin_labels, bin_overflow, label="Overflow", bottom=bin_acc, color='green')
    
    ax.legend(loc="upper left")

    lr = Ridge()

    np_bins = np.array(range(n_bins)).reshape(-1, 1)

    print(np_bins)
    print(bin_acc)

    lr.fit(np_bins, bin_acc)
    plt.plot(bin_labels, lr.coef_*np_bins+lr.intercept_, color='orange')

    ax.tick_params(axis='x', rotation=90)
    ax.set_title(f'{title} | ECE: {data["ece"]:.5f}')
    
    # Add ECE to the footer of the chart
    # ax.text(0.15, .94, f"ECE: {data['ece']:.5f}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    if filename:
        fig.savefig(filename)
    else:
        plt.show()

def calculateECE(y, y_pred, cnf, n_bins=10, min=0.5, max=1):
    return getBins(y, y_pred, cnf, n_bins, min, max)['ece']