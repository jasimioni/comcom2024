#!/usr/bin/env python3

from helper import *
import argparse
import os

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--datafile', type=str)
    argparse.add_argument('--savefolder', type=str)
    argparse.add_argument('--title', type=str)
    
    args = argparse.parse_args()
    
    df = pd.read_csv(args.datafile)
    
    y = df['y']
    y_exit_1 = df['y_exit_1']
    y_exit_2 = df['y_exit_2']
    cnf_exit_1 = df['cnf_exit_1']
    cnf_exit_2 = df['cnf_exit_2']
    
    ece_1 = calculateECE(y, y_exit_1, cnf_exit_1)
    ece_2 = calculateECE(y, y_exit_2, cnf_exit_2)
    
    print(f"ECE Exit 1: {ece_1:.5f}")
    print(f"ECE Exit 2: {ece_2:.5f}")
    
    basename = os.path.basename(args.datafile)[0:-4]
    
    genHistogram(y, y_exit_1, cnf_exit_1, title=f"{args.title}|Exit 1", filename=os.path.join(args.savefolder, f"{basename}_exit_1.png"))
    genHistogram(y, y_exit_2, cnf_exit_2, title=f"{args.title}|Exit 2", filename=os.path.join(args.savefolder, f"{basename}_exit_2.png"))