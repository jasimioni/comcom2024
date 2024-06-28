#!/usr/bin/env python3

from helper import *
import argparse

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("file")
    args = argparse.parse_args()
    
    df = pd.read_csv(args.file)
    
    y = df['y']
    y_exit_1 = df['y_exit_1']
    y_exit_2 = df['y_exit_2']
    cnf_exit_1 = df['cnf_exit_1']
    cnf_exit_2 = df['cnf_exit_2']
    
    ece_1 = calculateECE(y, y_exit_1, cnf_exit_1)
    ece_2 = calculateECE(y, y_exit_2, cnf_exit_2)
    
    print(f"ECE Exit 1: {ece_1:.5f}")
    print(f"ECE Exit 2: {ece_2:.5f}")
    
    