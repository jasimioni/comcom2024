#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

os.environ['PYTHONUNBUFFERED'] = '1'

directories = {
    # 'alexnet'   : '../evaluations/AlexNet/cuda/saves/AlexNetWithExits/2023-10-31-01-01-09/epoch_19_90.1_91.1.pth',
    # 'mobilenet' : '../evaluations/MobileNet/cuda/saves/MobileNetV2WithExits/2023-08-20-05-20-25/epoch_19_89.7_90.9.pth',
    'alexnet'   : '../calibration/calibrated/AlexNetWithExits/01',
    'mobilenet' : '../calibration/calibrated/MobileNetWithExits/01',
}

'''
Fixar: Taxa de aceite

2 objetivos:
- Acurácia total do sistema
- Tempo médio de inferência

4 parâmetros -> 
Limiar de normal / ataque na primeira
Limiar de normal / ataque na segunda
'''

def get_objectives(df, normal_threshold_exit1, attack_threshold_exit1, normal_threshold_exit2, attack_threshold_exit2):
    f_n_exit1 = 'y_exit_1 == 0 and cnf_exit_1 >= @normal_threshold_exit1'
    f_a_exit1 = 'y_exit_1 == 1 and cnf_exit_1 >= @attack_threshold_exit1'

    f_n_exit2 = 'y_exit_2 == 0 and cnf_exit_2 >= @normal_threshold_exit2'
    f_a_exit2 = 'y_exit_2 == 1 and cnf_exit_2 >= @attack_threshold_exit2'

    exit1_normal = df.query(f_n_exit1)
    exit1_attack = df.query(f_a_exit1)

    exit2 = df.query(f'not ({f_n_exit1}) and not ({f_a_exit1})')

    exit2_normal = exit2.query(f_n_exit2)
    exit2_attack = exit2.query(f_a_exit2)

    not_accepted = exit2.query(f'not ({f_n_exit2}) and not ({f_a_exit2})')

    total = df['y'].count()

    exit1_normal_cnt = exit1_normal['y'].count()
    exit1_attack_cnt = exit1_attack['y'].count()
    exit2_normal_cnt = exit2_normal['y'].count()
    exit2_attack_cnt = exit2_attack['y'].count()

    accepted = exit1_normal_cnt + exit1_attack_cnt + exit2_normal_cnt + exit2_attack_cnt

    acceptance_rate = accepted / total

    correct = exit1_normal.query('y == y_exit_1')['y'].count() + \
              exit1_attack.query('y == y_exit_1')['y'].count() + \
              exit2_normal.query('y == y_exit_2')['y'].count() + \
              exit2_attack.query('y == y_exit_2')['y'].count()

    accuracy = correct / accepted

    exit1_total_time = exit1_normal['bb_time_exit_1'].sum() + exit1_normal['exit_time_exit_1'].sum() + \
                       exit1_attack['bb_time_exit_1'].sum() + exit1_attack['exit_time_exit_1'].sum()

    exit2_total_time = exit2_normal['bb_time_exit_1'].sum() + exit2_normal['bb_time_exit_2'].sum() + exit2_normal['exit_time_exit_2'].sum() + \
                       exit2_attack['bb_time_exit_1'].sum() + exit2_attack['bb_time_exit_2'].sum() + exit2_attack['exit_time_exit_2'].sum()

    not_accepted_total_time = not_accepted['bb_time_exit_1'].sum() + not_accepted['bb_time_exit_2'].sum() + not_accepted['exit_time_exit_2'].sum()

    total_time = exit1_total_time + exit2_total_time + not_accepted_total_time

    # print(f"Total: {total}")
    # print(f"exit1_normal_cnt: {exit1_normal_cnt}, exit1_attack_cnt: {exit1_attack_cnt}")
    # print(f"exit2_normal_cnt: {exit2_normal_cnt}, exit2_attack_cnt: {exit2_attack_cnt}")
    # print(f"Accepted: {accepted}, Accepted: {total - not_accepted['y'].count()}")
    # print(f"exit1_total_time: {exit1_total_time:.4f}, exit2_total_time: {exit2_total_time:.4f}, not_accepted_total_time: {not_accepted_total_time:.4f}")
    # print(f"exit1_rate: {100 * ( exit1_normal_cnt + exit1_attack_cnt ) / total:.2f}, exit2_rate: {100 * ( exit2_normal_cnt + exit2_attack_cnt ) / total:.2f}")
    # print(f"Accuracy: {100 * accuracy:.2f}, Acceptance: {100 * acceptance_rate:.2f}, Average Time: {1e6 * total_time / total:.2f}")

    return [ accuracy, acceptance_rate, 1e6 * total_time / total ]

class MyProblem(ElementwiseProblem):
    def __init__(self, df, min_acceptance=0.7):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=np.array([ 0.5, 0.5, 0.5, 0.5 ]),
                         xu=np.array([ 1, 1, 1, 1 ]))
            
        self.df = df
        self.accuracy_e1, self.acceptance_e1, self.min_time = get_objectives(self.df, 0, 0, 1, 1)
        self.accuracy_e2, self.acceptance_e2, self.max_time = get_objectives(self.df, 2, 2, 0, 0)
        self.min_acceptance = min_acceptance      

    def _evaluate(self, x, out, *args, **kwargs):
        accuracy, acceptance, time = get_objectives(self.df, *x)
        out["F"] = [ 1 - accuracy, (time - self.min_time) / (self.max_time - self.min_time) ]
        out["G"] = [ self.min_acceptance - acceptance ]

def process(directory='alexnet', glob='2016_01', min_acceptance=0.7):
    directory = directories[directory]
    files = Path(directory).glob(f'*{glob}*')
    dfs = []
    for file in sorted(files):
        print(file)
        dfs.append(pd.read_csv(file))

    df = pd.concat(dfs, ignore_index=True)

    problem = MyProblem(df, min_acceptance=min_acceptance)

    algorithm = NSGA2(
        pop_size=100, # 100
        n_offsprings=80, # 80
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 1000) # 1000

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    X = res.X
    F = res.F

    print(f'{directory}')
    print(f'{glob}')

    print(f'Exit1: {problem.accuracy_e1*100:.2f}% - {problem.acceptance_e1*100:.2f}% - {problem.min_time:.2f}us')
    print(f'Exit2: {problem.accuracy_e2*100:.2f}% - {problem.acceptance_e2*100:.2f}% - {problem.max_time:.2f}us')
    print()

    for i in range(len(F)):
        f = F[i]
        x = X[i]
        print(f'{i:02d}: {100 * (1 - f[0]):.2f}% : {problem.min_time + (f[1] * (problem.max_time - problem.min_time)):.2f}us', end='')
        print(f'\t{x[0]:.4f} : {x[1]:.4f} : {x[2]:.4f} : {x[3]:.4f}')

    return X, F, problem.min_time, problem.max_time, problem.accuracy_e1, problem.acceptance_e1, problem.accuracy_e2, problem.acceptance_e2

if __name__ == '__main__':
    network = sys.argv[1]
    try:
        min_acceptance = float(sys.argv[2])
    except Exception as e:
        min_acceptance = 0.7

    print(f"Processing {network} with min_acceptance = {min_acceptance}")

    X, F, min_time, max_time, accuracy_e1, acceptance_e1, accuracy_e2, acceptance_e2 = process(network, '2016_0[23].csv', min_acceptance)

    print(f"{min_time}, {max_time}, {accuracy_e1}, {acceptance_e1}, {accuracy_e2}, {acceptance_e2}")

    with open(f'{network}_x_f_{min_acceptance}_2016_23.sav', 'wb') as f:
        pickle.dump([ X, F, min_time, max_time, accuracy_e1, acceptance_e1, accuracy_e2, acceptance_e2 ], f)
