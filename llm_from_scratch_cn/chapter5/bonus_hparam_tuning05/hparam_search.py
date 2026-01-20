import itertools
import math
import os
import tiktoken
import torch



# Define a grid of hyperparameters to search over
HPARAM_GRID = {
    "batch_size": [2, 4, 8, 16],
    "drop_rate": [0.0, 0.1, 0.2],
    "warmup_iters": [10, 20, 30],
    "weight_decay": [0.1, 0.01, 0.0],
    "peak_lr": [0.0001, 0.0005, 0.001, 0.005],
    "initial_lr": [0.00005, 0.0001],
    "min_lr": [0.00005, 0.00001, 0.0001],
    "n_epochs": [5, 10, 15, 20, 25],
} # 3*3*3*3*4*2*3*5 = 12960

if __name__ == '__main__':
    # 笛卡尔积
    hyperparameter_combinations = list(itertools.product(*HPARAM_GRID.values()))
    total_combinations = len(hyperparameter_combinations)
    print(f"Total hyperparameter configurations: {total_combinations}") # 一共12960种组合
