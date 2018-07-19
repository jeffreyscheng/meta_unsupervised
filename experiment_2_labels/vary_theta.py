from experiment_0_util.experiment import *
import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

fn = os.path.join(os.path.dirname(__file__), 'raw_theta_experiment.csv')

acc_dict = []

for theta in np.arange(0.001, 0.011, 0.001):
    acc_dict += run_theta_phi_pair(1, theta)

for theta in np.arange(0.01, 0.105, 0.005):
    acc_dict += run_theta_phi_pair(1, theta)

for theta in np.arange(0.10, 1.01, 0.01):
    acc_dict += run_theta_phi_pair(1, theta)

acc_df = pd.DataFrame(acc_dict)
try:
    prev_acc_df = pd.read_csv(fn)
    acc_df = pd.concat([acc_df, prev_acc_df])
except FileNotFoundError:
    pass
acc_df.to_csv(fn)
