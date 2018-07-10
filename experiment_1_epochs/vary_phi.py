from experiment_0_util.experiment import *
import numpy as np
import pandas as pd


acc_dict = []

for phi in np.arange(0.01, 0.11, 0.01):
    acc_dict += run_theta_phi_pair(1, phi)

for phi in np.arange(0.10, 1.01, 0.05):
    acc_dict += run_theta_phi_pair(1, phi)

for phi in np.arange(1.0, 5, 0.2):
    acc_dict += run_theta_phi_pair(1, phi)

for phi in np.arange(5, 20, 1):
    acc_dict += run_theta_phi_pair(1, phi)

acc_df = pd.DataFrame(acc_dict)
prev_acc_df = pd.read_csv('raw_phi_experiment.csv')
acc_df = pd.concat(acc_df, prev_acc_df)
acc_df.to_csv('raw_phi_experiment.csv')
