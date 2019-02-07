from experiment_0_util.run_experiment import *
import numpy as np
import pandas as pd
import os

experiment_1_data_path = os.path.join(final_data_path, 'experiment_1_data.csv')

acc_dict = []

for phi in np.arange(0.01, 0.11, 0.01):
    acc_dict += run_theta_phi_pair(1, phi)

for phi in np.arange(0.10, 1.01, 0.05):
    acc_dict += run_theta_phi_pair(1, phi)

for phi in np.arange(1.0, 5, 0.2):
    acc_dict += run_theta_phi_pair(1, phi)

acc_df = pd.DataFrame(acc_dict)
try:
    prev_acc_df = pd.read_csv(experiment_1_data_path)
    acc_df = pd.concat([acc_df, prev_acc_df])
except FileNotFoundError:
    pass
acc_df.to_csv(experiment_1_data_path)
