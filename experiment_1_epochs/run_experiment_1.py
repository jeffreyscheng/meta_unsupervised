from experiment_0_util.run_experiment import *
import numpy as np
import pandas as pd
import os

experiment_1_data_path = os.path.join(final_data_path, 'experiment_1_data.csv')

acc_dict = []

# for phi in np.arange(0.01, 0.10, 0.01):
#     acc_dict += run_theta_phi_pair(1, phi)
#
# for phi in np.arange(0.10, 0.5, 0.05):
#     acc_dict += run_theta_phi_pair(1, phi)

for phi in np.arange(0.50, 1.0, 0.02):  # this is the critical point
    acc_dict += run_theta_phi_pair(1, phi)

# for phi in np.arange(1.0, 2, 0.1):
#     acc_dict += run_theta_phi_pair(1, phi)
#
# for phi in np.arange(2.0, 3, 0.2):
#     acc_dict += run_theta_phi_pair(1, phi)

acc_df = pd.DataFrame(acc_dict)
try:
    prev_acc_df = pd.read_csv(experiment_1_data_path)
    acc_df = pd.concat([acc_df, prev_acc_df])
except FileNotFoundError:
    pass
acc_df.to_csv(experiment_1_data_path)
