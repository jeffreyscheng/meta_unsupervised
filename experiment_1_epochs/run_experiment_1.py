from experiment_0_util.run_experiment import *
import pandas as pd
import os

experiment_1_data_path = os.path.join(final_data_path, 'experiment_1_data.csv')

acc_dict = run_theta_phi_pair(phi_val=5, theta_val=1)
acc_df = pd.DataFrame(acc_dict)

try:
    prev_acc_df = pd.read_csv(experiment_1_data_path)
    acc_df = pd.concat([acc_df, prev_acc_df])
except FileNotFoundError:
    pass
acc_df.to_csv(experiment_1_data_path)
