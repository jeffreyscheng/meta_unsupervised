from experiment_0_util.run_experiment import *
import pandas as pd
import os

experiment_1_data_path = os.path.join(final_data_path, 'experiment_1_data.csv')

# def label_hebbian(d, bool_hebbian):
  #   d['bool_hebbian'] = bool_hebbian
  #   return d

# hebbian_list = [hebbian_frame.train_model(phi=phi_val, theta=theta_val, intermediate_accuracy=True) for _ in range(experiment_iterations)]
# hebbian_list = [label_hebbian(d, 1) for iteration in hebbian_list for d in iteration]  # flattens

# print("Running Control:", phi_val, theta_val)
# control_list = [control_frame.train_model(phi=phi_val, theta=theta_val, intermediate_accuracy=True) for _ in range(experiment_iterations)]
# control_list = [label_hebbian(d, 0) for iteration in control_list for d in iteration]  # flattens

# return hebbian_list + control_list



acc_dict = run_theta_phi_pair(phi_val=5, theta_val=1)
acc_df = pd.DataFrame(acc_dict)

try:
    prev_acc_df = pd.read_csv(experiment_1_data_path)
    acc_df = pd.concat([acc_df, prev_acc_df])
except FileNotFoundError:
    pass
acc_df.to_csv(experiment_1_data_path)
