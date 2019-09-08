from experiment_0_util.run_experiment import *
import pandas as pd
import os


# run the control
# save it to Fashion-MNIST/final_data

# for update_ratio in [100, 1000, 10000, 100000]
# for metalearner's learning rate in [1, 0.1, 0.01, 0.001, 0.0001]
# run, save to /final_data
# see them all together in Tensorboard
# plot update_ratio, mllr, max accuracy on curve as a heat map.
# if the best one works, we good!!!
# if there's nothing, give up.

# fix phi = whatever, theta = 0.6
# make a function for (update_ratio, mllr) pair.  And then FIX that pair





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
