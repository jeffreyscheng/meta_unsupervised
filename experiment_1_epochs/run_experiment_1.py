from experiment_0_util.run_experiment import *
import pickle
import pandas as pd
import os
from os.path import join

# run the control
# save it to Fashion-MNIST/final_data

experiment_1_data_path = join(root_directory, dataset_name, 'final_data', 'experiment_1_data.pkl')


def test_mllr_update_pair(mllr, update_rate):
    hyperparameters['meta_learning_rate'] = mllr
    hyperparameters['hebbian_update_rate'] = update_rate
    return hebbian_frame.train_model(phi=50, theta=1, intermediate_accuracy=True, return_model=False)


results = {(0, 0): control_frame.train_model()}
for metalearner_learning_rate in [1, 0.1, 0.01, 0.001]:
    for update_rate in [0.0001, 0.00001, 0.000001, 0.0000001, 0.0000000001]:
        results[(metalearner_learning_rate, update_rate)] = test_mllr_update_pair(metalearner_learning_rate,
                                                                                  update_rate)

pickle.dump(results, open(experiment_1_data_path, 'wb'))
