from experiment_5_rule_over_time.experiment_metacaching import *
import numpy as np
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

fn = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                  'final_data',
                  'raw_phi_experiment.csv')

for phi in np.arange(0.01, 0.11, 0.01):
    run_theta_phi_pair_with_cache(1, phi)

for phi in np.arange(0.10, 1.01, 0.05):
    run_theta_phi_pair_with_cache(1, phi)

for phi in np.arange(1.0, 5, 0.2):
    run_theta_phi_pair_with_cache(1, phi)

for phi in np.arange(5, 20, 1):
    run_theta_phi_pair_with_cache(1, phi)
