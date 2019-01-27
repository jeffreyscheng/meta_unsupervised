from experiment_5_rule_over_time.experiment_metacaching import *
import numpy as np
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tick = time.time()
for _ in range(50):
    for phi in np.arange(0.01, 0.11, 0.01):
        run_theta_phi_pair_with_cache(1, phi)

    for phi in np.arange(0.10, 1.01, 0.05):
        run_theta_phi_pair_with_cache(1, phi)

    for phi in np.arange(1.0, 5, 0.2):
        run_theta_phi_pair_with_cache(1, phi)

    for phi in np.arange(5, 20, 1):
        run_theta_phi_pair_with_cache(1, phi)

print(time.time() - tick)