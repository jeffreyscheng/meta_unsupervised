from experiment_4_prelearned.optimal_frame import *
from hyperparameters import *

n = 90  # iterations per experimental pair


def run_theta_phi_pair(theta_val, phi_val):
    print("Running Optimal:", theta_val, phi_val)
    optimal_acc = [optimal_frame.train_model(hyperparameters['mid1'],
                                             hyperparameters['mid2'],
                                             hyperparameters['learning_rate'],
                                             hyperparameters['learner_batch_size'],
                                             hyperparameters['update_rate'],
                                             theta=theta_val, phi=phi_val) for _ in range(n)]
    formatted_optimal = [{'theta': theta_val, 'phi': phi_val, 'bool_optimal': 1, 'acc': a} for a in optimal_acc]

    return formatted_optimal
