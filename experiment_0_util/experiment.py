from experiment_0_util.control_frame import *
from experiment_0_util.hebbian_frame import *
from hyperparameters import *

n = 1  # iterations per experimental pair


def run_theta_phi_pair(theta_val, phi_val):
    print("Running Hebbian:", theta_val, phi_val)
    hebbian_acc = [hebbian_frame.train_model(hyperparameters['mid1'],
                                             hyperparameters['mid2'],
                                             hyperparameters['meta_mid'],
                                             hyperparameters['learning_rate'],
                                             hyperparameters['learner_batch_size'],
                                             hyperparameters['update_rate'],
                                             theta=theta_val, phi=phi_val) for _ in range(n)]
    formatted_hebbian = [{'theta': theta_val, 'phi': phi_val, 'bool_hebbian': 1, 'acc': a} for a in hebbian_acc]

    print("Running Control:", theta_val, phi_val)
    control_acc = [control_frame.train_model(hyperparameters['mid1'],
                                             hyperparameters['mid2'],
                                             hyperparameters['learning_rate'],
                                             hyperparameters['learner_batch_size'],
                                             theta=theta_val, phi=phi_val) for _ in range(n)]
    formatted_control = [{'theta': theta_val, 'phi': phi_val, 'bool_hebbian': 0, 'acc': a} for a in control_acc]

    return formatted_control + formatted_hebbian
