from experiment_0_util.control_frame import *
from experiment_0_util.hebbian_frame import *
from hyperparameters import *
import matplotlib.pyplot as plt
import pandas as pd


def run_theta_phi_pair(theta_val, phi_val):
    print("Running Hebbian:", theta_val, phi_val)
    hebbian_acc = [hebbian_frame.train_model(theta=theta_val, phi=phi_val) for _ in range(experiment_iterations)]
    formatted_hebbian = [{'theta': theta_val, 'phi': phi_val, 'bool_hebbian': 1, 'acc': a} for a in hebbian_acc]

    print("Running Control:", theta_val, phi_val)
    control_acc = [control_frame.train_model(theta=theta_val, phi=phi_val) for _ in range(experiment_iterations)]
    formatted_control = [{'theta': theta_val, 'phi': phi_val, 'bool_hebbian': 0, 'acc': a} for a in control_acc]

    return formatted_control + formatted_hebbian


def visualize_phi(input_path, fn, y_name, image_path, image_name):

    phi_df = pd.read_csv(input_path)
    agg = []
    for phi in set(phi_df['phi']):
        # if phi < 5.0:
        hebbian_sub = phi_df.loc[(phi_df['phi'] == phi) & (phi_df['bool_hebbian']),]
        control_sub = phi_df.loc[(phi_df['phi'] == phi) & (~phi_df['bool_hebbian']),]
        agg.append({'phi': phi, 'hebbian_acc': fn(hebbian_sub['acc']), 'control_acc': fn(control_sub['acc'])})

    print(agg)
    agg_df = pd.DataFrame(agg)
    agg_df = agg_df.sort_values(by=['phi'], ascending=True)

    plt.plot(agg_df['phi'], agg_df['hebbian_acc'])
    plt.plot(agg_df['phi'], agg_df['control_acc'])
    # plt.legend(['y = x', 'y = 2x', 'y = 3x', 'y = 4x'], loc='upper left')

    plt.xlabel('Number of Epochs')
    plt.ylabel(y_name)

    plt.savefig(os.path.join(image_path, image_name))
    plt.clf()


def visualize_theta(input_path, fn, y_name, image_path, image_name):

    theta_df = pd.read_csv(input_path)
    agg = []
    for theta in set(theta_df['phi']):
        # if phi < 5.0:
        hebbian_sub = theta_df.loc[(theta_df['phi'] == phi) & (theta_df['bool_hebbian']),]
        control_sub = theta_df.loc[(theta_df['phi'] == phi) & (~theta_df['bool_hebbian']),]
        agg.append({'theta': theta, 'hebbian_acc': fn(hebbian_sub['acc']), 'control_acc': fn(control_sub['acc'])})

    print(agg)
    agg_df = pd.DataFrame(agg)
    agg_df = agg_df.sort_values(by=['theta'], ascending=True)

    plt.plot(agg_df['theta'], agg_df['hebbian_acc'])
    plt.plot(agg_df['theta'], agg_df['control_acc'])
    # plt.legend(['y = x', 'y = 2x', 'y = 3x', 'y = 4x'], loc='upper left')

    plt.xlabel('Proportion of Labeled Examples')
    plt.ylabel(y_name)

    plt.savefig(os.path.join(image_path, image_name))
    plt.clf()
