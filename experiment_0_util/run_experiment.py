from experiment_0_util.control_frame import *
from experiment_0_util.hebbian_frame import *
from hyperparameters import *
import matplotlib.pyplot as plt
import pandas as pd


def label_hebbian(d):
    d['bool_hebbian'] = 1
    return d


def label_control(d):
    d['bool_hebbian'] = 0
    return d


def run_theta_phi_pair(phi_val, theta_val):
    print("Running Hebbian:", phi_val, theta_val)
    hebbian_list = [label_hebbian(hebbian_frame.train_model(phi=phi_val, theta=theta_val, intermediate_accuracy=True)) for _ in range(experiment_iterations)]
    hebbian_list = [d for iteration in hebbian_list for d in iteration] # flattens

    print("Running Control:", phi_val, theta_val)
    control_list = [label_control(control_frame.train_model(phi=phi_val, theta=theta_val)) for _ in
                    range(experiment_iterations)]
    control_list = [d for iteration in control_list for d in iteration]

    return hebbian_list + control_list


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
    plt.legend()

    plt.savefig(os.path.join(image_path, image_name))
    plt.clf()


def visualize_theta(input_path, fn, y_name, image_path, image_name):
    theta_df = pd.read_csv(input_path)
    agg = []
    for theta in set(theta_df['phi']):
        # if phi < 5.0:
        hebbian_sub = theta_df.loc[(theta_df['phi'] == theta) & (theta_df['bool_hebbian']),]
        control_sub = theta_df.loc[(theta_df['phi'] == theta) & (~theta_df['bool_hebbian']),]
        agg.append({'theta': theta, 'hebbian_acc': fn(hebbian_sub['acc']), 'control_acc': fn(control_sub['acc'])})

    print(agg)
    agg_df = pd.DataFrame(agg)
    agg_df = agg_df.sort_values(by=['theta'], ascending=True)

    plt.plot(agg_df['theta'], agg_df['hebbian_acc'])
    plt.plot(agg_df['theta'], agg_df['control_acc'])
    # plt.legend(['y = x', 'y = 2x', 'y = 3x', 'y = 4x'], loc='upper left')

    plt.xlabel('Proportion of Labeled Examples')
    plt.ylabel(y_name)
    plt.legend()

    plt.savefig(os.path.join(image_path, image_name))
    plt.clf()
