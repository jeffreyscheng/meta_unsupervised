import numpy as np
from experiment_0_util.run_experiment import visualize_phi
from hyperparameters import *
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join

experiment_1_data_path = join(root_directory, dataset_name, 'final_data', 'experiment_1_data.pkl')

#  VISUALIZATION 1: just plot the learning curves to tune hyperparameters
results = pickle.load(open(experiment_1_data_path, 'rb'))
hyperparams = set([(key[0], key[1]) for key in results.keys()])
for h in hyperparams:
    if h != (0, 0):
        list_of_dfs = [pd.DataFrame(results[key]) for key in results if key[0] == h[0] and key[1] == h[1]]
        epoch = list_of_dfs[0]['batch_num'] * 10 / 60000
        mean_accuracy = pd.DataFrame([df['accuracy'] for df in list_of_dfs]).median(axis=0)
        if mean_accuracy.max() > 0.8:
            plt.plot(epoch, mean_accuracy, label=h)
    else:
        results_df = pd.DataFrame(results[h])
        epoch = results_df['batch_num'] * 10 / 60000
        plt.plot(epoch, results_df['accuracy'], label=h)

plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
# plt.ylim((0.7, 0.9))
plt.legend()
plt.savefig(os.path.join(root_directory, dataset_name, 'result_images', 'experiment_1_accuracy.png'))
plt.show()
plt.clf()


#  VISUALIZATION 2: see if the metalearner is actually converging before the learner
#  ANSWER: gradients are actually getting LARGER
best_result = results[(10 ** -4, 10 ** -10)]
result_df = pd.DataFrame(best_result)
result_df['epoch'] = result_df['batch_num'] * 10 / 60000
result_df['scaled_learner_gradient_norm'] = result_df['learner_gradient_norm'] / result_df['learner_gradient_norm'].mean()
result_df['scaled_metalearner_gradient_norm'] = result_df['metalearner_gradient_norm'] / result_df['metalearner_gradient_norm'].mean()
result_df['scaled_hebbian_update_norm'] = result_df['hebbian_update_norm'] / result_df['hebbian_update_norm'].mean()
plt.plot(result_df['epoch'], result_df['scaled_learner_gradient_norm'], label='Scaled Learner Gradient')
plt.plot(result_df['epoch'], result_df['scaled_metalearner_gradient_norm'], label='Scaled Metalearner Gradient')
plt.plot(result_df['epoch'], result_df['scaled_hebbian_update_norm'], label='Scaled Hebbian Update')
plt.xlabel('Number of Epochs')
plt.ylabel('Update Norm')
plt.legend()
plt.grid()
plt.savefig(os.path.join(root_directory, dataset_name, 'result_images', 'experiment_1_gradient_comparison.png'))
plt.show()
plt.clf()

plt.plot(result_df['epoch'], result_df['hebbian_update_norm'])
plt.xlabel('Number of Epochs')
plt.ylabel('Hebbian Update Norm')
plt.legend()
plt.grid()
plt.savefig(os.path.join(root_directory, dataset_name, 'result_images', 'experiment_1_hebbian_updates.png'))
plt.show()
plt.clf()