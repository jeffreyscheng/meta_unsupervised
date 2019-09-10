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
for key in results:
    result_df = pd.DataFrame(results[key])
    result_df['epoch'] = result_df['batch_num'] * 10 / 60000

    plt.plot(result_df['epoch'], result_df['accuracy'], label=key)

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
plt.plot(result_df['epoch'], result_df['scaled_learner_gradient_norm'], label='Scaled Learner Gradient')
plt.plot(result_df['epoch'], result_df['scaled_metalearner_gradient_norm'], label='Scaled Metalearner Gradient')
plt.xlabel('Number of Epochs')
plt.ylabel('Gradient')
plt.legend()

plt.savefig(os.path.join(root_directory, dataset_name, 'result_images', 'experiment_1_gradient_comparison.png'))
plt.show()
plt.clf()


#  VISUALIZATION 3: see if the metalearner outputs are related to the gradients
best_result = results[(10 ** -4, 10 ** -10)]
result_df = pd.DataFrame(best_result)
result_df['epoch'] = result_df['batch_num'] * 10 / 60000
plt.scatter(result_df['learner_gradient_norm'], result_df['hebbian_update_norm'], c=result_df['epoch'])
plt.ylim((0, 0.00000002))
plt.xlabel('Gradient Norm')
plt.ylabel('Hebbian Update Norm')
plt.legend()
plt.savefig(os.path.join(root_directory, dataset_name, 'result_images', 'experiment_1_gradient_vs_Hebb.png'))
plt.show()
plt.clf()