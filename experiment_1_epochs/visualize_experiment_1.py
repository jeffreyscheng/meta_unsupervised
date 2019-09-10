import numpy as np
from experiment_0_util.run_experiment import visualize_phi
from hyperparameters import *
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join

experiment_1_data_path = join(root_directory, dataset_name, 'final_data', 'experiment_1_data.pkl')

# first, just plot the learning curves
results = pickle.load(open(experiment_1_data_path, 'rb'))
for key in results:
    result_df = pd.DataFrame(results[key])
    result_df['epoch'] = result_df['batch_num'] * 10 / 60000
    result_df[str(key)] = result_df['accuracy']

    plt.plot(result_df['batch_num'], result_df[str(key)], label=key)

plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
# plt.ylim((0.7, 0.9))
plt.legend()
plt.savefig(os.path.join(root_directory, dataset_name, 'result_images', 'experiment_1_accuracy.png'))
plt.show()