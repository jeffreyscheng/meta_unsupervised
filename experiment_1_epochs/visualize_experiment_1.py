import numpy as np
from experiment_0_util.run_experiment import visualize_phi
from hyperparameters import *

experiment_1_data_path = os.path.join(root_directory,
                                      'final_data',
                                      'old',
                                      'experiment_1_data.csv')
results_path = os.path.join(root_directory, 'experiment_results')

visualize_phi(experiment_1_data_path, np.max, 'Maximum Accuracy', results_path, 'experiment_1_max.png')
visualize_phi(experiment_1_data_path, np.median, 'Median Accuracy', results_path, 'experiment_1_median.png')
visualize_phi(experiment_1_data_path, np.min, 'Minimum Accuracy', results_path, 'experiment_1_minimum.png')
