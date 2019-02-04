import numpy as np
from experiment_0_util.experiment import visualize_theta
from hyperparameters import *

experiment_2_data_path = os.path.join(root_directory,
                                      'final_data',
                                      'old',
                                      'experiment_2_data.csv')
results_path = os.path.join(root_directory, 'experiment_results')

visualize_theta(experiment_2_data_path, np.max, 'Maximum Accuracy', results_path, 'experiment_2_max.png')
visualize_theta(experiment_2_data_path, np.median, 'Median Accuracy', results_path, 'experiment_2_median.png')
visualize_theta(experiment_2_data_path, np.min, 'Minimum Accuracy', results_path, 'experiment_2_minimum.png')
