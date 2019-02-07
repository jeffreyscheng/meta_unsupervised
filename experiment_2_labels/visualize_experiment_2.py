import numpy as np
from experiment_0_util.run_experiment import visualize_theta
from hyperparameters import *

experiment_2_data_path = os.path.join(final_data_path, 'experiment_2_data.csv')

visualize_theta(experiment_2_data_path, np.max, 'Maximum Accuracy', result_images_path, 'experiment_2_max.png')
visualize_theta(experiment_2_data_path, np.median, 'Median Accuracy', result_images_path, 'experiment_2_median.png')
visualize_theta(experiment_2_data_path, np.min, 'Minimum Accuracy', result_images_path, 'experiment_2_minimum.png')
