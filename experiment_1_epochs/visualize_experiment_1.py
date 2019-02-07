import numpy as np
from experiment_0_util.run_experiment import visualize_phi
from hyperparameters import *

experiment_1_data_path = os.path.join(final_data_path, 'experiment_2_data.csv')

visualize_phi(experiment_1_data_path, np.max, 'Maximum Accuracy', result_images_path, 'experiment_1_max.png')
visualize_phi(experiment_1_data_path, np.median, 'Median Accuracy', result_images_path, 'experiment_1_median.png')
visualize_phi(experiment_1_data_path, np.min, 'Minimum Accuracy', result_images_path, 'experiment_1_minimum.png')
