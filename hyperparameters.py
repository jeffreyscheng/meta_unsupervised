import os
import torch

print(__file__)
root_directory = os.path.dirname(__file__)
fixed_parameters = {'meta_input': 3,
                    'meta_output': 1,
                    'input_size': 784,
                    'num_classes': 10}
hyperparameters = {'mid1': 100,
                   'mid2': 100,
                   'meta_mid': 50,
                   'learning_rate': 0.005,
                   'learner_batch_size': 50,
                   'update_rate': 0.001}
dataset_name = 'MNIST'
gpu_bool = torch.cuda.device_count() > 0
experiment_iterations = 1
base_optimizer = torch.optim.SGD
time_out = 20 * 60
num_data = 60000


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


temp_data_path = os.path.join(root_directory, dataset_name, 'temp_data')
final_data_path = os.path.join(root_directory, dataset_name, 'final_data')
result_images_path = os.path.join(root_directory, dataset_name, 'result_images')

safe_mkdir(os.path.join(root_directory, dataset_name))
safe_mkdir(temp_data_path)
safe_mkdir(final_data_path)
safe_mkdir(result_images_path)
