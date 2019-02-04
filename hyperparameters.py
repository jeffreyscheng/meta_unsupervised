import os
import torch

root_directory = os.path.dirname(__file__)
fixed_parameters = {'meta_input': 3,
                    'meta_output': 1,
                    'input_size': 784,
                    'num_classes': 10}
hyperparameters = {'mid1': 100,
                   'mid2': 100,
                   'meta_mid': 50,
                   'learning_rate': 0.001,
                   'learner_batch_size': 50,
                   'update_rate': 0.001}
dataset_name = 'MNIST'
gpu_bool = torch.cuda.device_count() > 0
experiment_iterations = 100


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


safe_mkdir(os.path.join(root_directory, dataset_name))
safe_mkdir(os.path.join(root_directory, dataset_name, 'temp_data'))
safe_mkdir(os.path.join(root_directory, dataset_name, 'final_data'))
safe_mkdir(os.path.join(root_directory, dataset_name, 'result_images'))
