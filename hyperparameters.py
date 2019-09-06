import os
import torch
import torch.nn as nn

root_directory = os.path.dirname(__file__)
dataset_name = 'Fashion-MNIST'
gpu_bool = torch.cuda.device_count() > 0


def push_to_gpu(x):
    if gpu_bool:
        return x.cuda()
    else:
        return x


experiment_iterations = 1
base_optimizer = torch.optim.Adam
learner_criterion = nn.CrossEntropyLoss()
time_out = 20 * 60
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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

# if dataset_name == 'MNIST':
#     fixed_parameters = {'meta_input': 3,
#                         'meta_output': 1,
#                         'input_size': 784,
#                         'num_classes': 10}
#     hyperparameters = {'learner_hidden_widths': (256, 128, 100),
#                        'meta_hidden_width': 50,
#                        'learning_rate': 0.001,
#                        'learner_batch_size': 10,
#                        'update_rate': 0.001}
#     num_data = 60000
if dataset_name == 'Fashion-MNIST':
    fixed_parameters = {'meta_input': 4,
                        'meta_output': 1,
                        'input_size': 784,
                        'num_classes': 10}
    hyperparameters = {'learner_hidden_widths': (256, 128, 100),
                       'meta_hidden_width': 30,
                       'learning_rate': 0.001,
                       'learner_batch_size': 10,
                       'update_rate': 0.001}
    num_data = 60000
