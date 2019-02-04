from experiment_0_util.meta_framework import *
from hyperparameters import *
import torch
import torch.nn as nn


# Template for Control Structure
class ControlNet(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(ControlNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    # get new weight
    def get_update(self, meta_input_stack):
        out = self.conv1(meta_input_stack)
        out = self.conv2(out)
        out = torch.squeeze(out, 1)
        return out

    def forward(self, x, batch_num=1):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out


class ControlFrame(MetaFramework):
    def __init__(self, name, fixed_params):
        super(ControlFrame, self).__init__(name, fixed_params)

    def create_learner_and_optimizer(self):
        learner = ControlNet(fixed_parameters['input_size'],
                             hyperparameters['mid1'],
                             hyperparameters['mid2'],
                             fixed_parameters['num_classes'])
        optimizer = base_optimizer(list(learner.parameters()), lr=hyperparameters['learning_rate'])
        return learner, optimizer


control_frame = ControlFrame('control', fixed_parameters)
