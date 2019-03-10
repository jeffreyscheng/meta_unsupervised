from experiment_0_util.meta_framework import *
from hyperparameters import *
import torch
import torch.nn as nn


# Template for Control Structure
class ControlNet(nn.Module):

    def __init__(self, input_size, hidden, output_size):
        super(ControlNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, output_size)

    def forward(self, x, batch_num=1):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out


class ControlFrame(MetaFramework):
    def __init__(self, name, fixed_params):
        super(ControlFrame, self).__init__(name, fixed_params)

    def create_learner_and_optimizer(self):
        learner = ControlNet(fixed_parameters['input_size'],
                             hyperparameters['learner_hidden_width'],
                             fixed_parameters['num_classes'])
        optimizer = base_optimizer(list(learner.parameters()), lr=hyperparameters['learning_rate'])
        return learner, optimizer


control_frame = ControlFrame('control', fixed_parameters)
