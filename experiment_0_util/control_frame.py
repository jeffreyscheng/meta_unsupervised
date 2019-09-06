from experiment_0_util.meta_framework import *
from hyperparameters import *
import torch.nn as nn


# Template for Control Structure
class ControlNet(nn.Module):

    def __init__(self, input_size, hidden_widths, output_size):
        super(ControlNet, self).__init__()
        self.relu = nn.ReLU()
        hidden1, hidden2, hidden3 = hidden_widths
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, output_size)

    def forward(self, x, batch_num=1):
        return self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))

    def train_forward(self, x, batch_num=1):
        return self.forward(x, batch_num)


class ControlFrame(MetaFramework):
    def __init__(self, name, fixed_params):
        super(ControlFrame, self).__init__(name, fixed_params)

    def create_learner_and_optimizer(self):
        learner = ControlNet(fixed_parameters['input_size'],
                             hyperparameters['learner_hidden_widths'],
                             fixed_parameters['num_classes'])
        optimizer = base_optimizer(list(learner.parameters()), lr=hyperparameters['learning_rate'])
        return learner, optimizer


control_frame = ControlFrame('control', fixed_parameters)
