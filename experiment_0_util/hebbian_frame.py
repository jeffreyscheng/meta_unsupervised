from experiment_0_util.meta_framework import *
from hyperparameters import *
import torch


# Template for Single Structure
class HebbianNet(nn.Module):

    def __init__(self, input_size, hidden, output_size, meta_input, meta_hidden, meta_output, batch_size, rate):
        super(HebbianNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, output_size)
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(in_channels=meta_input, out_channels=meta_hidden, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=meta_hidden, out_channels=meta_output, kernel_size=1, bias=True)
        self.param_state = self.state_dict(keep_vars=True)
        self.param_names = ['fc1.weight', 'fc2.weight', 'fc3.weight']
        self.layers = [self.fc1, self.fc2]
        self.rate = rate

    # get new weight
    def get_update(self, meta_input_stack):
        return torch.squeeze(self.conv2(self.relu(self.conv1(meta_input_stack))), 1)

    # @timeit
    def forward(self, x, batch_num=1):
        out = x
        for layer_num in range(0, len(self.layers)):
            layer = self.param_state[self.param_names[layer_num]]
            vi = out
            old_vj = self.layers[layer_num](out)
            old_vj = self.relu(old_vj)
            stack_dim = self.batch_size, layer.size()[0], layer.size()[1]
            input_stack = vi.unsqueeze(1).expand(stack_dim)
            output_stack = old_vj.unsqueeze(2).expand(stack_dim)
            weight_stack = layer.unsqueeze(0).expand(stack_dim)
            meta_inputs = torch.stack((input_stack, weight_stack, output_stack), dim=3).permute(0, 3, 1, 2)
            shift = self.get_update(meta_inputs) * self.rate / batch_num

            # output, update weights
            out = old_vj + torch.sum(input_stack * shift, dim=2)
            layer.data += torch.mean(shift.data, dim=0)
            del old_vj, input_stack, output_stack, weight_stack, meta_inputs, shift
        return out


class HebbianFrame(MetaFramework):

    def __init__(self, name, fixed_params):
        super(HebbianFrame, self).__init__(name, fixed_params)

    def create_learner_and_optimizer(self):
        learner = HebbianNet(fixed_parameters['input_size'],
                             hyperparameters['learner_hidden_width'],
                             fixed_parameters['num_classes'],
                             fixed_parameters['meta_input'],
                             hyperparameters['meta_hidden_width'],
                             fixed_parameters['meta_output'],
                             hyperparameters['learner_batch_size'],
                             hyperparameters['update_rate'])
        optimizer = base_optimizer(learner.parameters(), lr=hyperparameters['learning_rate'])
        return learner, optimizer


hebbian_frame = HebbianFrame('hebbian', fixed_parameters)
