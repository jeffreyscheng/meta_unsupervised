from experiment_0_util.meta_framework import *
from hyperparameters import *
import torch
import torch.nn as nn
import gc


class MetaLearnerNet(nn.Module):

    def __init__(self, meta_input, meta_hidden, meta_output):
        super(MetaLearnerNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(meta_input, meta_hidden)
        self.fc2 = nn.Linear(meta_hidden, meta_output)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1))


# Template for Single Structure
class LearnerNet(nn.Module):

    def __init__(self, input_size, learner_hidden, output_size, meta_input, meta_hidden, meta_output, batch_size,
                 rate):
        super(LearnerNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, learner_hidden)
        self.fc2 = nn.Linear(learner_hidden, output_size)
        self.batch_size = batch_size
        self.metalearner = MetaLearnerNet(meta_input, meta_hidden, meta_output)
        self.param_state = self.state_dict(keep_vars=True)
        self.param_names = ['fc1.weight', 'fc2.weight']
        self.layers = [self.fc1, self.fc2]
        self.rate = rate

    def forward(self, x, batch_num):
        out = x
        for layer_num in range(0, len(self.layers)):
            layer = self.param_state[self.param_names[layer_num]]
            vi = out
            old_vj = self.layers[layer_num](out)
            old_vj = self.relu(old_vj)
            stack_dim = self.batch_size, layer.size()[0], layer.size()[1]
            try:
                input_stack = vi.unsqueeze(1).expand(stack_dim)
                output_stack = old_vj.unsqueeze(2).expand(stack_dim)
                weight_stack = layer.unsqueeze(0).expand(stack_dim)
            except RuntimeError:  # frequent memory errors happen on this step
                print(self.batch_size)
                print(stack_dim)
                print(vi.size())
                print(old_vj.size())
                print(layer.size())
                input_stack = vi.unsqueeze(1).expand(stack_dim)
                output_stack = old_vj.unsqueeze(2).expand(stack_dim)
                weight_stack = layer.unsqueeze(0).expand(stack_dim)
                for obj in gc.get_objects():
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
            stack_prod = stack_dim[0] * stack_dim[1] * stack_dim[2]
            meta_inputs = torch.stack((input_stack, weight_stack, output_stack), dim=3).permute(1, 2, 3, 0).contiguous().view(3, stack_prod)
            print(meta_inputs.size())

            def test_fn(x_i):
                print(x_i.size())
                print(self.metalearner(x_i).size())
                return self.metalearner(x_i) * self.rate / batch_num
            shift = torch.stack([test_fn(x_i) for _, x_i in enumerate(torch.unbind(meta_inputs, dim=1), 0)], dim=0).squeeze(0)

            # output, update weights
            print(old_vj.size())
            print(input_stack.size())
            print(shift.size())
            out = old_vj + torch.sum(input_stack * shift, dim=2)
            layer.data += torch.mean(shift.data, dim=0)
            del old_vj, input_stack, output_stack, weight_stack, meta_inputs, shift
        return out


class HebbianFrame(MetaFramework):

    def __init__(self, name, fixed_params):
        super(HebbianFrame, self).__init__(name, fixed_params)

    def create_learner_and_optimizer(self):
        learner = LearnerNet(fixed_parameters['input_size'],
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
