import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


# Template for Bias and Weight Update Metalearner
# inputs in order v_i, w_ij, v_j
class Meta(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Meta, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out


# Template for Vanilla Structure
class VanillaNet(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, output_size, meta_input, meta_hidden, meta_output, batch_size):
        super(VanillaNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.batch_size = batch_size
        self.impulse = None
        self.conv1 = nn.Conv2d(in_channels=meta_input, out_channels=meta_hidden, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=meta_hidden, out_channels=meta_output, kernel_size=1, bias=True)
        self.metadata = {}

        self.param_state = self.state_dict(keep_vars=True)
        keys = list(self.param_state.keys())

        def is_weight_param(param):
            return (".weight" in param) and ("meta" not in param) and ("conv" not in param)

        self.weight_params = [key for key in keys if is_weight_param(key)]

    # get new weight
    def get_update(self, meta_input_stack):
        out = self.conv1(meta_input_stack)
        out = self.conv2(out)
        out = torch.squeeze(out, 1)
        return out

    def forward(self, x):
        if self.impulse is not None:
            if len(self.impulse) > 4:
                raise ValueError("long impulse!")
        self.metadata = {}
        out = x
        self.impulse = [out]
        out = self.fc1(out)
        out = self.relu(out)
        self.impulse.append(out)
        out = self.fc2(out)
        out = self.relu(out)
        self.impulse.append(out)
        out = self.fc3(out)
        out = self.relu(out)
        self.impulse.append(out)
        return out

    def update(self, rate, epoch, change_weights=True):
        # print(weight_params)
        if len(self.weight_params) != len(self.impulse) - 1:  # LHS: learner param layers, RHS: intermediate outputs
            print("Keys:" + str(len(self.weight_params)))
            print(self.weight_params)
            print("Impulse:" + str(len(self.impulse)))
            print(self.impulse)
            raise ValueError("Num keys not 1 less than num impulses")
        for i in range(0, len(self.weight_params)):
            layer = self.param_state[self.weight_params[i]]
            input_layer = self.impulse[i]
            output_layer = self.impulse[i + 1]
            stack_dim = self.batch_size, layer.size()[0], layer.size()[1]
            input_stack = input_layer.unsqueeze(1).expand(stack_dim)
            output_stack = output_layer.unsqueeze(2).expand(stack_dim)
            weight_stack = layer.unsqueeze(0).expand(stack_dim)
            meta_inputs = torch.stack((input_stack, weight_stack, output_stack), dim=3)
            meta_inputs = meta_inputs.permute(0, 3, 1, 2)
            self.metadata[self.weight_params[i]] = meta_inputs
            if change_weights:
                batch_shift = torch.mean(torch.clamp(self.get_update(meta_inputs), -1000000, 1000000), 0)
                layer.data += batch_shift.data * rate / epoch
                del batch_shift
            del input_stack, output_stack, weight_stack, meta_inputs

    def check_convergence(self):
        return False


# Template for Single Structure
class DiffNet(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, output_size, meta_input, meta_hidden, meta_output, batch_size,
                 rate):
        super(DiffNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.batch_size = batch_size
        self.impulse = None
        self.conv1 = nn.Conv2d(in_channels=meta_input, out_channels=meta_hidden, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=meta_hidden, out_channels=meta_output, kernel_size=1, bias=True)
        self.metadata = {}
        self.param_state = self.state_dict(keep_vars=True)
        self.param_names = ['fc1.weight', 'fc2.weight', 'fc3.weight']
        self.layers = [self.fc1, self.fc2, self.fc3]
        self.rate = rate

    # get new weight
    def get_update(self, meta_input_stack):
        return torch.squeeze(self.conv2(self.conv1(meta_input_stack)), 1)

    # @timeit
    def forward(self, x, batch_num):
        if self.impulse is not None:
            if len(self.impulse) > 4:
                raise ValueError("long impulse!")
        self.metadata = {}
        out = x
        for layer_num in range(0, 3):
            layer = self.param_state[self.param_names[layer_num]]
            vi = out
            old_vj = self.layers[layer_num](out)
            old_vj = self.relu(old_vj)
            stack_dim = self.batch_size, layer.size()[0], layer.size()[1]
            # try:
            input_stack = vi.unsqueeze(1).expand(stack_dim)
            output_stack = old_vj.unsqueeze(2).expand(stack_dim)
            weight_stack = layer.unsqueeze(0).expand(stack_dim)
            # except RuntimeError:
            #     print(stack_dim)
            #     print(vi.size())
            #     print(old_vj.size())
            #     print(layer.size())
            #     input_stack = vi.unsqueeze(1).expand(stack_dim)
            #     output_stack = old_vj.unsqueeze(2).expand(stack_dim)
            #     weight_stack = layer.unsqueeze(0).expand(stack_dim)
            meta_inputs = torch.stack((input_stack, weight_stack, output_stack), dim=3).permute(0, 3, 1, 2)
            shift = self.get_update(meta_inputs) * self.rate / batch_num

            # output, update weights
            out = old_vj + torch.sum(input_stack * shift, dim=2)
            layer.data += torch.mean(shift.data, dim=0)
            del old_vj, input_stack, output_stack, weight_stack, meta_inputs, shift
        return out

    def check_convergence(self):
        return False
