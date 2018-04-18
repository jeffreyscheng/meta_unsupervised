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


# Template for Single Structure
class SingleNet(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, output_size, meta_input, meta_hidden, meta_output, batch_size):
        super(SingleNet, self).__init__()
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
        # print("BEGINNING UPDATE")
        out = self.conv1(meta_input_stack)
        # print("First conv:", out.size())
        out = self.conv2(out)
        # print("Second conv:", out.size())
        out = torch.squeeze(out, 1)
        return out

    def forward(self, x):
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
            # print(layer)
            input_layer = self.impulse[i]
            output_layer = self.impulse[i + 1]
            # print(input_layer.size())
            # print(output_layer.size())
            # print(layer.size())
            stack_dim = self.batch_size, layer.size()[0], layer.size()[1]
            input_stack = input_layer.unsqueeze(1).expand(stack_dim)
            output_stack = output_layer.unsqueeze(2).expand(stack_dim)
            weight_stack = layer.unsqueeze(0).expand(stack_dim)
            meta_inputs = torch.stack((input_stack, weight_stack, output_stack), dim=3)
            meta_inputs = meta_inputs.permute(0, 3, 1, 2)
            self.metadata[self.weight_params[i]] = meta_inputs
            # print(meta_inputs.size())
            if change_weights:
                shift = self.get_update(meta_inputs)
                clipped_shift = torch.clamp(shift, -1000000, 1000000)
                batch_shift = torch.mean(clipped_shift, 0)
                layer.data += batch_shift.data * rate / epoch

    def check_convergence(self):
        return False


# Template for Single Structure
class UnsupervisedNet(nn.Module):

    def __init__(self, input_size, hidden, output_size, meta_weight, batch_size):
        super(UnsupervisedNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, output_size)
        self.batch_size = batch_size
        self.impulse = None
        self.meta_weight = meta_weight
        self.metadata = {}

        self.param_state = self.state_dict(keep_vars=True)
        keys = list(self.param_state.keys())

        def is_weight_param(param):
            return (".weight" in param) and ("meta" not in param)

        self.weight_params = [key for key in keys if is_weight_param(key)]

    # get new weight
    def get_update(self, v_i, w_ij, v_j):
        inputs = Variable(torch.Tensor([v_i, w_ij, v_j]), requires_grad=True)
        # print("break")
        return self.meta_weight.forward(inputs).data[0]

    def forward(self, x):
        self.metadata = {}
        out = x
        self.impulse = [out]
        out = self.fc1(out)
        out = self.relu(out)
        self.impulse.append(out)
        out = self.fc2(out)
        out = self.relu(out)
        self.impulse.append(out)
        return out

    # @timeit
    # compute output and propagate Hebbian updates
    # takes roughly 10 ms per
    def update(self, rate, epoch, change_weights=True):
        # print(weight_params)
        if len(self.weight_params) != len(self.impulse) - 1:
            print("Keys:" + str(len(self.weight_params)))
            print(self.weight_params)
            print("Impulse:" + str(len(self.impulse)))
            print(self.impulse)
            raise ValueError("Num keys not 1 less than num impulses")
        for i in range(0, len(self.weight_params)):
            layer = self.param_state[self.weight_params[i]]
            # print(layer)
            input_layer = self.impulse[i]
            output_layer = self.impulse[i + 1]
            # print(input_layer.size())
            # print(output_layer.size())
            # print(layer.size())
            # stack_dim = self.batch_size, layer.size()[0], layer.size()[1]
            # input_stack = input_layer.unsqueeze(1).expand(stack_dim)
            # output_stack = output_layer.unsqueeze(2).expand(stack_dim)
            # weight_stack = layer.unsqueeze(0).expand(stack_dim)
            # meta_inputs = torch.stack((input_stack, weight_stack, output_stack), dim=3)
            # print(meta_inputs.size())

            input_stack = input_layer.repeat(output_layer.size(1), 1)
            output_stack = output_layer.repeat(input_layer.size(1), 1).t()
            meta_inputs = torch.stack((input_stack, layer, output_stack), dim=2)
            self.metadata[self.weight_params[i]] = meta_inputs
            # layer.data = torch.stack([self.meta_weight() for i, x_i in enumerate(torch.unbind(x, dim=0), 0)], dim=0)
            # TODO: vectorize with apply_()
            if change_weights:
                for input_index in range(0, len(input_layer)):
                    for output_index in range(0, len(output_layer)):
                        # print(input_layer.data)
                        input_to_neuron = input_layer.data[0, input_index]
                        # print(input_to_neuron)
                        output_from_neuron = output_layer.data[0, output_index]
                        # print(output_from_neuron)
                        neuron_weight = layer.data[output_index, input_index]
                        # print(neuron_weight)
                        shift = self.get_update(input_to_neuron, neuron_weight, output_from_neuron)
                        # print(new_weight)
                        layer.data[output_index, input_index] -= shift * rate / epoch


# Template for Bias and Weight Update Metalearner
# inputs in order v_i, w_ij, v_j
class SupervisedNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SupervisedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out


# Template for Single Structure
class SingleNetOld(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, output_size, meta_weight, batch_size):
        super(SingleNetOld, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.batch_size = batch_size
        self.impulse = None
        self.meta_weight = meta_weight
        self.metadata = {}

        self.param_state = self.state_dict(keep_vars=True)
        keys = list(self.param_state.keys())

        def is_weight_param(param):
            return (".weight" in param) and ("meta" not in param)

        self.weight_params = [key for key in keys if is_weight_param(key)]

    # get new weight
    def get_update(self, v_i, w_ij, v_j):
        print(v_i.data)
        print(w_ij)
        print(v_j)
        inputs = torch.Tensor([v_i, w_ij, v_j])
        inputs = Variable(inputs, requires_grad=True)
        print("break")
        return self.meta_weight.forward(inputs).data[0]

    def get_update2(self, vec):
        return self.meta_weight.forward(vec).data[0]

    def forward(self, x):
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

    # @timeit
    # compute output and propagate Hebbian updates
    # takes roughly 10 ms per
    def update(self, rate, epoch, change_weights=True):
        # print(weight_params)
        if len(self.weight_params) != len(self.impulse) - 1:
            print("Keys:" + str(len(self.weight_params)))
            print(self.weight_params)
            print("Impulse:" + str(len(self.impulse)))
            print(self.impulse)
            raise ValueError("Num keys not 1 less than num impulses")
        for i in range(0, len(self.weight_params)):
            layer = self.param_state[self.weight_params[i]]
            # print(layer)
            input_layer = self.impulse[i]
            output_layer = self.impulse[i + 1]
            print(input_layer.size())
            print(output_layer.size())
            print(layer.size())
            stack_dim = self.batch_size, layer.size()[0], layer.size()[1]
            input_stack = input_layer.unsqueeze(1).expand(stack_dim)
            output_stack = output_layer.unsqueeze(2).expand(stack_dim)
            weight_stack = layer.unsqueeze(0).expand(stack_dim)
            meta_inputs = torch.stack((input_stack, weight_stack, output_stack), dim=3)
            print(meta_inputs.size())
            tick = time.time()

            meta_outputs = torch.zeros(meta_inputs.size()[0], meta_inputs.size()[1], meta_inputs.size()[2])
            if change_weights:
                ### iteration
                for batch_num in range(0, meta_inputs.size()[0]):
                    for input_index in range(0, len(input_layer)):
                        for output_index in range(0, len(output_layer)):
                            v_i = meta_inputs[batch_num, input_index, output_index, 0]
                            w_ij = meta_inputs[batch_num, input_index, output_index, 1]
                            v_j = meta_inputs[batch_num, input_index, output_index, 2]
                            meta_outputs[batch_num, input_index, output_index] = self.get_update(v_i, w_ij, v_j)
                            print("got here")
                ### unbind -> apply -> stack // reduction op // module
                # torch.stack([function(x_i, other_input[i]) for i, x_i in enumerate(torch.unbind(x, dim=axis), 0)], dim=axis)
            meta_outputs = torch.mean(meta_outputs, 0)
            print(meta_outputs.size())
            print(time.time() - tick)
            raise ValueError("lol")

            # input_stack = input_layer.repeat(output_layer.size(1), 1)
            # output_stack = output_layer.repeat(input_layer.size(1), 1).t()
            # meta_inputs = torch.stack((input_stack, layer, output_stack), dim=2)
            # self.metadata[self.weight_params[i]] = meta_inputs
            # layer.data = torch.stack([self.meta_weight() for i, x_i in enumerate(torch.unbind(x, dim=0), 0)], dim=0)
            # # TODO: vectorize with apply_()
            # if change_weights:
            #     for input_index in range(0, len(input_layer)):
            #         for output_index in range(0, len(output_layer)):
            #             # print(input_layer.data)
            #             input_to_neuron = input_layer.data[0, input_index]
            #             # print(input_to_neuron)
            #             output_from_neuron = output_layer.data[0, output_index]
            #             # print(output_from_neuron)
            #             neuron_weight = layer.data[output_index, input_index]
            #             # print(neuron_weight)
            #             shift = self.get_update(input_to_neuron, neuron_weight, output_from_neuron)
            #             # print(new_weight)
            #             layer.data[output_index, input_index] -= shift * rate / epoch
