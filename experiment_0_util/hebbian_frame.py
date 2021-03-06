from experiment_0_util.meta_framework import *
from hyperparameters import *
import torch


# Template for Single Structure
class HebbianNet(nn.Module):

    def __init__(self, input_size, hidden_widths, output_size, meta_input, meta_hidden, meta_output, batch_size, rate):
        super(HebbianNet, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        hidden1, hidden2, hidden3 = hidden_widths
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, output_size)
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(in_channels=meta_input, out_channels=meta_hidden, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=meta_hidden, out_channels=meta_output, kernel_size=1, bias=True)
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        self.rate = rate
        self.most_recent_Hebbian_updates = []

    # get new weight
    def get_update(self, meta_input_stack):
        return torch.squeeze(self.conv2(self.tanh(self.conv1(meta_input_stack))), 1)

    def forward(self, x, batch_num=1):
        return self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))

    def train_forward(self, x, batch_num=1):
        self.most_recent_Hebbian_updates = []
        out = x
        for layer_num, layer_obj in enumerate(self.layers):
            layer = layer_obj.weight
            vi = out
            old_vj = self.layers[layer_num](out)
            stack_dim = self.batch_size, layer.size()[0], layer.size()[1]
            input_stack = vi.unsqueeze(1).expand(stack_dim)
            output_stack = old_vj.unsqueeze(2).expand(stack_dim)
            weight_stack = layer.unsqueeze(0).expand(stack_dim)
            if batch_num <= 1:
                grad_stack = push_to_gpu(torch.zeros(weight_stack.size()))
            else:
                grad_stack = torch.abs(layer.grad.unsqueeze(0).expand(stack_dim))
            meta_inputs = torch.stack((input_stack, weight_stack, output_stack, grad_stack), dim=3).permute(0, 3, 1, 2)
            shift = self.get_update(meta_inputs) * self.rate
            # if batch_num % 100 == 0:
            # print(batch_num, torch.mean(torch.abs(layer.grad)) / torch.mean(torch.abs(shift)))

            # output, update weights
            self.most_recent_Hebbian_updates.append(shift)
            out = old_vj + torch.sum(input_stack * shift, dim=2)
            if layer_num < len(self.layers):
                out = self.relu(out)
            layer.data += torch.mean(shift.data, dim=0)
            del old_vj, input_stack, output_stack, weight_stack, meta_inputs, shift
        return out

    def get_learner_parameters(self):
        return list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc3.parameters()) + list(
            self.fc4.parameters())

    def get_metalearner_parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters())

    def get_learner_gradient_norm(self):
        norms = []
        for p in list(filter(lambda p: p.grad is not None, self.get_learner_parameters())):
            norms.append(p.grad.data)
        return sum([torch.sum(x ** 2).item() for x in norms]) ** 0.5

    def get_metalearner_gradient_norm(self):
        norms = []
        for p in list(filter(lambda p: p.grad is not None, self.get_metalearner_parameters())):
            norms.append(p.grad.data)
        return sum([torch.sum(x ** 2).item() for x in norms]) ** 0.5

    def get_hebbian_update_norm(self):
        norms = [t.norm(2) for t in self.most_recent_Hebbian_updates]
        return sum([torch.sum(x ** 2).item() for x in norms]) ** 0.5


class HebbianFrame(MetaFramework):

    def __init__(self, name, fixed_params):
        super(HebbianFrame, self).__init__(name, fixed_params)

    def create_learner_and_optimizer(self):
        learner = HebbianNet(fixed_parameters['input_size'],
                             hyperparameters['learner_hidden_widths'],
                             fixed_parameters['num_classes'],
                             fixed_parameters['meta_input'],
                             hyperparameters['meta_hidden_width'],
                             fixed_parameters['meta_output'],
                             hyperparameters['learner_batch_size'],
                             hyperparameters['hebbian_update_rate'])
        learner_optimizer = base_optimizer(learner.get_learner_parameters(), lr=hyperparameters['learner_learning_rate'])
        meta_optimizer = base_optimizer(learner.get_metalearner_parameters(), lr=hyperparameters['meta_learning_rate'])
        return learner, [learner_optimizer, meta_optimizer]


hebbian_frame = HebbianFrame('hebbian', fixed_parameters)
