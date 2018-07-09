from experiment_0_util.meta_framework import *
import random
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# Template for Single Structure
class HebbianNet(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, output_size, meta_input, meta_hidden, meta_output, batch_size,
                 rate):
        super(HebbianNet, self).__init__()
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
            try:
                input_stack = vi.unsqueeze(1).expand(stack_dim)
                output_stack = old_vj.unsqueeze(2).expand(stack_dim)
                weight_stack = layer.unsqueeze(0).expand(stack_dim)
            except RuntimeError:
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
            meta_inputs = torch.stack((input_stack, weight_stack, output_stack), dim=3).permute(0, 3, 1, 2)
            shift = self.get_update(meta_inputs) * self.rate / batch_num

            # output, update weights
            out = old_vj + torch.sum(input_stack * shift, dim=2)
            layer.data += torch.mean(shift.data, dim=0)
            del old_vj, input_stack, output_stack, weight_stack, meta_inputs, shift
        return out

    def check_convergence(self):
        return False


class HebbianFrame(MetaFramework):
    def __init__(self, name, fixed_params, variable_params_range, variable_params_init):
        super(HebbianFrame, self).__init__(name, fixed_params, variable_params_range, variable_params_init)

    @bandaid
    def train_model(self, mid1, mid2, meta_mid, learning_rate, learner_batch_size, update_rate, theta=1, phi=15):
        mid1 = math.floor(mid1)
        mid2 = math.floor(mid2)
        meta_mid = math.floor(meta_mid)
        meta_input = self.fixed_params['meta_input']
        meta_output = self.fixed_params['meta_output']
        input_size = self.fixed_params['input_size']
        num_classes = self.fixed_params['num_classes']
        learner_batch_size = math.floor(learner_batch_size)
        learner = HebbianNet(input_size, mid1, mid2, num_classes, meta_input, meta_mid, meta_output, learner_batch_size,
                             update_rate)
        # print(learner_batch_size)

        # check if GPU is available
        gpu_bool = torch.cuda.device_count() > 0
        if gpu_bool:
            learner.cuda()

        # MNIST Dataset
        train_dataset = dsets.MNIST(root='./data',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

        test_dataset = dsets.MNIST(root='./data',
                                   train=False,
                                   transform=transforms.ToTensor())

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=learner_batch_size,
                                                   shuffle=True, drop_last=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=learner_batch_size,
                                                  shuffle=False, drop_last=True)

        # Loss and Optimizer
        learner_criterion = nn.CrossEntropyLoss()
        learner_optimizer = torch.optim.Adam(list(learner.parameters()) +
                                             list(learner.conv1.parameters()) +
                                             list(learner.conv2.parameters()), lr=learning_rate)

        tick = time.time()
        # meta_converged = False
        batch_num = 0

        def stop_training(tock, batch):
            return tock - tick > MetaFramework.time_out or batch * learner_batch_size / MetaFramework.num_data > phi

        for i, (images, labels) in enumerate(train_loader):
            batch_num += 1
            if stop_training(time.time(), batch_num):
                # print("time out!")
                break
            # if meta_converged is False:
            #     meta_converged = learner.check_convergence()
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            # move to CUDA
            if gpu_bool:
                images = images.cuda()
                labels = labels.cuda()

            # most stuff before here

            # Learner Forward + Backward + Optimize
            learner_optimizer.zero_grad()  # zero the gradient buffer
            outputs = learner.forward(images, batch_num)
            if random.uniform(0, 1) < theta:
                learner_loss = learner_criterion(outputs, labels)
                # print(labels.data[0], ',', str(learner_loss.data[0]))
                learner_loss.backward()
                learner_optimizer.step()
                del images, labels, outputs, learner_loss

        tick2 = time.time()
        # Test the Model
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images.view(-1, 28 * 28))
            # to CUDA
            if gpu_bool:
                images = images.cuda()
                labels = labels.cuda()
            outputs = learner(images, batch_num)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            del images, outputs, predicted
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        print("Time spent training:", tick2 - tick)
        print("Time spent testing:", time.time() - tick2)
        del learner
        return correct / total


hebbian_fixed_params = {'meta_input': 3, 'meta_output': 1, 'input_size': 784, 'num_classes': 10}
hebbian_params_range = {'mid1': (20, 800), 'mid2': (20, 800), 'meta_mid': (2, 10),
                        'learning_rate': (0.000001, 0.001), 'update_rate': (0.000001, 0.001),
                        'learner_batch_size': (1, 500)}
hebbian_params_init = {'mid1': [400, 20], 'mid2': [200, 20],
                       'meta_mid': [5, 10],
                       'learning_rate': [0.0001, 0.00093], 'update_rate': [0.0001, 0.00087],
                       'learner_batch_size': [50, 200]}

hebbian_frame = HebbianFrame('hebbian', hebbian_fixed_params, hebbian_params_range, hebbian_params_init)
# hebbian_frame.train_model(246, 146, 6, 0.00066695, 47, 0.00020694, phi=0.001)
# hebbian_frame.optimize(MetaFramework.optimize_num)
# hebbian_frame.analyze()
