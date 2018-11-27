from experiment_0_util.meta_framework import *
from hyperparameters import *
import random
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import os


class OptimalNet(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, output_size, batch_size, rate):
        super(OptimalNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.batch_size = batch_size
        self.impulse = None
        self.metadata = {}
        self.param_state = self.state_dict(keep_vars=True)
        self.param_names = ['fc1.weight', 'fc2.weight', 'fc3.weight']
        self.layers = [self.fc1, self.fc2, self.fc3]
        self.rate = rate

    @staticmethod
    def get_update(v_j):
        return v_j * 0.01

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
                # weight_stack = layer.unsqueeze(0).expand(stack_dim)
            except RuntimeError:
                print(self.batch_size)
                print(stack_dim)
                print(vi.size())
                print(old_vj.size())
                print(layer.size())
                input_stack = vi.unsqueeze(1).expand(stack_dim)
                output_stack = old_vj.unsqueeze(2).expand(stack_dim)
                # weight_stack = layer.unsqueeze(0).expand(stack_dim)
                for obj in gc.get_objects():
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
            shift = self.get_update(output_stack) * self.rate / batch_num
            shift = torch.mean(shift, dim=0)

            # output, update weights
            # print(type(input_stack))
            # print(type(shift))
            vj_update = input_stack * shift
            out = old_vj + torch.sum(vj_update, dim=2)
            layer.data += shift.data
            del old_vj, output_stack, shift
        return out

    def check_convergence(self):
        return False


class OptimalFrame(MetaFramework):
    def __init__(self, name, fixed_params):
        super(OptimalFrame, self).__init__(name, fixed_params)

    @bandaid
    def train_model(self, mid1, mid2, learning_rate, learner_batch_size, update_rate, theta=1, phi=15):
        mid1 = math.floor(mid1)
        mid2 = math.floor(mid2)
        input_size = self.fixed_params['input_size']
        num_classes = self.fixed_params['num_classes']
        learner_batch_size = math.floor(learner_batch_size)
        learner = OptimalNet(input_size, mid1, mid2, num_classes, learner_batch_size, update_rate)
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
        learner_optimizer = torch.optim.Adam(list(learner.parameters()), lr=learning_rate)

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


optimal_frame = OptimalFrame('optimal', fixed_parameters)
