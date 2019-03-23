import time
from functools import wraps
import traceback
from hyperparameters import *
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import random
import torch.nn as nn
from torch.autograd import Variable
import abc


class MetaFramework(object):

    def __init__(self, name, fixed_params):
        __metaclass__ = abc.ABCMeta
        self.name = name
        self.fixed_params = fixed_params
        if dataset_name == 'MNIST':
            # MNIST Dataset
            train_dataset = dsets.MNIST(root=root_directory + '/' + dataset_name + '/data',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

            test_dataset = dsets.MNIST(root=root_directory + '/' + dataset_name + '/data',
                                       train=False,
                                       transform=transforms.ToTensor())

        if dataset_name == 'Fashion-MNIST':
            # Fashion-MNIST Dataset
            train_dataset = dsets.FashionMNIST(root=root_directory + '/' + dataset_name + '/data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

            test_dataset = dsets.FashionMNIST(root=root_directory + '/' + dataset_name + '/data',
                                              train=False,
                                              transform=transforms.ToTensor())

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=hyperparameters['learner_batch_size'],
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=hyperparameters['learner_batch_size'],
                                                       shuffle=False)

    @abc.abstractmethod
    def create_learner_and_optimizer(self):
        return None, None

    def train_model(self, phi=5, theta=1, intermediate_accuracy=False, return_model=False):
        learner, optimizer = self.create_learner_and_optimizer()
        tick = time.time()
        if gpu_bool:
            learner.cuda()

        batch_num = 0
        learning_curve_list = []

        def test_model(model):
            # Test the Model
            correct = 0
            total = 0
            for test_images, test_labels in self.test_loader:
                test_images = Variable(test_images.view(-1, 28 * 28))
                # to CUDA
                if gpu_bool:
                    test_images = test_images.cuda()
                    test_labels = test_labels.cuda()
                test_outputs = model.forward(test_images, batch_num)
                _, predicted = torch.max(test_outputs.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum()
                if not isinstance(correct, int):
                    correct = correct.item()
                del test_images, test_outputs, predicted, test_labels
            print('Accuracy of the network on ' + str(
                batch_num * hyperparameters['learner_batch_size']) + ' test images: ' + str(100 * correct / total))
            accuracy = correct / total
            if intermediate_accuracy:
                now_phi = batch_num * hyperparameters['learner_batch_size'] / num_data
            else:
                now_phi = phi
            learning_curve_list.append({'phi': now_phi, 'theta': theta, 'accuracy': accuracy})
            return learning_curve_list

        def stop_training(tock, batch):
            return tock - tick > time_out or batch * hyperparameters['learner_batch_size'] / num_data > phi

        for i, (images, labels) in enumerate(self.train_loader):
            batch_num += 1
            if stop_training(time.time(), batch_num):
                break

            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            # move to CUDA
            if gpu_bool:
                images = images.cuda()
                labels = labels.cuda()

            # Learner Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = learner.train_forward(images, batch_num)
            if random.uniform(0, 1) < theta:
                learner_loss = learner_criterion(outputs, labels)
                # print(labels.data[0], ',', str(learner_loss.data[0]))
                learner_loss.backward()
                optimizer.step()
                del learner_loss

            if batch_num % 100 == 0 and not return_model and intermediate_accuracy:
                test_model(learner)

            del images, labels, outputs

        if return_model:
            return learner
        else:
            if not intermediate_accuracy:
                test_model(learner)
            del learner
            return learning_curve_list
