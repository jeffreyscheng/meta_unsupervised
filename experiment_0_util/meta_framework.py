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


def bandaid(method):
    @wraps(method)
    def bounced(*args, **kw):
        tick = time.time()
        while True:
            try:
                if time.time() - tick > time_out:
                    return 0
                result = method(*args, **kw)
                return result
            except (RuntimeError, MemoryError) as e:
                print("Encountered Error!")
                print("_____")
                traceback.print_exc()
                print("_____")
                # print(e.__dict__)
                # num = str(random.randint(0, 3))
                # # print("Bounced to machine " + num)
                # # os.environ["CUDA_VISIBLE_DEVICES"] = num
                # for obj in gc.get_objects():
                #     if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #         print(type(obj), obj.size())
                return 0

    return bounced


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

    @bandaid
    def train_model(self, phi=5, theta=1, intermediate_accuracy=False, return_model=False):
        learner, optimizer = self.create_learner_and_optimizer()
        tick = time.time()
        if gpu_bool:
            learner.cuda()

        # Loss and Optimizer
        learner_criterion = nn.CrossEntropyLoss()

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
                test_outputs = model(test_images, batch_num)
                _, predicted = torch.max(test_outputs.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum()
                if not isinstance(correct, int):
                    correct = correct.item()
                del test_images, test_outputs, predicted, test_labels
            print('Accuracy of the network on ' + batch_num * hyperparameters['learner_batch_size'] + ' test images: ' + str(100 * correct / total))
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
            outputs = learner.forward(images, batch_num)
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
