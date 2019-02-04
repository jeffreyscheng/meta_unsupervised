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
                if time.time() - tick > MetaFramework.time_out:
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


def create_learner():
    return 0


class MetaFramework(object):
    time_out = 20 * 60
    num_data = 60000

    def __init__(self, name, fixed_params):
        __metaclass__ = abc.ABCMeta
        self.name = name
        self.fixed_params = fixed_params
        if dataset_name == 'MNIST':
            # MNIST Dataset
            train_dataset = dsets.MNIST(root='./' + dataset_name + '/data',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

            test_dataset = dsets.MNIST(root='./' + dataset_name + '/data',
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
    def train_model(self, phi=5, theta=1, return_model=False):
        learner, optimizer = self.create_learner_and_optimizer()
        tick = time.time()
        if gpu_bool:
            learner.cuda()

        # Loss and Optimizer
        learner_criterion = nn.CrossEntropyLoss()

        batch_num = 0

        def stop_training(tock, batch):
            return tock - tick > MetaFramework.time_out or batch * hyperparameters[
                'learner_batch_size'] / MetaFramework.num_data > phi

        for i, (images, labels) in enumerate(self.train_loader):
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
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = learner.forward(images, batch_num)
            if random.uniform(0, 1) < theta:
                learner_loss = learner_criterion(outputs, labels)
                # print(labels.data[0], ',', str(learner_loss.data[0]))
                learner_loss.backward()
                optimizer.step()
                del images, labels, outputs, learner_loss

        tick2 = time.time()
        # Test the Model
        correct = 0
        total = 0
        for images, labels in self.test_loader:
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
        if return_model:
            return learner
        else:
            del learner
            return correct / total
