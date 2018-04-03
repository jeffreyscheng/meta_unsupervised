from net_components import *
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import math
from bayes_opt import BayesianOptimization

required = object()

# Hyper2 Parameters
total_runtime = 3600  # half an hour
# total_runtime = 5

# Fixed Hyper Parameters
meta_input = 3
meta_output = 1
input_size = 784
num_classes = 10
learner_batch_size = 10

# Variable Hyper Parameters
starting_learner_mid1 = 400
starting_learner_mid2 = 200
starting_meta_mid = 5
# starting_num_epochs = 5
starting_meta_sample_per_iter = 10000
starting_meta_batch_size = 100
starting_learning_rate = 0.0001
starting_meta_rate = 0.0001

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
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=learner_batch_size,
                                          shuffle=False)


# MetaDataset

class MetaDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, metadata):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        metadata = metadata.data
        self.len = metadata.size()[0]
        self.x_data = metadata[:, 0:3]
        self.y_data = metadata[:, 3:4]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# DEFINE MODELS

def train_model(mid1=starting_learner_mid1, mid2=starting_learner_mid2, meta_mid=starting_meta_mid,
                meta_sample_per_iter=starting_meta_sample_per_iter, meta_batch_size=starting_meta_batch_size,
                learning_rate=starting_learning_rate, meta_rate=starting_meta_rate):
    mid1 = math.floor(mid1)
    mid2 = math.floor(mid2)
    meta_mid = math.floor(meta_mid)
    meta_sample_per_iter = math.floor(meta_sample_per_iter)
    meta_batch_size = math.floor(meta_batch_size)
    train_start_time = time.time()
    meta_weight = Meta(meta_input, meta_mid, meta_output)
    learner = SingleNet(input_size, mid1, mid2, num_classes, meta_weight, learner_batch_size)

    # Loss and Optimizer
    learner_criterion = nn.CrossEntropyLoss()
    learner_optimizer = torch.optim.Adam(learner.parameters(), lr=learning_rate)

    meta_criterion = nn.MSELoss()
    meta_optimizer = torch.optim.Adam(meta_weight.parameters(), lr=learning_rate)

    meta_epoch = 1
    while time.time() - train_start_time < total_runtime:
        for i, (images, labels) in enumerate(train_loader):
            if time.time() - train_start_time > total_runtime:
                break
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            learner_optimizer.zero_grad()  # zero the gradient buffer
            outputs = learner(images)
            learner.update(meta_rate, meta_epoch)
            learner_loss = learner_criterion(outputs, labels)
            # print("Loss:" + str(learner_loss))
            learner_loss.backward()
            learner_optimizer.step()

            for param in learner.weight_params:
                grad = torch.unsqueeze(learner.param_state[param].grad, 2)
                # print(grad.size())
                # print(learner.metadata[param].size())
                learner.metadata[param] = torch.cat((learner.metadata[param], grad), dim=2)
                cube_dim = learner.metadata[param].size()
                learner.metadata[param] = learner.metadata[param].view(cube_dim[0] * cube_dim[1], cube_dim[2])
            all_metadata = torch.cat(list(learner.metadata.values()), dim=0)
            # print(all_metadata.size())
            metadata_size = all_metadata.size()[0]
            # trol = time.time()
            if meta_sample_per_iter > metadata_size:
                return 0
            sample_idx = np.random.choice(metadata_size, meta_sample_per_iter, replace=False)
            # print("yay samples")
            # print(time.time() - trol)
            sampled_metadata = all_metadata[sample_idx, :]
            metadata_from_forward = MetaDataset(sampled_metadata)
            meta_loader = torch.utils.data.DataLoader(dataset=metadata_from_forward,
                                                      batch_size=meta_batch_size,
                                                      shuffle=True)
            for j, (triplets, grads) in enumerate(meta_loader):
                tock = time.time()
                triplets = Variable(triplets)
                grads = Variable(grads)

                # Forward + Backward + Optimize
                meta_optimizer.zero_grad()  # zero the gradient buffer
                meta_outputs = meta_weight(triplets)
                meta_loss = meta_criterion(meta_outputs, grads)
                # print("Meta-Loss:" + str(meta_loss))
                meta_loss.backward()
                meta_optimizer.step()
            #     # print(time.time() - tock)
            # print("ONE FULL PASS")
            # print(time.clock() - tick)
            #
            # if (i + 1) % 100 == 0:
                # print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                #       % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, learner_loss.data[0]))
                # print('Epoch [%d], Loss: %.4f' % (meta_epoch + 1, learner_loss.data[0]))
            #     print('Took ', time.clock() - tick, ' seconds')
            # meta_epoch += 1

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28 * 28))
        outputs = learner(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return correct / total


# train_model()

param_dict = {'mid1': (20, 1000), 'mid2': (20, 1000), 'meta_mid': (2, 10), 'meta_sample_per_iter': (10001, 100000),
              'meta_batch_size': (0, 10000), 'learning_rate': (0, 0.1), 'meta_rate': (0, 0.01)}
bayes = BayesianOptimization(train_model, param_dict)

bayes.explore({'mid1': [starting_learner_mid1], 'mid2': [starting_learner_mid2], 'meta_mid': [starting_meta_mid],
               'meta_sample_per_iter': [starting_meta_sample_per_iter], 'meta_batch_size': [starting_meta_batch_size],
               'learning_rate': [starting_learning_rate], 'meta_rate': [starting_meta_rate]})

bayes.maximize(init_points=5, n_iter=2, kappa=2)

print(bayes.res['max'])
print(bayes.res['all'])

# Save the Model
# torch.save(learner.state_dict(), 'single_single_weights.pkl')
# torch.save(learner.state_dict(), 'single_single_model.pkl')
