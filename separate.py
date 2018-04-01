import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from collections import defaultdict, Iterable

from copy import deepcopy
from itertools import chain
import math

required = object()

# Hyper Parameters
meta_input = 3
meta_output = 1
input_size = 784
mid1 = 400
mid2 = 200
mid3 = 100
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

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
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model (1 hidden layer)
class Meta(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, num_unsupervised_iterations=0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.num_unsupervised_iterations = num_unsupervised_iterations
        self.impulse = []

    def update(self, x):
        out = self.relu(x)
        self.impulse = []
        self.impulse += out
        out = self.fc1(out)
        out = self.relu(out)
        self.impulse += out
        out = self.fc2(out)
        out = self.relu(out)
        self.impulse += out
        param_state = self.state_dict(keep_vars=True)
        keys = list(param_state.keys())
        if len(keys) != len(self.impulse) - 1:
            raise ValueError("Num keys not 1 less than num impulses")
        for i in range(0, len(keys)):
            layer = param_state[i]
            input_layer = self.impulse[i]
            output_layer = self.impulse[i + 1]


        return out

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def loss(outputs, labels):
unsupervised_learner = Net(input_size, mid1, mid2, num_unsupervised_iterations=100)
# TODO: train  unsupervised_learner


supervised_learner = Net(mid2, mid3, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(supervised_learner.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        supervised_input = unsupervised_learner(images)
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = supervised_learner(supervised_input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28))
    supervised_input = unsupervised_learner(images)
    outputs = supervised_learner(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(unsupervised_learner.state_dict(), 'unsupervised_model.pkl')
torch.save(supervised_learner.state_dict(), 'supervised_model.pkl')
