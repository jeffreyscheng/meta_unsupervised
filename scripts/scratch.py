from experiment_0_util.run_experiment import *
import pandas as pd
import os
import time
from torch.utils.tensorboard import SummaryWriter
from os.path import join

tick = time.time()
logs_dir = join(root_directory, 'logs', 'exp1-tuning' + str(time.time()))
writer = SummaryWriter(logs_dir)

experiment_1_data_path = os.path.join(final_data_path, 'experiment_1_data.csv')

learner = ControlNet(784, (256, 128, 100), 10)
optimizers = [base_optimizer(learner.parameters(), lr=0.0000001)]

phi = 5
theta = 1
intermediate_accuracy = True
return_model = False

batch_num = 0
learning_curve_list = []

train_dataset = dsets.FashionMNIST(root=root_directory + '/' + dataset_name + '/data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

test_dataset = dsets.FashionMNIST(root=root_directory + '/' + dataset_name + '/data',
                                  train=False,
                                  transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=hyperparameters['learner_batch_size'],
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=hyperparameters['learner_batch_size'],
                                          shuffle=False)


def test_model(model):
    # Test the Model
    correct = 0
    total = 0
    for test_images, test_labels in test_loader:
        test_images = Variable(test_images.view(-1, 28 * 28))
        # to CUDA
        test_images = push_to_gpu(test_images)
        test_labels = push_to_gpu(test_labels)
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
    writer.add_scalar('Accuracy', accuracy, batch_num)


for epoch in range(50):
    for i, (images, labels) in enumerate(train_loader):
        batch_num += 1

        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # move to CUDA
        images = push_to_gpu(images)
        labels = push_to_gpu(labels)

        # Learner Forward + Backward + Optimize
        # optimizer.zero_grad()  # zero the gradient buffer
        outputs = learner.train_forward(images, batch_num)
        for optimizer in optimizers:
            optimizer.zero_grad()  # we do this here since the forward pass needs the gradient
        learner_loss = learner_criterion(outputs, labels)
        # print(labels.data[0], ',', str(learner_loss.data[0]))
        learner_loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        writer.add_scalar('Loss', learner_loss, batch_num)
        del images, labels, learner_loss
        if batch_num % 10000 == 0:
            test_model(learner)
