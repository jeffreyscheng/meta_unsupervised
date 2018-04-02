from net_components import *
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import time

required = object()

# Hyper Parameters
meta_input = 3
meta_output = 1
input_size = 784
mid = 800
num_classes = 10
num_epochs = 5
batch_size = 1
meta_batch_size = 1000
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

meta_weight = Meta(3, 5, 1)
learner = SingleNet(784, 400, 200, 10, meta_weight)

# Loss and Optimizer
learner_criterion = nn.CrossEntropyLoss()
learner_optimizer = torch.optim.Adam(learner.parameters(), lr=learning_rate)

meta_criterion = nn.MSELoss()
meta_optimizer = torch.optim.Adam(meta_weight.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        tick = time.clock()
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # Forward + Backward + Optimize
        learner_optimizer.zero_grad()  # zero the gradient buffer
        outputs = learner(images)
        learner_loss = learner_criterion(outputs, labels)
        # print("Loss:" + str(learner_loss))
        learner_loss.backward()

        # print("LOLAL")
        # crie = list(learner.parameters())
        # for cri in crie:
        #     print("NEW")
        #     print(sum(cri.grad.data))
        # # print(list(learner.parameters())[0].grad)
        # # print(list(learner.parameters())[1].grad)
        # print("HOUFSDJ")

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

        metadata_from_forward = MetaDataset(all_metadata)
        meta_loader = torch.utils.data.DataLoader(dataset=metadata_from_forward,
                                                   batch_size=meta_batch_size,
                                                   shuffle=True)
        for j, (triplets, grads) in enumerate(meta_loader):
            triplets = Variable(triplets)
            grads = Variable(grads)

            # Forward + Backward + Optimize
            meta_optimizer.zero_grad()  # zero the gradient buffer
            meta_outputs = meta_weight(triplets)
            meta_loss = meta_criterion(meta_outputs, grads)
            # print("Meta-Loss:" + str(meta_loss))
            meta_loss.backward()

            meta_optimizer.step()

        # raise ValueError("dundun")

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, learner_loss.data[0]))
            print('Took ', time.clock() - tick, ' seconds')

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

# Save the Model
torch.save(learner.state_dict(), 'single_single_weights.pkl')
torch.save(learner.state_dict(), 'single_single_model.pkl')
