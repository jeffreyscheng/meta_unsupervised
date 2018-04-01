from net_components import *
import torch
from torch.autograd import Variable

required = object()

# Hyper Parameters
meta_input = 3
meta_output = 1
input_size = 784
mid = 800
num_classes = 10
num_epochs = 5
batch_size = 1
meta_batch_size = 100
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

# DEFINE MODELS

meta_weight = Meta(3, 5, 1)
learner = SingleNet(784, 400, 200, 10, meta_weight)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
# print(len(list(learner.parameters())))
#
# for name, param in learner.named_parameters():
#     if param.requires_grad:
#         print(name)
# raise ValueError("cri")
optimizer = torch.optim.Adam(learner.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = learner(images)
        loss = criterion(outputs, labels)
        print("Loss:" + str(loss))
        loss.backward()

        # print("LOLAL")
        # crie = list(learner.parameters())
        # for cri in crie:
        #     print("NEW")
        #     print(sum(cri.grad.data))
        # # print(list(learner.parameters())[0].grad)
        # # print(list(learner.parameters())[1].grad)
        # print("HOUFSDJ")

        optimizer.step()

        for param in learner.weight_params:
            grad = learner.param_state[param].grad
            learner.metadata[param]['grad'] = grad
            print(grad.size())


        # print(learner.meta_weight.get_loss())
        raise ValueError("dundun")

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

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
