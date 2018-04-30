from meta_framework import *
from net_components import *
import numpy as np
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class FullyDiff(MetaFramework):
    def __init__(self, name, fixed_params, variable_params_range, variable_params_init):
        super(FullyDiff, self).__init__(name, fixed_params, variable_params_range, variable_params_init)

    # @bounce_gpu
    def train_model(self, mid1, mid2, meta_mid, learning_rate, learner_batch_size, update_rate):
        mid1 = math.floor(mid1)
        mid2 = math.floor(mid2)
        meta_mid = math.floor(meta_mid)
        train_start_time = time.time()
        meta_input = self.fixed_params['meta_input']
        meta_output = self.fixed_params['meta_output']
        # meta_weight = Meta(meta_input, meta_mid, meta_output)
        input_size = self.fixed_params['input_size']
        num_classes = self.fixed_params['num_classes']
        learner_batch_size = math.floor(learner_batch_size)
        # if learner_batch_size > 50:
        #     print(learner_batch_size)
        #     raise ValueError("wtf")
        # learner_batch_size = 1
        learner = DiffNet(input_size, mid1, mid2, num_classes, meta_input, meta_mid, meta_output, learner_batch_size,
                          update_rate)

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
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=learner_batch_size,
                                                  shuffle=False)

        # Loss and Optimizer
        learner_criterion = nn.CrossEntropyLoss()
        learner_optimizer = torch.optim.Adam(list(learner.parameters()) +
                                             list(learner.conv1.parameters()) +
                                             list(learner.conv2.parameters()), lr=learning_rate)

        tick = time.time()
        meta_converged = False
        batch_num = 0
        for epoch in range(1, MetaFramework.num_epochs + 1):
            if time.time() - tick > MetaFramework.time_out:
                break
            for i, (images, labels) in enumerate(train_loader):
                batch_num += 1
                if time.time() - tick > MetaFramework.time_out:
                    break
                if meta_converged is False:
                    meta_converged = learner.check_convergence()
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
                learner_loss = learner_criterion(outputs, labels)
                # print(labels.data[0], ',', str(learner_loss.data[0]))
                learner_loss.backward()
                # for param in learner.weight_params:
                #     print(param)
                #     print(learner.param_state[param].grad)
                learner_optimizer.step()
                # print("finished example!")
                del images, labels, outputs, learner_loss
            # print("finished epoch")


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
        # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        del learner
        return correct / total


fully_diff_fixed_params = {'meta_input': 3, 'meta_output': 1, 'input_size': 784, 'num_classes': 10}
fully_diff_params_range = {'mid1': (20, 800), 'mid2': (20, 800), 'meta_mid': (2, 10),
                           'learning_rate': (0.000001, 0.001), 'update_rate': (0.000001, 0.001),
                           'learner_batch_size': (1, 1000)}
fully_diff_params_init = {'mid1': [400, 20], 'mid2': [200, 20],
                          'meta_mid': [5, 10],
                          'learning_rate': [0.0001, 0.00093], 'update_rate': [0.0001, 0.00087],
                          'learner_batch_size': [50, 500]}

fully_diff_frame = FullyDiff('fully_diff', fully_diff_fixed_params, fully_diff_params_range, fully_diff_params_init)
# fully_diff_frame.train_model(400, 200, 10, 3000, 0.001, 0.0001, 10)
fully_diff_frame.optimize(10)
# fully_diff_frame.analyze()
