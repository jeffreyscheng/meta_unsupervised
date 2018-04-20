from meta_framework import *
from net_components import *
import numpy as np


class Vanilla(MetaFramework):
    def __init__(self, name, fixed_params, variable_params_range, variable_params_init):
        super(Vanilla, self).__init__(name, fixed_params, variable_params_range, variable_params_init)

    def train_model(self, mid1, mid2, meta_mid, meta_batch_size, learning_rate, update_rate, learner_batch_size):
        mid1 = math.floor(mid1)
        mid2 = math.floor(mid2)
        meta_mid = math.floor(meta_mid)
        meta_batch_size = math.floor(meta_batch_size)
        train_start_time = time.time()
        meta_input = self.fixed_params['meta_input']
        meta_output = self.fixed_params['meta_output']
        # meta_weight = Meta(meta_input, meta_mid, meta_output)
        input_size = self.fixed_params['input_size']
        num_classes = self.fixed_params['num_classes']
        learner_batch_size = math.floor(learner_batch_size)
        # learner_batch_size = 1
        learner = SingleNet(input_size, mid1, mid2, num_classes, meta_input, meta_mid, meta_output, learner_batch_size)

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
        learner_optimizer = torch.optim.Adam(learner.parameters(), lr=learning_rate)

        meta_criterion = nn.MSELoss()
        meta_optimizer = torch.optim.Adam(list(learner.conv1.parameters()) + list(learner.conv2.parameters()),
                                          lr=learning_rate)

        tick = time.time()
        meta_converged = False
        for meta_epoch in range(1, MetaFramework.num_epochs + 1):
            if time.time() - tick > MetaFramework.time_out:
                break
            for i, (images, labels) in enumerate(train_loader):
                if time.time() - tick > MetaFramework.time_out:
                    break
                if meta_converged is False:
                    meta_converged = learner.check_convergence()
                images = Variable(images.view(-1, 28 * 28))
                labels = Variable(labels)

                # move to CUDA
                if gpu_bool:
                    images = images.cuda()

                # Learner Forward + Backward + Optimize
                learner_optimizer.zero_grad()  # zero the gradient buffer
                outputs = learner(images)
                learner.update(update_rate, meta_epoch)
                learner_loss = learner_criterion(outputs, labels)
                # print(labels.data[0], ',', str(learner_loss.data[0]))
                learner_loss.backward()
                # for param in learner.weight_params:
                #     print(param)
                #     print(learner.param_state[param].grad)
                learner_optimizer.step()

                if not meta_converged:
                    # wrangling v_i, v_j, w_ij to go into MetaDataset
                    for param in learner.weight_params:
                        grad = torch.unsqueeze(learner.param_state[param].grad, 0)
                        grad = torch.unsqueeze(grad, 1).expand(learner_batch_size, -1, -1, -1)
                        learner.metadata[param] = torch.cat((learner.metadata[param], grad), dim=1)
                        cube = learner.metadata[param].size()
                        learner.metadata[param] = learner.metadata[param].view(cube[0] * cube[2] * cube[3], cube[1])
                    all_metadata = torch.cat(list(learner.metadata.values()), dim=0)
                    metadata_size = all_metadata.size()[0]
                    try:
                        # do with torch
                        sample_idx = np.random.choice(metadata_size, meta_batch_size, replace=False)
                        sampled_metadata = all_metadata[sample_idx, :]
                    except IndexError:
                        print("===INDEX ERROR===")
                        print(metadata_size)
                        print(meta_batch_size)
                        print(sample_idx)
                        sampled_metadata = all_metadata
                        print("===CLOSE===")
                    metadata_from_forward = MetaDataset(sampled_metadata)
                    meta_loader = torch.utils.data.DataLoader(dataset=metadata_from_forward,
                                                              batch_size=meta_batch_size,
                                                              shuffle=True)
                    # backprop error to metalearner with metadataset
                    for j, (triplets, grads) in enumerate(meta_loader):
                        tock = time.time()
                        triplets = Variable(triplets)
                        grads = Variable(grads)

                        # TODO check that triples and grad are on GPU
                        if gpu_bool:
                            print("should be 1")
                            print(torch.cuda.device_count())
                            print("should be GPU")
                            print(torch.cuda.get_device_name(0))

                        # Forward + Backward + Optimize
                        meta_optimizer.zero_grad()  # zero the gradient buffer
                        aug_triplets = torch.unsqueeze(torch.unsqueeze(triplets, 2), 3)
                        meta_outputs = torch.squeeze(learner.get_update(aug_triplets))
                        meta_loss = meta_criterion(meta_outputs, grads)
                        # print("Meta-Loss:" + str(meta_loss.data[0]))
                        try:
                            meta_loss.backward()
                            meta_optimizer.step()
                        except RuntimeError:
                            print(meta_outputs.size())
                            print(meta_loss.size())
                            print("Runtime Error")

        # Test the Model
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images.view(-1, 28 * 28))
            # to CUDA
            if gpu_bool:
                images = images.cuda()
            outputs = learner(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        return correct / total


vanilla_fixed_params = {'meta_input': 3, 'meta_output': 1, 'input_size': 784, 'num_classes': 10}
vanilla_params_range = {'mid1': (20, 800), 'mid2': (20, 800), 'meta_mid': (2, 10), 'meta_batch_size': (1, 10000),
                        'learning_rate': (0.000001, 0.001), 'update_rate': (0.000001, 0.001),
                        'learner_batch_size': (1, 100)}
vanilla_params_init = {'mid1': [400, 20], 'mid2': [200, 20],
                       'meta_mid': [5, 10], 'meta_batch_size': [100, 2337],
                       'learning_rate': [0.0001, 0.00093], 'update_rate': [0.0001, 0.00087],
                       'learner_batch_size': [5, 10]}

vanilla_frame = Vanilla('vanilla', vanilla_fixed_params, vanilla_params_range, vanilla_params_init)
# vanilla_frame.train_model(400, 200, 10, 3000, 0.001, 0.0001, 10)
vanilla_frame.optimize(40)
# vanilla_frame.analyze()
