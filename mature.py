from meta_framework import *
from net_components import *


class Mature(MetaFramework):
    def __init__(self, name, fixed_params, variable_params_range, variable_params_init):
        super(Mature, self).__init__(name, fixed_params, variable_params_range, variable_params_init)

    def train_model(self, mid1, mid2, meta_mid, meta_batch_size, learning_rate, update_rate):
        mid1 = math.floor(mid1)
        mid2 = math.floor(mid2)
        meta_mid = math.floor(meta_mid)
        meta_batch_size = math.floor(meta_batch_size)
        train_start_time = time.time()
        meta_weight = Meta(meta_input, meta_mid, meta_output)
        learner = SingleNet(input_size, mid1, mid2, num_classes, meta_weight, learner_batch_size)

        # Loss and Optimizer
        learner_criterion = nn.CrossEntropyLoss()
        learner_optimizer = torch.optim.Adam(learner.parameters(), lr=learning_rate)

        meta_criterion = nn.MSELoss()
        meta_optimizer = torch.optim.Adam(meta_weight.parameters(), lr=learning_rate)

        # training meta-learner
        while time.time() - train_start_time < total_runtime:
            for i, (images, labels) in enumerate(train_loader):
                if time.time() - train_start_time > total_runtime:
                    break
                images = Variable(images.view(-1, 28 * 28))
                labels = Variable(labels)

                # Forward + Backward + Optimize
                learner_optimizer.zero_grad()  # zero the gradient buffer
                outputs = learner(images)
                # learner.update(update_rate, meta_epoch)
                learner_loss = learner_criterion(outputs, labels)
                # print(labels.data[0], ',', str(learner_loss.data[0]))
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
                try:
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

        # training actual learner
        while time.time() - train_start_time < total_runtime:
            for i, (images, labels) in enumerate(train_loader):
                if time.time() - train_start_time > total_runtime:
                    break
                images = Variable(images.view(-1, 28 * 28))
                labels = Variable(labels)

                # Forward + Backward + Optimize
                learner_optimizer.zero_grad()  # zero the gradient buffer
                outputs = learner(images)
                learner.update(update_rate, meta_epoch)
                learner_loss = learner_criterion(outputs, labels)
                # print(labels.data[0], ',', str(learner_loss.data[0]))
                learner_loss.backward()
                learner_optimizer.step()

        # Test the Model
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images.view(-1, 28 * 28))
            outputs = learner(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        return correct / total


mature_fixed_params = {'meta_input': 3, 'meta_output': 1, 'input_size': 784, 'num_classes': 10,
                       'learner_batch_size': 1}
mature_params_range = {'mid1': (20, 800), 'mid2': (20, 800), 'meta_mid': (2, 10), 'meta_batch_size': (1, 10000),
                       'learning_rate': (0.000001, 0.001), 'update_rate': (0.000001, 0.001)}
mature_params_init = {'mid1': [400, 20], 'mid2': [200, 20],
                      'meta_mid': [5, 10], 'meta_batch_size': [100, 2337],
                      'learning_rate': [0.0001, 0.00093], 'update_rate': [0.0001, 0.00087]}

mature_frame = Mature('mature', mature_fixed_params, mature_params_range, mature_params_init)
mature_frame.optimize(1)
mature_frame.analyze()
