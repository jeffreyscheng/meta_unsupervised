from meta_framework import *
from net_components import *


class Separate(MetaFramework):
    def __init__(self, name, fixed_params, variable_params_range, variable_params_init):
        super(Separate, self).__init__(name, fixed_params, variable_params_range, variable_params_init)

    def train_model(self, mid1, mid2, meta_mid, meta_batch_size, learning_rate, update_rate, bridge):
        mid1 = math.floor(mid1)
        mid2 = math.floor(mid2)
        meta_mid = math.floor(meta_mid)
        meta_batch_size = math.floor(meta_batch_size)
        bridge = math.floor(bridge)
        train_start_time = time.time()
        meta_weight = Meta(meta_input, meta_mid, meta_output)
        unsupervised_learner = UnsupervisedNet(input_size, mid1, bridge, meta_weight, learner_batch_size)
        supervised_learner = SupervisedNet(bridge, mid2, num_classes)

        # Loss and Optimizer
        learner_criterion = nn.CrossEntropyLoss()
        learner_optimizer = torch.optim.Adam(supervised_learner.parameters(), lr=learning_rate)

        meta_criterion = nn.MSELoss()
        meta_optimizer = torch.optim.Adam(meta_weight.parameters(), lr=learning_rate)

        meta_epoch = 0
        while time.time() - train_start_time < 360:
        # while time.time() - train_start_time < total_runtime:
            meta_epoch += 1
            for i, (images, labels) in enumerate(train_loader):
                if time.time() - train_start_time > total_runtime:
                    break
                images = Variable(images.view(-1, 28 * 28))
                labels = Variable(labels)

                # Forward + Backward + Optimize
                learner_optimizer.zero_grad()  # zero the gradient buffer
                middle_outputs = unsupervised_learner(images)
                outputs = supervised_learner(middle_outputs)
                unsupervised_learner.update(update_rate, meta_epoch)
                learner_loss = learner_criterion(outputs, labels)
                print(labels.data[0], ',', str(learner_loss.data[0]))
                learner_loss.backward()
                learner_optimizer.step()

                for param in unsupervised_learner.weight_params:
                    grad = torch.unsqueeze(unsupervised_learner.param_state[param].grad, 2)
                    # print(grad.size())
                    # print(learner.metadata[param].size())
                    unsupervised_learner.metadata[param] = torch.cat((unsupervised_learner.metadata[param], grad), dim=2)
                    cube_dim = unsupervised_learner.metadata[param].size()
                    unsupervised_learner.metadata[param] = unsupervised_learner.metadata[param].view(cube_dim[0] * cube_dim[1], cube_dim[2])
                all_metadata = torch.cat(list(unsupervised_learner.metadata.values()), dim=0)
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
                    # print(time.time() - tock)
                # print("ONE FULL PASS")
                # print(time.clock() - tick)
                #
                # if (i + 1) % 100 == 0:
                # print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                #       % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, learner_loss.data[0]))
                # print('Epoch [%d], Loss: %.4f' % (meta_epoch, learner_loss.data[0]))
                # print('Took ', time.clock() - tick, ' seconds')
                # meta_epoch += 1

        # Test the Model
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images.view(-1, 28 * 28))
            middle_outputs = unsupervised_learner(images)
            outputs = supervised_learner(middle_outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        return correct / total


separate_fixed_params = {'meta_input': 3, 'meta_output': 1, 'input_size': 784, 'num_classes': 10,
                         'learner_batch_size': 1}
separate_params_range = {'mid1': (20, 800), 'mid2': (20, 800), 'meta_mid': (2, 10), 'meta_batch_size': (1, 10000),
                         'learning_rate': (0.000001, 0.001), 'update_rate': (0.000001, 0.001), 'bridge': (20, 800)}
separate_params_init = {'mid1': [400, 20], 'mid2': [200, 20],
                        'meta_mid': [5, 10], 'meta_batch_size': [100, 2337],
                        'learning_rate': [0.0001, 0.00093], 'update_rate': [0.0001, 0.00087], 'bridge': [300, 20]}

separate_frame = Separate('separate', separate_fixed_params, separate_params_range, separate_params_init)
separate_frame.train_model(400, 100, 10, 3000, 0.0009, 0.0001, 200)
# separate_frame.optimize(1)
# separate_frame.analyze()
