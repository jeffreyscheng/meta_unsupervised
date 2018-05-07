from meta_framework import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# Template for Vanilla Structure
class VanillaNet(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, output_size, meta_input, meta_hidden, meta_output, batch_size):
        super(VanillaNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.batch_size = batch_size
        self.impulse = None
        self.conv1 = nn.Conv2d(in_channels=meta_input, out_channels=meta_hidden, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=meta_hidden, out_channels=meta_output, kernel_size=1, bias=True)
        self.metadata = {}

        self.param_state = self.state_dict(keep_vars=True)
        keys = list(self.param_state.keys())

        def is_weight_param(param):
            return (".weight" in param) and ("meta" not in param) and ("conv" not in param)

        self.weight_params = [key for key in keys if is_weight_param(key)]

    # get new weight
    def get_update(self, meta_input_stack):
        out = self.conv1(meta_input_stack)
        out = self.conv2(out)
        out = torch.squeeze(out, 1)
        return out

    def forward(self, x):
        if self.impulse is not None:
            if len(self.impulse) > 4:
                raise ValueError("long impulse!")
        self.metadata = {}
        out = x
        self.impulse = [out]
        out = self.fc1(out)
        out = self.relu(out)
        self.impulse.append(out)
        out = self.fc2(out)
        out = self.relu(out)
        self.impulse.append(out)
        out = self.fc3(out)
        out = self.relu(out)
        self.impulse.append(out)
        return out

    def update(self, rate, epoch, change_weights=True):
        # print(weight_params)
        if len(self.weight_params) != len(self.impulse) - 1:  # LHS: learner param layers, RHS: intermediate outputs
            print("Keys:" + str(len(self.weight_params)))
            print(self.weight_params)
            print("Impulse:" + str(len(self.impulse)))
            print(self.impulse)
            raise ValueError("Num keys not 1 less than num impulses")
        for i in range(0, len(self.weight_params)):
            layer = self.param_state[self.weight_params[i]]
            input_layer = self.impulse[i]
            output_layer = self.impulse[i + 1]
            stack_dim = self.batch_size, layer.size()[0], layer.size()[1]
            input_stack = input_layer.unsqueeze(1).expand(stack_dim)
            output_stack = output_layer.unsqueeze(2).expand(stack_dim)
            weight_stack = layer.unsqueeze(0).expand(stack_dim)
            meta_inputs = torch.stack((input_stack, weight_stack, output_stack), dim=3)
            meta_inputs = meta_inputs.permute(0, 3, 1, 2)
            self.metadata[self.weight_params[i]] = meta_inputs
            if change_weights:
                layer.data += torch.mean(torch.clamp(self.get_update(meta_inputs), -1000000, 1000000), 0).data * \
                              rate / epoch
            del input_stack, output_stack, weight_stack, meta_inputs

    def check_convergence(self):
        return False


class Vanilla(MetaFramework):
    def __init__(self, name, fixed_params, variable_params_range, variable_params_init):
        super(Vanilla, self).__init__(name, fixed_params, variable_params_range, variable_params_init)

    @bandaid
    def train_model(self, mid1, mid2, meta_mid, meta_batch_size, learning_rate, update_rate, learner_batch_size,
                    theta=1):
        mid1 = math.floor(mid1)
        mid2 = math.floor(mid2)
        meta_mid = math.floor(meta_mid)
        meta_batch_size = math.floor(meta_batch_size)
        meta_input = self.fixed_params['meta_input']
        meta_output = self.fixed_params['meta_output']
        input_size = self.fixed_params['input_size']
        num_classes = self.fixed_params['num_classes']
        learner_batch_size = math.floor(learner_batch_size)
        learner = VanillaNet(input_size, mid1, mid2, num_classes, meta_input, meta_mid, meta_output, learner_batch_size)

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

                # Learner Forward + Backward + Optimize
                learner_optimizer.zero_grad()  # zero the gradient buffer
                outputs = learner(images)
                learner.update(update_rate, batch_num)
                if random.uniform(0, 1) < theta:
                    learner_loss = learner_criterion(outputs, labels)
                    learner_loss.backward()
                    learner_optimizer.step()
                    if not meta_converged:
                        # wrangling v_i, v_j, w_ij to go into MetaDataset
                        for param in learner.weight_params:
                            grad = torch.unsqueeze(learner.param_state[param].grad, 0)
                            grad = torch.unsqueeze(grad, 1).expand(learner_batch_size, -1, -1, -1)
                            learner.metadata[param] = torch.cat((learner.metadata[param], grad), dim=1)
                            cube = learner.metadata[param].size()
                            learner.metadata[param] = learner.metadata[param].view(cube[0] * cube[2] * cube[3], cube[1])
                            del grad
                        try:
                            all_metadata = torch.cat(list(learner.metadata.values()), dim=0)
                        except(RuntimeError, MemoryError):
                            for elem in list(learner.metadata.values()):
                                print(elem.size())
                            traceback.print_exc()
                            raise MemoryError
                        metadata_size = all_metadata.size()[0]
                        sample_idx = np.random.choice(metadata_size, meta_batch_size, replace=False)
                        sampled_metadata = all_metadata[sample_idx, :]
                        metadata_from_forward = MetaDataset(sampled_metadata)
                        meta_loader = torch.utils.data.DataLoader(dataset=metadata_from_forward,
                                                                  batch_size=meta_batch_size,
                                                                  shuffle=True)
                        # backprop error to metalearner with metadataset
                        for j, (triplets, grads) in enumerate(meta_loader):
                            triplets = Variable(triplets)
                            grads = Variable(grads)

                            # Forward + Backward + Optimize
                            meta_optimizer.zero_grad()  # zero the gradient buffer
                            aug_triplets = torch.unsqueeze(torch.unsqueeze(triplets, 2), 3)
                            meta_outputs = torch.squeeze(learner.get_update(aug_triplets))
                            meta_loss = meta_criterion(meta_outputs, grads)
                            # try:
                            #     meta_loss.backward()
                            #     meta_optimizer.step()
                            # except RuntimeError:
                            #     print(meta_outputs.size())
                            #     print(meta_loss.size())
                            #     print("Runtime Error")
                            del triplets, grads, meta_loss, aug_triplets, meta_outputs
                        del all_metadata
                    del images, labels, outputs, learner_loss

        # Test the Model
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images.view(-1, 28 * 28))
            # to CUDA
            if gpu_bool:
                images = images.cuda()
                labels = labels.cuda()
            outputs = learner(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            del images, outputs, predicted
        # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        del learner
        return correct / total


vanilla_fixed_params = {'meta_input': 3, 'meta_output': 1, 'input_size': 784, 'num_classes': 10}
vanilla_params_range = {'mid1': (20, 800), 'mid2': (20, 800), 'meta_mid': (2, 10), 'meta_batch_size': (1, 10000),
                        'learning_rate': (0.000001, 0.001), 'update_rate': (0.000001, 0.001),
                        'learner_batch_size': (1, 500)}
vanilla_params_init = {'mid1': [400, 20], 'mid2': [200, 20],
                       'meta_mid': [5, 10], 'meta_batch_size': [100, 2337],
                       'learning_rate': [0.0001, 0.00093], 'update_rate': [0.0001, 0.00087],
                       'learner_batch_size': [50, 200]}

vanilla_frame = Vanilla('vanilla', vanilla_fixed_params, vanilla_params_range, vanilla_params_init)
# vanilla_frame.train_model(400, 200, 10, 3000, 0.001, 0.0001, 10)
vanilla_frame.optimize(MetaFramework.optimize_num)
# vanilla_frame.analyze()
