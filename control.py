from meta_framework import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# Template for Control Structure
class ControlNet(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, output_size, batch_size):
        super(ControlNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.batch_size = batch_size
        self.impulse = None
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

    def check_convergence(self):
        return False


class Control(MetaFramework):
    def __init__(self, name, fixed_params, variable_params_range, variable_params_init, theta):
        super(Control, self).__init__(name, fixed_params, variable_params_range, variable_params_init, theta)

    @bandaid
    def train_model(self, mid1, mid2, learning_rate, learner_batch_size):
        mid1 = math.floor(mid1)
        mid2 = math.floor(mid2)
        meta_input = self.fixed_params['meta_input']
        meta_output = self.fixed_params['meta_output']
        input_size = self.fixed_params['input_size']
        num_classes = self.fixed_params['num_classes']
        learner_batch_size = math.floor(learner_batch_size)
        learner = ControlNet(input_size, mid1, mid2, num_classes, learner_batch_size)

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
                # learner.update(update_rate, batch_num)
                if random.uniform(0, 1) < self.theta:
                    learner_loss = learner_criterion(outputs, labels)
                    learner_loss.backward()
                    learner_optimizer.step()

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


control_fixed_params = {'meta_input': 3, 'meta_output': 1, 'input_size': 784, 'num_classes': 10}
control_params_range = {'mid1': (20, 800), 'mid2': (20, 800),
                        'learning_rate': (0.000001, 0.001),
                        'learner_batch_size': (1, 500)}
control_params_init = {'mid1': [400, 20], 'mid2': [200, 20],
                       'learning_rate': [0.0001, 0.00093],
                       'learner_batch_size': [50, 200]}

control_frame = Control('control', control_fixed_params, control_params_range, control_params_init, 1)
# control_frame.train_model(400, 200, 10, 3000, 0.001, 0.0001, 10)
control_frame.optimize(10)
# control_frame.analyze()
