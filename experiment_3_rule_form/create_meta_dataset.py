from experiment_0_util.hebbian_frame import *
from experiment_0_util.meta_framework import *
import random
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import os
import pandas as pd
import gc

metalearner_directory = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                                     'temp_data',
                                     'metalearners')
metadata_path = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                             'temp_data',
                             'metadata.csv')

#
# here = os.path.dirname(os.path.abspath(__file__))
# metalearner_directory = here + '/metalearners'
# metadata_path = here + os.sep + 'metadata.csv'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# Template for Single Structure
class WritableHebbianNet(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, output_size, meta_input, meta_hidden, meta_output, batch_size,
                 rate):
        super(WritableHebbianNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.batch_size = batch_size
        self.impulse = None
        self.conv1 = nn.Conv2d(in_channels=meta_input, out_channels=meta_hidden, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=meta_hidden, out_channels=meta_output, kernel_size=1, bias=True)
        self.impulse = {}
        self.param_state = self.state_dict(keep_vars=True)
        self.param_names = ['fc1.weight', 'fc2.weight', 'fc3.weight']
        self.layers = [self.fc1, self.fc2, self.fc3]
        self.rate = rate

    # get new weight
    def get_update(self, meta_input_stack):
        return torch.squeeze(self.conv2(self.conv1(meta_input_stack)), 1)

    # get new weight
    def get_single_update(self, meta_inputs):
        return torch.squeeze(
            self.conv2(self.conv1(torch.Tensor(meta_inputs).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3))))

    # @timeit
    def forward(self, x, batch_num):
        if self.impulse is not None:
            if len(self.impulse) > 4:
                raise ValueError("long impulse!")
        self.impulse.clear()
        del self.impulse
        gc.collect()
        self.impulse = {}
        out = x
        for layer_num in range(0, 3):
            layer = self.param_state[self.param_names[layer_num]]
            vi = out
            old_vj = self.layers[layer_num](out)
            old_vj = self.relu(old_vj)
            stack_dim = self.batch_size, layer.size()[0], layer.size()[1]
            try:
                input_stack = vi.unsqueeze(1).expand(stack_dim)
                output_stack = old_vj.unsqueeze(2).expand(stack_dim)
                weight_stack = layer.unsqueeze(0).expand(stack_dim)
            except RuntimeError:
                print(self.batch_size)
                print(stack_dim)
                print(vi.size())
                print(old_vj.size())
                print(layer.size())
                input_stack = vi.unsqueeze(1).expand(stack_dim)
                output_stack = old_vj.unsqueeze(2).expand(stack_dim)
                weight_stack = layer.unsqueeze(0).expand(stack_dim)
                for obj in gc.get_objects():
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
            meta_inputs = torch.stack((input_stack, weight_stack, output_stack), dim=3).permute(0, 3, 1, 2)
            self.impulse[self.param_names[layer_num]] = meta_inputs
            shift = self.get_update(meta_inputs) * self.rate / batch_num

            # output, update weights
            out = old_vj + torch.sum(input_stack * shift, dim=2)
            layer.data += torch.mean(shift.data, dim=0)
            del old_vj, input_stack, output_stack, weight_stack, meta_inputs, shift
        return out

    def check_convergence(self):
        return False


class WritableHebbianFrame(MetaFramework):
    num_samp = 10

    def __init__(self, name, fixed_params, variable_params_range, variable_params_init):
        super(WritableHebbianFrame, self).__init__(name, fixed_params, variable_params_range, variable_params_init)

    @bandaid
    def train_model(self, mid1, mid2, meta_mid, learning_rate, learner_batch_size, update_rate, theta=1, phi=15):
        mid1 = math.floor(mid1)
        mid2 = math.floor(mid2)
        meta_mid = math.floor(meta_mid)
        meta_input = self.fixed_params['meta_input']
        meta_output = self.fixed_params['meta_output']
        input_size = self.fixed_params['input_size']
        num_classes = self.fixed_params['num_classes']
        learner_batch_size = math.floor(learner_batch_size)
        learner = WritableHebbianNet(input_size, mid1, mid2, num_classes, meta_input, meta_mid, meta_output,
                                     learner_batch_size, update_rate)
        # print(learner_batch_size)
        if os.path.isfile(metadata_path) and False:
            metadata_df = pd.read_csv(metadata_path)
        else:
            metadata_df = pd.DataFrame(columns=['v_i', 'w_ij', 'v_j', 'grad'])

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
                                                   shuffle=True, drop_last=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=learner_batch_size,
                                                  shuffle=False, drop_last=True)

        # Loss and Optimizer
        learner_criterion = nn.CrossEntropyLoss()
        learner_optimizer = torch.optim.Adam(list(learner.parameters()) +
                                             list(learner.conv1.parameters()) +
                                             list(learner.conv2.parameters()), lr=learning_rate)

        tick = time.time()
        # meta_converged = False
        batch_num = 0

        def stop_training(tock, batch):
            return tock - tick > MetaFramework.time_out or batch * learner_batch_size / MetaFramework.num_data > phi

        for i, (images, labels) in enumerate(train_loader):
            batch_num += 1
            if stop_training(time.time(), batch_num):
                # print("time out!")
                break
            # if meta_converged is False:
            #     meta_converged = learner.check_convergence()
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            # move to CUDA
            if gpu_bool:
                images = images.cuda()
                labels = labels.cuda()

            # Learner Forward + Backward + Optimize
            learner_optimizer.zero_grad()  # zero the gradient buffer
            outputs = learner.forward(images, batch_num)
            if random.uniform(0, 1) < theta * 0.01:  # only sample 1% of the time... otherwise the dset blows up to 2GB
                learner_loss = learner_criterion(outputs, labels)
                # print(labels.data[0], ',', str(learner_loss.data[0]))
                learner_loss.backward()

                tick = time.time()
                grad_of_param = {}
                for name, parameter in learner.named_parameters():
                    grad_of_param[name] = parameter.grad

                # pushes gradients into the metalearner stack
                for layer_name in learner.impulse:
                    # print(layer_name)
                    # print(learner.impulse[layer_name].size())
                    meta_stack_size = list(learner.impulse[layer_name].size())
                    meta_stack_size[1] = 1
                    layer_grad = grad_of_param[layer_name].unsqueeze(0).unsqueeze(1).expand(meta_stack_size)
                    # print(layer_grad.size())
                    learner.impulse[layer_name] = torch.cat((learner.impulse[layer_name], layer_grad), dim=1)
                    # print(learner.impulse[layer_name].size())

                    # samples for metadata_df
                    batch = [random.randint(0, meta_stack_size[0] - 1) for _ in range(WritableHebbianFrame.num_samp)]
                    i = [random.randint(0, meta_stack_size[3] - 1) for _ in range(WritableHebbianFrame.num_samp)]
                    j = [random.randint(0, meta_stack_size[2] - 1) for _ in range(WritableHebbianFrame.num_samp)]

                    def label_tuples(t):
                        return {'v_i': float(t[0].data[0]),
                                'w_ij': float(t[1].data[0]),
                                'v_j': float(t[2].data[0]),
                                'grad': float(t[3].data[0])}

                    samples = [label_tuples(learner.impulse[layer_name][batch[x], :, j[x], i[x]])
                               for x in range(WritableHebbianFrame.num_samp)]
                    correct_columns = ['v_i', 'w_ij', 'v_j', 'grad']
                    metadata_df = pd.concat([metadata_df,
                                             pd.DataFrame(samples, columns=correct_columns)], axis=0)
                    if len(set(metadata_df.columns) - set(correct_columns)) > 0:
                        raise ValueError("metadata_df columns corrupted")
                    # print(metadata_df.count)
                    # del meta_stack_size, layer_grad
                    # del meta_stack_size, layer_grad, batch, i, j
                    del meta_stack_size, layer_grad, samples, batch, i, j
                    gc.collect()
                    if gpu_bool:
                        torch.cuda.empty_cache()

                learner_optimizer.step()
                # print(time.time() - tick)
                grad_of_param.clear()
                del images, labels, outputs, learner_loss, grad_of_param
                gc.collect()

        # gets number of files in directory
        idx = len([name for name in os.listdir(metalearner_directory)
                   if os.path.isfile(os.path.join(metalearner_directory, name))])

        torch.save(learner, metalearner_directory + '/' + str(idx) + '.model')
        metadata_df.to_csv(metadata_path)
        del learner


run = False
if run:
    writable_hebbian_frame = WritableHebbianFrame('hebbian', hebbian_fixed_params, hebbian_params_range,
                                                  hebbian_params_init)
    for i in range(100):
        writable_hebbian_frame.train_model(183, 43, 10, 0.001, 50, 0.001, 1, 15)
