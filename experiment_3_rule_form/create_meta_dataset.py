from experiment_0_util.hebbian_frame import *
import os

here = os.path.dirname(os.path.abspath(__file__))
metalearner_directory = here + '/metalearners'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class WritableHebbianFrame(MetaFramework):
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
        learner = HebbianNet(input_size, mid1, mid2, num_classes, meta_input, meta_mid, meta_output, learner_batch_size,
                             update_rate)
        # print(learner_batch_size)

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

            # most stuff before here

            # Learner Forward + Backward + Optimize
            learner_optimizer.zero_grad()  # zero the gradient buffer
            outputs = learner.forward(images, batch_num)
            if random.uniform(0, 1) < theta:
                learner_loss = learner_criterion(outputs, labels)
                # print(labels.data[0], ',', str(learner_loss.data[0]))
                learner_loss.backward()
                learner_optimizer.step()
                del images, labels, outputs, learner_loss

        # gets number of files in directory
        idx = len([name for name in os.listdir(metalearner_directory)
                   if os.path.isfile(os.path.join(metalearner_directory, name))])

        torch.save(learner, metalearner_directory + '/' + str(idx) + '.model')
        del learner


# writable_hebbian_frame = WritableHebbianFrame('hebbian', hebbian_fixed_params, hebbian_params_range,
#                                               hebbian_params_init)
# for i in range(100):
#     writable_hebbian_frame.train_model(183, 43, 10, 0.001, 50, 0.001, 1, 15)
