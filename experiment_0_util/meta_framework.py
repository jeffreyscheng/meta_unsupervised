from bayes_opt import BayesianOptimization
import pickle
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn as nn
import time
import numpy as np
import math
from functools import wraps
import traceback


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


# Template for Bias and Weight Update Metalearner
# inputs in order v_i, w_ij, v_j
class MetaNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MetaNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out


class MetaFramework:
    time_out = 20 * 60
    num_data = 60000

    def __init__(self, name, fixed_params, variable_params_range, variable_params_init):
        self.name = name
        self.fixed_params = fixed_params
        self.variable_params_range = variable_params_range
        self.variable_params_init = variable_params_init

    def initialize_learner(self, **params):
        pass

    def train_model(self, **params):
        pass

    def optimize(self, n):
        file_name = self.name + '.bayes'
        bayes_file = Path(file_name)

        if not bayes_file.is_file():
            print("Initializing:", file_name)
            bayes = BayesianOptimization(self.train_model, self.variable_params_range)
            bayes.explore(self.variable_params_init)
            bayes.maximize(init_points=1, n_iter=n, kappa=1, acq="ucb")
        else:
            with open(file_name, 'rb') as bayes_file:
                bayes = pickle.load(bayes_file)
            print("Loaded file:", file_name)
            bayes.maximize(n_iter=n, kappa=1, acq="ucb", alpha=1e-3)

        print(bayes.res['max'])
        print(bayes.res['all'])
        with open(file_name, "wb") as output_file:
            pickle.dump(bayes, output_file)

    def analyze(self):
        pass


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


def bandaid(method):
    @wraps(method)
    def bounced(*args, **kw):
        tick = time.time()
        while True:
            try:
                if time.time() - tick > MetaFramework.time_out:
                    return 0
                result = method(*args, **kw)
                return result
            except (RuntimeError, MemoryError) as e:
                print("Encountered Error!")
                print("_____")
                traceback.print_exc()
                print("_____")
                # print(e.__dict__)
                # num = str(random.randint(0, 3))
                # # print("Bounced to machine " + num)
                # # os.environ["CUDA_VISIBLE_DEVICES"] = num
                # for obj in gc.get_objects():
                #     if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #         print(type(obj), obj.size())
                return 0

    return bounced
