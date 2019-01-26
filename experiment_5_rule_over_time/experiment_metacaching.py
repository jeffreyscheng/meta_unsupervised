from experiment_0_util.hebbian_frame import *
from hyperparameters import *
import os

temp_directory = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                              'temp_data')
metalearner_directory = os.path.join(temp_directory, 'metalearners_over_time')

if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)
if not os.path.exists(metalearner_directory):
    os.makedirs(metalearner_directory)


class CachedMetaLearner(nn.Module):

    def __init__(self, conv1, conv2, phi_val, theta_val):
        super(CachedMetaLearner, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = conv1
        self.conv2 = conv2
        self.phi = phi_val
        self.theta_val = theta_val

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


def run_theta_phi_pair_with_cache(theta_val, phi_val):
    print("Running Hebbian:", theta_val, phi_val)
    hebbian_model = hebbian_frame.train_model(hyperparameters['mid1'],
                                              hyperparameters['mid2'],
                                              hyperparameters['meta_mid'],
                                              hyperparameters['learning_rate'],
                                              hyperparameters['learner_batch_size'],
                                              hyperparameters['update_rate'],
                                              theta=theta_val, phi=phi_val,
                                              return_model=True)
    cached_metalearner = CachedMetaLearner(hebbian_model.conv1,
                                           hebbian_model.conv2,
                                           phi_val,
                                           theta_val)
    del hebbian_model
    idx = len([name for name in os.listdir(metalearner_directory)
               if os.path.isfile(os.path.join(metalearner_directory, name))])
    torch.save(cached_metalearner, metalearner_directory + '/' + str(idx) + '.model')
