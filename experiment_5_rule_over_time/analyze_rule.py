import torch
from torch.autograd import Variable
import os
from experiment_5_rule_over_time.experiment_metacaching import CachedMetaLearner

temp_directory = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                              'temp_data')
metalearner_directory = os.path.join(temp_directory, 'metalearners_over_time')

if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)
if not os.path.exists(metalearner_directory):
    os.makedirs(metalearner_directory)

num_models = len([name for name in os.listdir(metalearner_directory)
                  if os.path.isfile(os.path.join(metalearner_directory, name))])

for i in range(num_models):
    model = torch.load(metalearner_directory + os.sep + str(i) + '.model')
    print("loaded model")
    print(model(Variable(torch.Tensor([0, 0, 0]).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0))))
    del model
