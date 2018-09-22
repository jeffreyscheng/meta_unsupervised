import pandas as pd
import torch
from torch import nn
import os
import itertools

here = os.path.dirname(os.path.abspath(__file__))
metalearner_directory = here + '/metalearners'
metadata_path = here + os.sep + 'metadata.csv'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

metadata_df = pd.read_csv('metadata.csv')
num_models = len([name for name in os.listdir(metalearner_directory)
                  if os.path.isfile(os.path.join(metalearner_directory, name))])


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.fc = nn.linear(1, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


model = LinearRegressionModel()


# turns a combination selection of indices into the balls and urns representation (last element is unused)
def comb_to_sb(tup):
    return tup[0], tup[1] - tup[0] - 1, tup[2] - tup[1] - 1


# construct pandas dataset

for d in range(2):  # d is max degree of polynomial
    all_combinations = list(itertools.combinations(range(d + 3), 3))
    all_deg_arr = [comb_to_sb(c) for c in all_combinations]
    print(all_deg_arr)

    d_appx_df = metadata_df.copy()
    for arr in all_deg_arr:
        d_appx_df[str(arr)] = d_appx_df.apply(lambda row: (row['v_i'] ** arr[0]) *
                                                          (row['w_ij'] ** arr[1]) *
                                                          (row['v_j'] ** arr[2]), axis=1)

    # write degree-appx dataset to file

    # train the linear regression model on the degree appx

    # write linear regression model to file

# for i in range(num_models):
#     model = torch.load(metalearner_directory + os.sep + str(i) + '.model')
