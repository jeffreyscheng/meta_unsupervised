import pandas as pd
import torch
from torch import nn
import os
import itertools
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# from experiment_0_util.experiment import *
from experiment_3_rule_form.create_meta_dataset import WritableHebbianFrame
from experiment_3_rule_form.create_meta_dataset import WritableHebbianNet
from torch.autograd import Variable

print("imports done")
final_path = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                          'final_data')
metalearner_directory = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                                     'temp_data',
                                     'metalearners')
temp_path = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                         'temp_data')
deg_appx_train_path = os.path.join(temp_path, 'degree_appx_train_sets')
if not os.path.exists(deg_appx_train_path):
    os.makedirs(deg_appx_train_path)

metadata_path = os.path.join(temp_path, 'metadata.csv')

metadata_df = pd.read_csv(metadata_path)
num_models = len([name for name in os.listdir(metalearner_directory)
                  if os.path.isfile(os.path.join(metalearner_directory, name))])


class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function

        out = self.linear(x)
        return out


# turns a combination selection of indices into the balls and urns representation (last element is unused)
def comb_to_sb(tup):
    return tup[0], tup[1] - tup[0] - 1, tup[2] - tup[1] - 1


# construct pandas dataset
# column format: (0, 3, 5)
# represents value of v_i^0 * w_ij^3 * v_j^5

for d in range(2, 5):  # d is max degree of polynomial
    all_combinations = list(itertools.combinations(range(d + 3), 3))
    all_deg_arr = [comb_to_sb(c) for c in all_combinations]
    print(all_deg_arr)

    deg_appx_train_df = metadata_df.copy()
    for arr in all_deg_arr:
        deg_appx_train_df[str(arr)] = deg_appx_train_df.apply(lambda row: (row['v_i'] ** arr[0]) *
                                                                          (row['w_ij'] ** arr[1]) *
                                                                          (row['v_j'] ** arr[2]), axis=1)
    del deg_appx_train_df['v_i']
    del deg_appx_train_df['v_j']
    del deg_appx_train_df['w_ij']
    del deg_appx_train_df['grad']

    # write degree-appx dataset to file
    d_train_df_path = os.path.join(deg_appx_train_path, str(d) + '.csv')
    deg_appx_train_df.to_csv(d_train_df_path)

    # for each metalearner, train the linear regression model on the degree appx
    # create a new df degree_i_df
    # use d_appx's columns of format (0, 3, 5)
    # value represents coefficient of v_i^0 * w_ij^3 * v_j^5
    # add a column for error
    # want to grab avg error for each table degree_i_df, plot against each other.
    # want to graph distribution of each column of each table degree_i_df
    deg_appx_model_df = pd.DataFrame(columns=deg_appx_train_df.columns + ['error'])
    list_of_regression_params = []
    for i in range(num_models):
        model = torch.load(metalearner_directory + os.sep + str(i) + '.model')
        model_update_ser = deg_appx_train_df.apply(lambda row: model.get_single_update((row['(1, 0, 0)'],
                                                                                        row['(0, 1, 0)'],
                                                                                        row['(0, 0, 1)'])), axis=1)

        print(model_update_ser)
        print(deg_appx_train_df)
        print(deg_appx_model_df)
        input_dim = len(deg_appx_train_df.columns)
        regression = LinearRegressionModel(input_dim, 1)

        criterion = nn.MSELoss()  # Mean Squared Loss
        l_rate = 0.01
        optimiser = torch.optim.SGD(regression.parameters(), lr=l_rate)  # Stochastic Gradient Descent

        epochs = 2000

        for epoch in range(epochs):
            epoch += 1
            # increase the number of epochs by 1 every time
            inputs = Variable(torch.Tensor(deg_appx_train_df.values))
            labels = Variable(torch.Tensor(model_update_ser.values))

            # clear grads as discussed in prev post
            optimiser.zero_grad()
            # forward to get predicted values
            outputs = regression.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # back props
            optimiser.step()  # update the parameters
            print('epoch {}, loss {}'.format(epoch, loss.data[0]))

        print(regression.parameters())
        raise ValueError("finished with first model")

        # list_of_regression_params.append(regression.parameters())
        ## fix regression.parameters()

    deg_appx_model_df.append(list_of_regression_params, ignore_index=True)



    # need two dfs for results
    # degree vs. avg error df (just write every degree_appx, you can back it out)
    # model coefficients
    # also should plot the grad vs the hebbian update
    # just write every degree_appx, can get both out.
