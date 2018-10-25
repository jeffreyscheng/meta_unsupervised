import pandas as pd
import torch
import numpy as np
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
deg_appx_model_path = os.path.join(final_path, 'degree_appx_model_sets')
if not os.path.exists(deg_appx_model_path):
    os.makedirs(deg_appx_model_path)

metadata_path = os.path.join(temp_path, 'metadata.csv')

metadata_df = pd.read_csv(metadata_path)
num_models = len([name for name in os.listdir(metalearner_directory)
                  if os.path.isfile(os.path.join(metalearner_directory, name))])


class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
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

list_of_degree_dfs = []

for d in range(1, 10):  # d is max degree of polynomial
    print("starting degree " + str(d))
    all_combinations = list(itertools.combinations(range(d + 3), 3))
    all_deg_arr = [comb_to_sb(c) for c in all_combinations]
    str_all_deg_arr = [str(arr) for arr in all_deg_arr] + ['error']
    # print(str_all_deg_arr)

    def list_to_row_dict(lst):
        row_dict = {}
        if len(lst) != len(str_all_deg_arr):
            raise ValueError('regression tuple is not same length as column vector!')
        for idx in range(0, len(lst)):
            row_dict[str_all_deg_arr[idx]] = lst[idx]
        return row_dict


    deg_appx_train_df = metadata_df[['v_i', 'w_ij', 'v_j', 'grad']].copy()
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

    # print(type(deg_appx_train_df.columns))
    # print(deg_appx_train_df.columns + ['question?'])
    deg_appx_model_df = pd.DataFrame(columns=str_all_deg_arr)
    list_of_regression_params = []
    # num_models = 5 # for testing, remove
    for i in range(num_models):
        model = torch.load(metalearner_directory + os.sep + str(i) + '.model')
        model_update_ser = deg_appx_train_df.apply(lambda row: model.get_single_update((row['(1, 0, 0)'],
                                                                                        row['(0, 1, 0)'],
                                                                                        row['(0, 0, 1)'])), axis=1)
        # print(model_update_ser)
        # print(deg_appx_train_df)
        # print(deg_appx_model_df)
        input_dim = len(deg_appx_train_df.columns)
        regression = LinearRegressionModel(input_dim, 1)

        criterion = nn.MSELoss()  # Mean Squared Loss
        l_rate = 0.01
        optimiser = torch.optim.SGD(regression.parameters(), lr=l_rate)  # Stochastic Gradient Descent

        epochs = 20000
        error = 1
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
            error = float(loss.data)
            loss.backward()  # back props
            optimiser.step()  # update the parameters
            # print('epoch {}, loss {}'.format(epoch, loss.data[0]))
        # print("model " + str(i) + ", error " + str(error))
        # print(regression.parameters())
        # for name, param in regression.named_parameters():
            # if param.requires_grad:
                # print(name, param.data)
        regression_param = torch.squeeze(dict(regression.named_parameters())['linear.weight'].data)
        # print(regression_param)
        # print(regression_param.numpy())
        # print(tuple(regression_param.numpy()))
        # print(tuple(list(regression_param.numpy()) + [error]))
        list_of_regression_params.append(list_to_row_dict(list(regression_param.numpy()) + [error]))
        # raise ValueError("finished with first model")

        # list_of_regression_params.append(regression.parameters())
        ## fix regression.parameters()

    deg_appx_model_df = deg_appx_model_df.append(list_of_regression_params, ignore_index=True)
    # print(deg_appx_model_df)
    print(deg_appx_model_df['error'].mean())
    d_model_df_path = os.path.join(deg_appx_model_path, str(d) + '.csv')
    deg_appx_model_df.to_csv(d_model_df_path)
    list_of_degree_dfs.append(deg_appx_model_df)

# compute pointwise means
def pointwise_mean(tup):
    def output_i(i):
        model = torch.load(metalearner_directory + os.sep + str(i) + '.model')
        return model.get_single_update(tup)
    return sum([output_i(i) for i in range(num_models)]) / num_models

pointwise_mean_df = metadata_df.copy()
pointwise_mean_df['mean_Hebbian_update'] = pointwise_mean_df.apply(lambda row: pointwise_mean((row['v_i'], row['w_ij'], row['v_j'])), axis=1)
pointwise_path = os.path.join(final_path, 'pointwise_mean_df.csv')
pointwise_mean_df.to_csv(pointwise_path)



    # need two dfs for results
    # degree vs. avg error df (just write every degree_appx, you can back it out)
    # model coefficients
    # also should plot the grad vs the hebbian update
    # just write every degree_appx, can get both out.
