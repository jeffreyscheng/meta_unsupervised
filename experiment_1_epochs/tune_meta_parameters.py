from experiment_0_util.run_experiment import *
import pandas as pd
import os


def test_meta_parameters(update_ratio, mllr):


# run the control
# save dataframe to Fashion-MNIST/final_data
# schema: update_ratio:0, batch_num, loss, accuracy, ||Delta learner||, ||Delta ml||, ||Delta w_ij||

# for update_ratio in [100, 1000, 10000, 100000]
# for metalearner's learning rate in [1, 0.1, 0.01, 0.001, 0.0001]
# run, save to /final_data
# see them all together in Tensorboard
# plot update_ratio, mllr, max accuracy on curve as a heat map.
# if the best one works, we good!!!
# if there's nothing, give up.

# fix phi = whatever, theta = 0.6
# make a function for (update_ratio, mllr) pair.  And then FIX that pair