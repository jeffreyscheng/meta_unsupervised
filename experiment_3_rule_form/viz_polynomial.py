import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization
from matplotlib import pyplot
import os
#
fn = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                  'final_data',
                  'metadata.csv')
metadata_df = pd.read_csv(fn)
print(len(metadata_df.index))
print(metadata_df.columns)
print(metadata_df['v_i'])

# results_path = os.path.join(os.sep.join(fn.split(os.sep)[:-2]), 'experiment_results')
# print(results_path)
#
# plt.plot(metadata_df['grad'], metadata_df['hebbian_acc'])
# plt.plot(agg_df['phi'], agg_df['control_acc'])
# # plt.legend(['y = x', 'y = 2x', 'y = 3x', 'y = 4x'], loc='upper left')
#
# plt.xlabel('Number of Epochs')
# plt.ylabel('Minimum Accuracy')
#
# plt.savefig(os.path.join(results_path, 'experiment_1_min.png'))

