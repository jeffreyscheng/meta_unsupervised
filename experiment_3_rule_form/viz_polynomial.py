import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization
from matplotlib import pyplot
import os
# can't use Hebbian updates here
fn = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                  'temp_data',
                  'metadata.csv')
metadata_df = pd.read_csv(fn)
correct_columns = ['v_i', 'w_ij', 'v_j', 'grad']
for col in metadata_df.columns:
    if col not in correct_columns:
        del metadata_df[col]
print(len(metadata_df.index))
print(metadata_df.columns)
print(metadata_df['v_i'])

results_path = os.path.join(os.sep.join(fn.split(os.sep)[:-2]), 'experiment_results')
print(results_path)

# plt.plot(metadata_df['grad'], )

plt.xlabel('Gradient')
plt.ylabel('Hebbian Update')

plt.savefig(os.path.join(results_path, 'experiment_3_hebb_vs_grad.png'))

