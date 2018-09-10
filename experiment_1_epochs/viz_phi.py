import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization
from matplotlib import pyplot
import os
#
fn = os.path.join(os.path.dirname(__file__), 'raw_phi_experiment.csv')
phi_df = pd.read_csv(fn)
print(len(phi_df.index))

results_path = os.path.join(os.sep.join(fn.split(os.sep)[:-2]), 'experiment_results')
print(results_path)

agg = []
for phi in set(phi_df['phi']):
    # if phi < 5.0:
    hebbian_sub = phi_df.loc[(phi_df['phi'] == phi) & (phi_df['bool_hebbian']), ]
    control_sub = phi_df.loc[(phi_df['phi'] == phi) & (~phi_df['bool_hebbian']), ]
    agg.append({'phi': phi, 'hebbian_acc': np.max(hebbian_sub['acc']), 'control_acc': np.max(control_sub['acc'])})

print(agg)
agg_df = pd.DataFrame(agg)
agg_df = agg_df.sort_values(by=['phi'], ascending=True)

plt.plot(agg_df['phi'], agg_df['hebbian_acc'])
plt.plot(agg_df['phi'], agg_df['control_acc'])
# plt.legend(['y = x', 'y = 2x', 'y = 3x', 'y = 4x'], loc='upper left')

plt.xlabel('Number of Epochs')
plt.ylabel('Maximum Accuracy')

plt.savefig(os.path.join(results_path, 'experiment_1_max.png'))
plt.clf()

agg = []
for phi in set(phi_df['phi']):
    # if phi < 5.0:
    hebbian_sub = phi_df.loc[(phi_df['phi'] == phi) & (phi_df['bool_hebbian']), ]
    control_sub = phi_df.loc[(phi_df['phi'] == phi) & (~phi_df['bool_hebbian']), ]
    agg.append({'phi': phi, 'hebbian_acc': np.median(hebbian_sub['acc']), 'control_acc': np.median(control_sub['acc'])})

print(agg)
agg_df = pd.DataFrame(agg)
agg_df = agg_df.sort_values(by=['phi'], ascending=True)

plt.plot(agg_df['phi'], agg_df['hebbian_acc'])
plt.plot(agg_df['phi'], agg_df['control_acc'])
# plt.legend(['y = x', 'y = 2x', 'y = 3x', 'y = 4x'], loc='upper left')

plt.xlabel('Number of Epochs')
plt.ylabel('Median Accuracy')

plt.savefig(os.path.join(results_path, 'experiment_1_median.png'))
plt.clf()

agg = []
for phi in set(phi_df['phi']):
    # if phi < 5.0:
    hebbian_sub = phi_df.loc[(phi_df['phi'] == phi) & (phi_df['bool_hebbian']), ]
    control_sub = phi_df.loc[(phi_df['phi'] == phi) & (~phi_df['bool_hebbian']), ]
    agg.append({'phi': phi, 'hebbian_acc': np.min(hebbian_sub['acc']), 'control_acc': np.min(control_sub['acc'])})

print(agg)
agg_df = pd.DataFrame(agg)
agg_df = agg_df.sort_values(by=['phi'], ascending=True)

plt.plot(agg_df['phi'], agg_df['hebbian_acc'])
plt.plot(agg_df['phi'], agg_df['control_acc'])
# plt.legend(['y = x', 'y = 2x', 'y = 3x', 'y = 4x'], loc='upper left')

plt.xlabel('Number of Epochs')
plt.ylabel('Minimum Accuracy')

plt.savefig(os.path.join(results_path, 'experiment_1_min.png'))

