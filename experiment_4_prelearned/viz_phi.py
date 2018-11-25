import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization
from matplotlib import pyplot
import os

#
optimal_fn = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                          'final_data', 'optimal_phi_experiment.csv')
optimal_df = pd.read_csv(optimal_fn)
print(len(optimal_df.index))
control_fn = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                          'final_data', 'raw_phi_experiment.csv')
control_df = pd.read_csv(control_fn)
print(len(control_df.index))

results_path = os.path.join(os.sep.join(optimal_fn.split(os.sep)[:-2]), 'experiment_results')
print(results_path)

agg = []
for phi in set(optimal_df['phi']):
    # if phi < 5.0:
    optimal_sub = optimal_df.loc[(optimal_df['phi'] == phi) & (optimal_df['bool_optimal']), ]
    control_sub = control_df.loc[(control_df['phi'] == phi) & (~control_df['bool_hebbian']), ]
    agg.append({'phi': phi, 'optimal_acc': np.max(optimal_sub['acc']), 'control_acc': np.max(control_sub['acc'])})

print(agg)
agg_df = pd.DataFrame(agg)
agg_df = agg_df.sort_values(by=['phi'], ascending=True)

# plt.figure(figsize=(4, 3))
plt.plot(agg_df['phi'], agg_df['optimal_acc'])
plt.plot(agg_df['phi'], agg_df['control_acc'])
# plt.legend(['y = x', 'y = 2x', 'y = 3x', 'y = 4x'], loc='upper left')

plt.xlabel('Number of Epochs')
plt.ylabel('Maximum Accuracy')

plt.savefig(os.path.join(results_path, 'experiment_4_max.png'))
plt.clf()

agg = []
for phi in set(optimal_df['phi']):
    # if phi < 5.0:
    optimal_sub = optimal_df.loc[(optimal_df['phi'] == phi) & (optimal_df['bool_optimal']), ]
    control_sub = control_df.loc[(control_df['phi'] == phi) & (~control_df['bool_hebbian']), ]
    agg.append({'phi': phi, 'optimal_acc': np.median(optimal_sub['acc']), 'control_acc': np.median(control_sub['acc'])})

print(agg)
agg_df = pd.DataFrame(agg)
agg_df = agg_df.sort_values(by=['phi'], ascending=True)

plt.plot(agg_df['phi'], agg_df['optimal_acc'])
plt.plot(agg_df['phi'], agg_df['control_acc'])
# plt.legend(['y = x', 'y = 2x', 'y = 3x', 'y = 4x'], loc='upper left')

plt.xlabel('Number of Epochs')
plt.ylabel('Median Accuracy')
plt.title('Effect of HAT on Accuracy over Time')
# plt.ylim(0.6, 0.8)
plt.legend(['Hebbian', 'Control'])

plt.savefig(os.path.join(results_path, 'experiment_4_median.png'))
plt.savefig(os.path.join(results_path, 'experiment_4_median.eps'))
plt.clf()

agg = []
for phi in set(optimal_df['phi']):
    # if phi < 5.0:
    optimal_sub = optimal_df.loc[(optimal_df['phi'] == phi) & (optimal_df['bool_optimal']), ]
    control_sub = control_df.loc[(control_df['phi'] == phi) & (~control_df['bool_hebbian']), ]
    agg.append({'phi': phi, 'optimal_acc': np.min(optimal_sub['acc']), 'control_acc': np.min(control_sub['acc'])})

print(agg)
agg_df = pd.DataFrame(agg)
agg_df = agg_df.sort_values(by=['phi'], ascending=True)

plt.plot(agg_df['phi'], agg_df['optimal_acc'])
plt.plot(agg_df['phi'], agg_df['control_acc'])
# plt.legend(['y = x', 'y = 2x', 'y = 3x', 'y = 4x'], loc='upper left')

plt.xlabel('Number of Epochs')
plt.ylabel('Minimum Accuracy')

plt.savefig(os.path.join(results_path, 'experiment_4_min.png'))
