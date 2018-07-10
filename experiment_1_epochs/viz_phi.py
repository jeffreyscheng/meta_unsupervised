import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization
from matplotlib import pyplot
import os

# import seaborn
# seaborn.set(style='ticks')
#
fn = os.path.join(os.path.dirname(__file__), 'raw_phi_experiment.csv')
phi_df = pd.read_csv(fn)
print(len(phi_df.index))
# fg = seaborn.FacetGrid(data=phi_df, hue='bool_hebbian', aspect=1.61)
# fg.map(pyplot.scatter, 'phi', 'acc').add_legend()
# plt.show()

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

plt.show()
#
#
# smoothed = True
#
# if not smoothed:
#     ax = fig.add_subplot(111, projection='3d')
#     x = acc_df['theta']
#     y = acc_df['phi']
#     z_fd = acc_df['fully_diff']
#     z_c = acc_df['control']
#
#     ax.scatter(x, y, z_fd, color='blue')
#     ax.scatter(x, y, z_c, c='red')
#
#     ax.plot(x, z_fd, 'r+', zdir='y', zs=1)
#     ax.plot(y, z_fd, 'g+', zdir='x', zs=0)
#     # ax.plot(x, y, 'k+', zdir='z', zs=-0)
#
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#     ax.set_zlim([0, 1])
#
#     ax.set_xlabel('theta', fontsize=20)
#     ax.set_ylabel('phi', fontsize=20)
#     ax.set_zlabel('accuracy', fontsize=20)
#
#     plt.show()
# else:
#     ax = fig.add_subplot(111, projection='3d')
#     x = smooth_df['theta']
#     y = smooth_df['phi']
#     z_fd = smooth_df['fully_diff']
#     z_c = smooth_df['control']
#
#     ax.scatter(x, y, z_fd, color='blue')
#     ax.scatter(x, y, z_c, c='red')
#
#     ax.plot(x, z_fd, 'r+', zdir='y', zs=1)
#     ax.plot(y, z_fd, 'g+', zdir='x', zs=0)
#     # ax.plot(x, y, 'k+', zdir='z', zs=-0)
#
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#     ax.set_zlim([0, 1])
#
#     ax.set_xlabel('theta', fontsize=20)
#     ax.set_ylabel('phi', fontsize=20)
#     ax.set_zlabel('accuracy', fontsize=20)
#
#     plt.show()
# # plt.legend(['fully_diff', 'control', 'y = 4x'], loc='upper left')
# # plt.xlabel('Proportion of Labeled Examples', fontsize=18)
# # plt.xlabel('Proportion of Data Used', fontsize=18)
# # plt.zlabel('Accuracy', fontsize=16)
# plt.show()
# fig.savefig('acc-3D.png')
# smooth_df.to_csv('smooth.csv')
#
# theta_df = pd.DataFrame(columns=['theta', 'fully_diff', 'control'])
# phi_df = pd.DataFrame(columns=['phi', 'fully_diff', 'control'])
# for theta in np.arange(0, 1.01, 0.1):
#     theta_df = theta_df.append({'theta': theta,
#                                 'fully_diff': smooth_df[abs(smooth_df['theta'] - theta) < inc]['fully_diff'].mean(),
#                                 'control': smooth_df[abs(smooth_df['theta'] - theta) < inc]['control'].mean()}, ignore_index=True)
# for phi in np.arange(0, 1.01, 0.1):
#     phi_df = phi_df.append({'phi': phi,
#                             'fully_diff': smooth_df[abs(smooth_df['phi'] - phi) < inc]['fully_diff'].mean(),
#                             'control': smooth_df[abs(smooth_df['phi'] - phi) < inc]['control'].mean()}, ignore_index=True)
#
# fig = plt.figure()
# plt.scatter(theta_df['theta'], theta_df['fully_diff'], color='blue')
# plt.scatter(theta_df['theta'], theta_df['control'], color='red')
# plt.show()
# fig.savefig('acc-theta.png')
# fig = plt.figure()
# plt.scatter(phi_df['phi'], phi_df['fully_diff'], color='blue')
# plt.scatter(phi_df['phi'], phi_df['control'], color='red')
# plt.show()
# fig.savefig('acc-phi.png')
