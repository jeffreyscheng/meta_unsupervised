import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization

inc = 0.001
width = 10

acc_df = pd.read_csv('theta_phi_acc.csv')
acc_df['smooth_fully_diff'] = 0
acc_df['smooth_control'] = 0
for x in acc_df.index:
    if x > width:
        fully_diff_list = acc_df.loc[x - width:x, 'fully_diff']
        control_list = acc_df.loc[x - width:x, 'control']
        acc_df.loc[x, 'smooth_fully_diff'] = np.mean(fully_diff_list)
        acc_df.loc[x, 'smooth_control'] = np.mean(control_list)
    else:
        acc_df.loc[x, 'smooth_fully_diff'] = acc_df.loc[x, 'fully_diff']
        acc_df.loc[x, 'smooth_control'] = acc_df.loc[x, 'control']
print("Smoothed.")

fig = plt.figure()

smoothed = True

if smoothed:
    ax = fig.add_subplot(111, projection='3d')
    x = acc_df['theta']
    y = acc_df['phi']
    z_fd = acc_df['fully_diff']
    z_c = acc_df['control']

    ax.scatter(x, y, z_fd, color='red')
    ax.scatter(x, y, z_c, c='blue')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    plt.show()
else:
    # plt.plot(acc_df['theta'], acc_df['vanilla'])
    plt.scatter(acc_df['theta'], acc_df['smooth_fully_diff'], c='red')
    plt.scatter(acc_df['theta'], acc_df['smooth_control'], c='blue')
# plt.legend(['fully_diff', 'control', 'y = 4x'], loc='upper left')
# plt.xlabel('Proportion of Labeled Examples', fontsize=18)
# plt.xlabel('Proportion of Data Used', fontsize=18)
# plt.zlabel('Accuracy', fontsize=16)
plt.show()
fig.savefig('acc.png')
