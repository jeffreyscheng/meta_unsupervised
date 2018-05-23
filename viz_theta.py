import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    z_fd = acc_df['smooth_fully_diff']
    z_c = acc_df['smooth_control']

    ax.scatter(x, y, z_fd)
    ax.scatter(x, y, z_c)

    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    plt.show()
else:
    # plt.plot(acc_df['theta'], acc_df['vanilla'])
    plt.plot(acc_df['theta'], acc_df['fully_diff'])
    plt.plot(acc_df['theta'], acc_df['control'])
plt.legend(['fully_diff', 'control', 'y = 4x'], loc='upper left')
plt.xlabel('Proportion of Labeled Examples', fontsize=18)
plt.xlabel('Proportion of Data Used', fontsize=18)
plt.ylabel('Accuracy', fontsize=16)
plt.show()
fig.savefig('acc.png')
