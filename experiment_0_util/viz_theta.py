import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization

inc = 0.05
width = 10

acc_df = pd.read_csv('theta_phi_acc.csv')
acc_df['smooth_fully_diff'] = 0
acc_df['smooth_control'] = 0

smooth_df = pd.DataFrame(columns=['theta', 'phi', 'fully_diff', 'control'])
for theta in np.arange(0, 1.01, 0.1):
    for phi in np.arange(0, 1.01, 0.1):
        print(theta, phi)
        area_c = acc_df[abs((acc_df['theta'] - theta) < inc) & (abs(acc_df['phi'] - phi) < inc)]['control']
        area_fd = acc_df[abs((acc_df['theta'] - theta) < inc) & (abs(acc_df['phi'] - phi) < inc)]['fully_diff']
        smooth_df = smooth_df.append({'theta': theta, 'phi': phi, 'fully_diff': area_fd.max(),
                                      'control': area_c.max()}, ignore_index=True)

    # if x > width:
    #     fully_diff_list = acc_df.loc[x - width:x, 'fully_diff']
    #     control_list = acc_df.loc[x - width:x, 'control']
    #     acc_df.loc[x, 'smooth_fully_diff'] = np.mean(fully_diff_list)
    #     acc_df.loc[x, 'smooth_control'] = np.mean(control_list)
    # else:
    #     acc_df.loc[x, 'smooth_fully_diff'] = acc_df.loc[x, 'fully_diff']
    #     acc_df.loc[x, 'smooth_control'] = acc_df.loc[x, 'control']

print("Smoothed.")

fig = plt.figure()

smoothed = True

if not smoothed:
    ax = fig.add_subplot(111, projection='3d')
    x = acc_df['theta']
    y = acc_df['phi']
    z_fd = acc_df['fully_diff']
    z_c = acc_df['control']

    ax.scatter(x, y, z_fd, color='blue')
    ax.scatter(x, y, z_c, c='red')

    ax.plot(x, z_fd, 'r+', zdir='y', zs=1)
    ax.plot(y, z_fd, 'g+', zdir='x', zs=0)
    # ax.plot(x, y, 'k+', zdir='z', zs=-0)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    ax.set_xlabel('theta', fontsize=20)
    ax.set_ylabel('phi', fontsize=20)
    ax.set_zlabel('accuracy', fontsize=20)

    plt.show()
else:
    ax = fig.add_subplot(111, projection='3d')
    x = smooth_df['theta']
    y = smooth_df['phi']
    z_fd = smooth_df['fully_diff']
    z_c = smooth_df['control']

    ax.scatter(x, y, z_fd, color='blue')
    ax.scatter(x, y, z_c, c='red')

    ax.plot(x, z_fd, 'r+', zdir='y', zs=1)
    ax.plot(y, z_fd, 'g+', zdir='x', zs=0)
    # ax.plot(x, y, 'k+', zdir='z', zs=-0)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    ax.set_xlabel('theta', fontsize=20)
    ax.set_ylabel('phi', fontsize=20)
    ax.set_zlabel('accuracy', fontsize=20)

    plt.show()
# plt.legend(['fully_diff', 'control', 'y = 4x'], loc='upper left')
# plt.xlabel('Proportion of Labeled Examples', fontsize=18)
# plt.xlabel('Proportion of Data Used', fontsize=18)
# plt.zlabel('Accuracy', fontsize=16)
plt.show()
fig.savefig('acc-3D.png')
smooth_df.to_csv('smooth.csv')

theta_df = pd.DataFrame(columns=['theta', 'fully_diff', 'control'])
phi_df = pd.DataFrame(columns=['phi', 'fully_diff', 'control'])
for theta in np.arange(0, 1.01, 0.1):
    theta_df = theta_df.append({'theta': theta,
                                'fully_diff': smooth_df[abs(smooth_df['theta'] - theta) < inc]['fully_diff'].mean(),
                                'control': smooth_df[abs(smooth_df['theta'] - theta) < inc]['control'].mean()}, ignore_index=True)
for phi in np.arange(0, 1.01, 0.1):
    phi_df = phi_df.append({'phi': phi,
                            'fully_diff': smooth_df[abs(smooth_df['phi'] - phi) < inc]['fully_diff'].mean(),
                            'control': smooth_df[abs(smooth_df['phi'] - phi) < inc]['control'].mean()}, ignore_index=True)

fig = plt.figure()
plt.scatter(theta_df['theta'], theta_df['fully_diff'], color='blue')
plt.scatter(theta_df['theta'], theta_df['control'], color='red')
plt.show()
fig.savefig('acc-theta.png')
fig = plt.figure()
plt.scatter(phi_df['phi'], phi_df['fully_diff'], color='blue')
plt.scatter(phi_df['phi'], phi_df['control'], color='red')
plt.show()
fig.savefig('acc-phi.png')
