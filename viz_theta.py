import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

inc = 0.001
width = 5

theta_df = pd.read_csv('theta.csv')
theta_df['smooth_fully_diff'] = 0
theta_df['smooth_control'] = 0
for x in theta_df.index:
    if x > width:
        fully_diff_list = theta_df.loc[x - width:x, 'fully_diff']
        control_list = theta_df.loc[x - width:x, 'control']
        theta_df.loc[x, 'smooth_fully_diff'] = np.mean(fully_diff_list)
        theta_df.loc[x, 'smooth_control'] = np.mean(control_list)
    else:
        theta_df.loc[x, 'smooth_fully_diff'] = theta_df.loc[x, 'fully_diff']
        theta_df.loc[x, 'smooth_control'] = theta_df.loc[x, 'control']
print("Smoothed.")

fig = plt.figure()

smoothed = True

if smoothed:
    # plt.plot(theta_df['theta'], theta_df['vanilla'])
    plt.plot(theta_df['theta'], theta_df['smooth_fully_diff'])
    plt.plot(theta_df['theta'], theta_df['smooth_control'])
else:
    # plt.plot(theta_df['theta'], theta_df['vanilla'])
    plt.plot(theta_df['theta'], theta_df['fully_diff'])
    plt.plot(theta_df['theta'], theta_df['control'])
plt.legend(['fully_diff', 'control', 'y = 4x'], loc='upper left')
plt.xlabel('Proportion of Labeled Examples', fontsize=18)
plt.ylabel('Accuracy', fontsize=16)
plt.show()
fig.savefig('theta.png')
