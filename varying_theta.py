import matplotlib.pyplot as plt
from vanilla import *
from fully_diff import *
from control import *
import pandas as pd

vanilla_frame = Vanilla('vanilla', {}, {}, {})
fully_diff_frame = FullyDiff('fully_diff', {}, {}, {})
control_frame = Control('control', {}, {}, {})

frames = [vanilla_frame]

x = np.arange(0, 1.01, 0.1)

theta_df = pd.DataFrame(x, columns=['theta'])
theta_df['vanilla'] = theta_df['theta']
theta_df['fully_diff'] = theta_df['theta']
theta_df['control'] = theta_df['theta']

for theta in np.nditer(x):
    vanilla_acc = vanilla_frame.train_model(400, 200, 10, 3000, 0.001, 0.0001, 10, theta)  # tune
    theta_df.loc[theta_df['theta'] == theta, 'vanilla'] = vanilla_acc
    fully_diff_acc = fully_diff_frame.train_model(400, 200, 10, 3000, 0.001, 0.0001, 10, theta)  # tune
    theta_df.loc[theta_df['theta'] == theta, 'fully_diff'] = fully_diff_acc
    control_acc = control_frame.train_model(400, 200, 10, 3000, 0.001, 0.0001, 10, theta)  # tune
    theta_df.loc[theta_df['theta'] == theta, 'control'] = control_acc

plt.plot(theta_df['theta'], theta_df['vanilla'])
plt.plot(theta_df['theta'], theta_df['fully_diff'])
plt.plot(theta_df['theta'], theta_df['control'])
plt.legend(['vanilla', 'fully_diff', 'control', 'y = 4x'], loc='upper left')

plt.show()
