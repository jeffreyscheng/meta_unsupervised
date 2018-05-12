import matplotlib.pyplot as plt
import pandas as pd

theta_df = pd.read_csv('theta.csv')

fig = plt.figure()

# plt.plot(theta_df['theta'], theta_df['vanilla'])
plt.plot(theta_df['theta'], theta_df['fully_diff'])
plt.plot(theta_df['theta'], theta_df['control'])
plt.legend(['fully_diff', 'control', 'y = 4x'], loc='upper left')
plt.xlabel('Proportion of Labeled Examples', fontsize=18)
plt.ylabel('Accuracy', fontsize=16)
plt.show()
fig.savefig('theta.png')
