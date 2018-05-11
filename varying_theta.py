import matplotlib.pyplot as plt
from vanilla import *
from fully_diff import *
from control import *
import pandas as pd

# vanilla_frame = Vanilla('vanilla', {}, {}, {})
fully_diff_frame = FullyDiff('fully_diff', {}, {}, {})
control_frame = Control('control', {}, {}, {})

frames = [vanilla_frame]

x = np.arange(0, 1.01, 0.1)

theta_df = pd.DataFrame(x, columns=['theta'])
# theta_df['vanilla'] = theta_df['theta']
theta_df['fully_diff'] = theta_df['theta']
theta_df['control'] = theta_df['theta']

for theta in np.nditer(x):
    # vanilla_acc = vanilla_frame.train_model(400, 200, 10, 3000, 0.001, 0.0001, 10, theta)  # tune
    # theta_df.loc[theta_df['theta'] == theta, 'vanilla'] = vanilla_acc

    # optimized fully_diff hyperparams
    # {'max_val': 0.9821928771508603,
    #  'max_params': {'mid1': 185.92848025411553, 'mid2': 255.30859815220873, 'meta_mid': 2.0,
    #                 'learning_rate': 0.00042919033808678115, 'update_rate': 1e-06,
    #                 'learner_batch_size': 17.424049877949205}}
    # mid1: 246.23106372530853
    # mid2: 146.42447917724752
    # meta_mid: 6.020968196157715
    # learning_rate: 0.0006669522749695926
    # update_rate: 0.00020694005810751248
    # learner_batch_size: 47.28394913558323
    fully_diff_acc = fully_diff_frame.train_model(246, 146, 6, 0.00066695, 0.00020694, 47, theta)
    theta_df.loc[theta_df['theta'] == theta, 'fully_diff'] = fully_diff_acc
    print("Finished:", theta, "-fully_diff-", fully_diff_acc)
    # optimized control hyperparams
    # {'max_val': 0.9817, 'max_params': {'mid1': 612.5576463625512, 'mid2': 746.756482014418, 'learning_rate': 0.001,
    #                                    'learner_batch_size': 309.07721048030334}}
    control_acc = control_frame.train_model(612, 746, 0.001, 309, theta)
    theta_df.loc[theta_df['theta'] == theta, 'control'] = control_acc
    print("Finished:", theta, "-control-", control_acc)

fig = plt.figure()

# plt.plot(theta_df['theta'], theta_df['vanilla'])
plt.plot(theta_df['theta'], theta_df['fully_diff'])
plt.plot(theta_df['theta'], theta_df['control'])
plt.legend(['vanilla', 'fully_diff', 'control', 'y = 4x'], loc='upper left')
plt.xlabel('Proportion of Labeled Examples', fontsize=18)
plt.ylabel('Accuracy', fontsize=16)
plt.show()
fig.savefig('theta.jpg')
