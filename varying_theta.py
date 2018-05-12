import matplotlib.pyplot as plt
# from vanilla import *
from fully_diff import *
from control import *
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MetaFramework.num_epochs = 1

frames = [fully_diff_frame]

x = np.arange(0, 1.01, 0.01)

theta_df = pd.DataFrame(x, columns=['theta'])
# theta_df['vanilla'] = theta_df['theta']
theta_df['fully_diff'] = theta_df['theta']
theta_df['control'] = theta_df['theta']

for theta in np.nditer(x):
    # vanilla_acc = vanilla_frame.train_model(400, 200, 10, 3000, 0.001, 0.0001, 10, theta)  # tune
    # theta_df.loc[theta_df['theta'] == theta, 'vanilla'] = vanilla_acc

    # optimized fully_diff hyperparams
    # mid1: 246.23106372530853
    # mid2: 146.42447917724752
    # meta_mid: 6.020968196157715
    # learning_rate: 0.0006669522749695926
    # update_rate: 0.00020694005810751248
    # learner_batch_size: 47.28394913558323
    fully_diff_acc = fully_diff_frame.train_model(246, 146, 6, 0.00066695, 47, 0.00020694, theta)
    theta_df.loc[theta_df['theta'] == theta, 'fully_diff'] = fully_diff_acc
    print("Finished:", theta, "-fully_diff-", fully_diff_acc)
    # optimized control hyperparams
    # mid1: 775.9687493498457
    # mid2: 252.21610495582954
    # learning_rate: 0.0007812704945456946
    # learner_batch_size: 216.94517685448233
    control_acc = control_frame.train_model(246, 146, 0.00066695, 47, theta)
    theta_df.loc[theta_df['theta'] == theta, 'control'] = control_acc
    print("Finished:", theta, "-control-", control_acc)

theta_df.to_csv('theta.csv')


