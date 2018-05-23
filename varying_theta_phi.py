from fully_diff import *
from control import *
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MetaFramework.num_epochs = 1
# frames = [fully_diff_frame]
x = np.arange(0, 1.01, 0.2)
acc_df = pd.DataFrame(columns=['theta', 'phi', 'fully_diff', 'control'])
run_before = False


def run_theta_phi_pair(theta_val, phi_val):
    # optimized fully_diff hyperparams
    # mid1: 246.23106372530853
    # mid2: 146.42447917724752
    # meta_mid: 6.020968196157715
    # learning_rate: 0.0006669522749695926
    # update_rate: 0.00020694005810751248
    # learner_batch_size: 47.28394913558323
    def get_fully_diff():
        print("getting fully_diff")
        acc = [fully_diff_frame.train_model(246, 146, 6, 0.00066695, 47, 0.00020694, theta_val, phi_val) for _ in
               range(5)]
        return max(acc)

    # optimized control hyperparams
    # mid1: 775.9687493498457
    # mid2: 252.21610495582954
    # learning_rate: 0.0007812704945456946
    # learner_batch_size: 216.94517685448233
    def get_control():
        print("getting control")
        acc = [control_frame.train_model(246, 146, 0.00066695, 47, theta_val, phi_val) for _ in range(5)]
        return max(acc)

    fully_diff_acc = get_fully_diff()
    control_acc = get_control()
    acc_df.loc[len(acc_df.index)] = [theta_val, phi_val, fully_diff_acc, control_acc]
    print("Finished:", theta_val, phi_val, "-fully_diff-", fully_diff_acc, "-control-", control_acc)


if not run_before:
    for theta in np.nditer(x):
        for phi in np.nditer(x):
            run_theta_phi_pair(theta, phi)


else:
    acc_df = pd.read_csv('theta_phi_acc.csv')
    theta_phi_list = []


    def get_mid(ser, idx):
        return (ser[idx] + ser[idx - 1]) / 2


    for _ in range(10):
        sort_by_theta = acc_df.sort_values(by=['theta, phi'])
        jumps1 = sort_by_theta['fully_diff'] - sort_by_theta['fully_diff'].shift()
        idx1 = jumps1.idxmax()
        theta_phi_list.append([get_mid(sort_by_theta['theta'], idx1), get_mid(sort_by_theta['phi'], idx1)])

        jumps2 = sort_by_theta['control'] - sort_by_theta['control'].shift()
        idx2 = jumps2.idxmax()
        theta_phi_list.append([get_mid(sort_by_theta['theta'], idx2), get_mid(sort_by_theta['phi'], idx2)])

        sort_by_phi = acc_df.sort_values(by=['phi, theta'])
        jumps3 = sort_by_phi['fully_diff'] - sort_by_phi['fully_diff'].shift()
        idx3 = jumps3.idxmax()
        theta_phi_list.append([get_mid(sort_by_phi['theta'], idx3), get_mid(sort_by_phi['phi'], idx3)])
        jumps4 = sort_by_phi['control'] - sort_by_phi['control'].shift()
        idx4 = jumps4.idxmax()
        theta_phi_list.append([get_mid(sort_by_phi['theta'], idx4), get_mid(sort_by_phi['phi'], idx4)])

    for pair in theta_phi_list:
        theta = pair[0]
        phi = pair[1]

acc_df.to_csv('theta_phi_acc.csv')
