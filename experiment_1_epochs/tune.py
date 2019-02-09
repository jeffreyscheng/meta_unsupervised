from experiment_0_util.run_experiment import *

print("begin architecture")
for phi in range(10):
    print(phi)
    control_acc = control_frame.train_model(theta=1, phi=phi)
    print(control_acc)
