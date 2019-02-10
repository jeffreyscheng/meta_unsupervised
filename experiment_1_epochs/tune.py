from experiment_0_util.run_experiment import *
import numpy as np

print("begin architecture")
for phi in range(1, 50):
    print(phi)
    control_acc = np.mean([control_frame.train_model(theta=1, phi=phi / 10) for _ in range(10)])
    print(control_acc)
