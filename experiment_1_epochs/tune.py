from experiment_0_util.run_experiment import *
import numpy as np

print("begin architecture")
perf = []
for phi in range(0, 6):
    print(phi)
    control_acc = np.mean([control_frame.train_model(theta=1, phi=phi) for _ in range(10)])
    perf.append(control_acc)
print(perf)
