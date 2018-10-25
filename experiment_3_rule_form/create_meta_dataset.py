import os
from experiment_0_util.hebbian_frame import *
from experiment_3_rule_form.writable_hebbian import WritableHebbianNet, WritableHebbianFrame
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

run = False
if run:
    writable_hebbian_frame = WritableHebbianFrame('hebbian', hebbian_fixed_params, hebbian_params_range,
                                                  hebbian_params_init)
    for i in range(100):
        writable_hebbian_frame.train_model(183, 43, 10, 0.001, 50, 0.001, 1, 15)
