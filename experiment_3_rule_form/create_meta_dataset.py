from experiment_0_util.hebbian_frame import *
from experiment_3_rule_form.writable_hebbian import WritableHebbianNet, WritableHebbianFrame

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

run = False
if run:
    writable_hebbian_frame = WritableHebbianFrame('hebbian', fixed_parameters)
    for i in range(100):
        writable_hebbian_frame.train_model(hyperparameters['mid1'],
                                           hyperparameters['mid2'],
                                           hyperparameters['meta_mid'],
                                           hyperparameters['learning_rate'],
                                           hyperparameters['learner_batch_size'],
                                           hyperparameters['update_rate'])
