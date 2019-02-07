from experiment_3_rule_form.writable_frames import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

writable_hebbian_frame = WritableActivationsFrame('writable activations', fixed_parameters)
writable_hebbian_frame.train_model(theta=1, phi=5)

for i in range(100):
    run_theta_phi_pair_with_cache(theta_val=1, phi_val=10)
