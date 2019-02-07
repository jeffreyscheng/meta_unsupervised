from experiment_0_util.hebbian_frame import *
from experiment_0_util.meta_framework import *
import random
import torch
import os
import pandas as pd
import gc

metadata_path = os.path.join(temp_data_path, 'metadata.csv')
metalearner_path = os.path.join(temp_data_path, 'metalearners')
safe_mkdir(metalearner_path)


class WritableActivationsNet(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(WritableActivationsNet, self).__init__()
        self.impulse = {}
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    # get new weight
    def get_update(self, meta_input_stack):
        out = self.conv1(meta_input_stack)
        out = self.conv2(out)
        out = torch.squeeze(out, 1)
        return out

    def forward(self, x, batch_num=1):
        if self.impulse is not None:
            if len(self.impulse) > 4:
                raise ValueError("long impulse!")
        self.impulse.clear()
        del self.impulse
        gc.collect()
        self.impulse = {}
        out = x
        for layer_num in range(0, 3):
            layer = self.param_state[self.param_names[layer_num]]
            vi = out
            old_vj = self.layers[layer_num](out)
            old_vj = self.relu(old_vj)
            stack_dim = self.batch_size, layer.size()[0], layer.size()[1]
            try:
                input_stack = vi.unsqueeze(1).expand(stack_dim)
                output_stack = old_vj.unsqueeze(2).expand(stack_dim)
                weight_stack = layer.unsqueeze(0).expand(stack_dim)
            except RuntimeError:
                print(self.batch_size)
                print(stack_dim)
                print(vi.size())
                print(old_vj.size())
                print(layer.size())
                input_stack = vi.unsqueeze(1).expand(stack_dim)
                output_stack = old_vj.unsqueeze(2).expand(stack_dim)
                weight_stack = layer.unsqueeze(0).expand(stack_dim)
                for obj in gc.get_objects():
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
            meta_inputs = torch.stack((input_stack, weight_stack, output_stack), dim=3).permute(0, 3, 1, 2)
            self.impulse[self.param_names[layer_num]] = meta_inputs
            out = old_vj
            del old_vj, input_stack, output_stack, weight_stack, meta_inputs
        if random.uniform(0, 1) < 0.01:  # only sample 1% of the time... otherwise the dset blows up to 2GB
            if os.path.isfile(metadata_path) and False:
                metadata_df = pd.read_csv(metadata_path)
            else:
                metadata_df = pd.DataFrame(columns=['v_i', 'w_ij', 'v_j', 'grad'])
            grad_of_param = {}
            for name, parameter in self.named_parameters():
                grad_of_param[name] = parameter.grad

            # pushes gradients into the metalearner stack
            for layer_name in self.impulse:
                meta_stack_size = list(self.impulse[layer_name].size())
                meta_stack_size[1] = 1
                layer_grad = grad_of_param[layer_name].unsqueeze(0).unsqueeze(1).expand(meta_stack_size)
                self.impulse[layer_name] = torch.cat((self.impulse[layer_name], layer_grad), dim=1)

                # samples for metadata_df
                num_samp = 100
                batch = [random.randint(0, meta_stack_size[0] - 1) for _ in range(num_samp)]
                i = [random.randint(0, meta_stack_size[3] - 1) for _ in range(num_samp)]
                j = [random.randint(0, meta_stack_size[2] - 1) for _ in range(num_samp)]

                def label_tuples(t):
                    return {'v_i': float(t[0].data[0]),
                            'w_ij': float(t[1].data[0]),
                            'v_j': float(t[2].data[0]),
                            'grad': float(t[3].data[0])}

                samples = [label_tuples(self.impulse[layer_name][batch[x], :, j[x], i[x]]) for x in range(num_samp)]
                correct_columns = ['v_i', 'w_ij', 'v_j', 'grad']
                metadata_df = pd.concat([metadata_df,
                                         pd.DataFrame(samples, columns=correct_columns)], axis=0)
                if len(set(metadata_df.columns) - set(correct_columns)) > 0:
                    raise ValueError("metadata_df columns corrupted")
                del meta_stack_size, layer_grad, samples, batch, i, j
                gc.collect()
                if gpu_bool:
                    torch.cuda.empty_cache()

            grad_of_param.clear()
            del grad_of_param
            gc.collect()
            metadata_df.to_csv(metadata_path)
        return out


class WritableActivationsFrame(MetaFramework):

    def __init__(self, name, fixed_params):
        super(WritableActivationsFrame, self).__init__(name, fixed_params)

    def create_learner_and_optimizer(self):
        learner = WritableActivationsNet(fixed_parameters['input_size'],
                                         hyperparameters['mid1'],
                                         hyperparameters['mid2'],
                                         fixed_parameters['num_classes'])
        optimizer = base_optimizer(list(learner.parameters()), lr=hyperparameters['learning_rate'])
        return learner, optimizer


class CachedMetaLearner(nn.Module):

    def __init__(self, conv1, conv2, phi_val, theta_val):
        super(CachedMetaLearner, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = conv1
        self.conv2 = conv2
        self.phi = phi_val
        self.theta_val = theta_val

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


def run_theta_phi_pair_with_cache(theta_val, phi_val):
    print("Running Hebbian:", theta_val, phi_val)
    hebbian_model = hebbian_frame.train_model(hyperparameters['mid1'],
                                              hyperparameters['mid2'],
                                              hyperparameters['meta_mid'],
                                              hyperparameters['learning_rate'],
                                              hyperparameters['learner_batch_size'],
                                              hyperparameters['update_rate'],
                                              theta=theta_val, phi=phi_val,
                                              return_model=True)
    cached_metalearner = CachedMetaLearner(hebbian_model.conv1,
                                           hebbian_model.conv2,
                                           phi_val,
                                           theta_val)
    del hebbian_model
    idx = len([name for name in os.listdir(metalearner_path)
               if os.path.isfile(os.path.join(metalearner_path, name))])
    torch.save(cached_metalearner, metalearner_path + '/' + str(idx) + '.model')
