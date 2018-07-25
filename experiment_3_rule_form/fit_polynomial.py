from experiment_3_rule_form.create_meta_dataset import *

model = torch.load(metalearner_directory + '/0.model')
meta_input_stack = torch.Tensor([0, 0, 0])
meta_input_stack = meta_input_stack.unsqueeze(0)
meta_input_stack = meta_input_stack.unsqueeze(2)
meta_input_stack = meta_input_stack.unsqueeze(3)
print(model.get_update(meta_input_stack))

# for idx in range(0, 100):
#     model = torch.load(metalearner_directory + '/' + str(idx))
#     for v_i in range(-10, 10, 0.1):
#         for v_j in range(-10, 10, 0.1):
#             for w_ij in range(-10, 10, 0.1):
#                 meta_input_stack = torch.tensor([v_i, v_j, w_ij])
#                 model.get_update(meta_input_stack)
#
