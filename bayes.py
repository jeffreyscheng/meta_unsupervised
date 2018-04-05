from bayes_opt import BayesianOptimization
from single_single import *

param_dict = {'mid1': (20, 800), 'mid2': (20, 800), 'meta_mid': (2, 10), 'meta_batch_size': (1, 10000),
              'learning_rate': (0.000001, 0.001), 'meta_rate': (0.000001, 0.001)}
bayes = BayesianOptimization(train_model, param_dict)

bayes.explore({'mid1': [starting_learner_mid1], 'mid2': [starting_learner_mid2], 'meta_mid': [starting_meta_mid],
               'meta_batch_size': [starting_meta_batch_size], 'learning_rate': [starting_learning_rate],
               'meta_rate': [starting_meta_rate]})

bayes.maximize(init_points=5, n_iter=30, kappa=1, acq="ucb")

print(bayes.res['max'])
print(bayes.res['all'])
