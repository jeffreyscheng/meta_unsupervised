from bayes_opt import BayesianOptimization
import pickle
from single_single import *
from pathlib import Path


class MetaFramework:
    def __init__(self, name, fixed_params, variable_params_range, variable_params_init):
        self.name = name
        self.fixed_params = fixed_params
        self.variable_params_range = variable_params_range
        self.variable_params_init = variable_params_init

    def train_model(self, **params):
        pass

    def optimize(self, n):
        file_name = self.name + '.bayes'
        bayes_file = Path(file_name)

        if not bayes_file.is_file():
            print("Initializing:", file_name)
            # param_dict = {'mid1': (20, 800), 'mid2': (20, 800), 'meta_mid': (2, 10), 'meta_batch_size': (1, 10000),
            #               'learning_rate': (0.000001, 0.001), 'meta_rate': (0.000001, 0.001)}
            # bayes = BayesianOptimization(train_model, param_dict)
            bayes = BayesianOptimization(self.train_model, self.variable_params_range)

            # bayes.explore(
            #     {'mid1': [starting_learner_mid1], 'mid2': [starting_learner_mid2], 'meta_mid': [starting_meta_mid],
            #      'meta_batch_size': [starting_meta_batch_size], 'learning_rate': [starting_learning_rate],
            #      'meta_rate': [starting_meta_rate]})
            bayes.explore(self.variable_params_init)

            bayes.maximize(init_points=1, n_iter=n, kappa=1, acq="ucb")

        else:
            with open(file_name, 'rb') as bayes_file:
                bayes = pickle.load(bayes_file)
            print("Loaded file:", file_name)
            bayes.maximize(n_iter=n, kappa=1, acq="ucb")

        print(bayes.res['max'])
        print(bayes.res['all'])
        with open(file_name, "wb") as output_file:
            pickle.dump(bayes, output_file)

    def analyze(self):
        pass
