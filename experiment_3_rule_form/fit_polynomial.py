import pandas as pd
import torch
import os

here = os.path.dirname(os.path.abspath(__file__))
metalearner_directory = here + '/metalearners'
metadata_path = here + os.sep + 'metadata.csv'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

metadata_df = pd.read_csv('metadata.csv')
num_models = len([name for name in os.listdir(metalearner_directory)
                  if os.path.isfile(os.path.join(metalearner_directory, name))])
for i in range(num_models):
    model = torch.load(metalearner_directory + os.sep + str(i) + '.model')
