import pandas as pd

lol = pd.read_csv('raw_phi_experiment.csv')
print(lol)
lol.to_csv('lol.csv')