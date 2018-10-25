import pandas as pd
import matplotlib.pyplot as plt
import os

pointwise_path = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                              'final_data',
                              'pointwise_mean_df.csv')
model_path = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                          'final_data',
                          'degree_appx_model_sets')
results_path = os.path.join(os.sep.join(pointwise_path.split(os.sep)[:-2]), 'experiment_results')

hebbian_vs_grad_df = pd.read_csv(pointwise_path)
hebbian_vs_grad_df['mean_Hebbian_update'] = hebbian_vs_grad_df['mean_Hebbian_update'] * 0.05
correct_columns = ['v_i', 'w_ij', 'v_j', 'grad']

plt.scatter(hebbian_vs_grad_df['grad'], hebbian_vs_grad_df['mean_Hebbian_update'])
plt.xlabel('Gradient')
plt.ylabel('Hebbian Update')
plt.savefig(os.path.join(results_path, 'experiment_3_grad_vs_Hebb.png'))

grad_mean = hebbian_vs_grad_df['grad'].mean()
grad_std = hebbian_vs_grad_df['grad'].std()
hebbian_vs_grad_df['scaled_grad'] = (hebbian_vs_grad_df['grad'] - grad_mean) / grad_std
hebb_mean = hebbian_vs_grad_df['mean_Hebbian_update'].mean()
hebb_std = hebbian_vs_grad_df['mean_Hebbian_update'].std()
hebbian_vs_grad_df['scaled_hebb'] = (hebbian_vs_grad_df['mean_Hebbian_update'] - hebb_mean) / hebb_std
plt.clf()
plt.scatter(hebbian_vs_grad_df['scaled_grad'], hebbian_vs_grad_df['scaled_hebb'])
plt.xlabel('Scaled Gradient')
plt.ylabel('Scaled Hebbian Update')
plt.savefig(os.path.join(results_path, 'experiment_3_scaled_grad_vs_Hebb.png'))


def good_column(s):
    return s[0] == '(' or s == 'error'


deg_by_avg_error = []

for deg in range(0, 5):
    plt.clf()
    deg_i_df = pd.read_csv(os.path.join(model_path, str(deg) + '.csv'))
    good_columns = [col for col in list(deg_i_df.columns.values) if good_column(col)]
    deg_i_hist = deg_i_df.hist(column=good_columns)
    deg_by_avg_error.append({'degree': deg, 'error': deg_i_df['error'].mean()})
    plt.savefig(os.path.join(results_path, 'experiment_3_' + 'degree_' + str(deg) + '_models.png'))

plt.clf()
deg_by_avg_error_df = pd.DataFrame(deg_by_avg_error)
plt.plot(deg_by_avg_error_df['degree'], deg_by_avg_error_df['error'])
plt.xlabel('Degree')
plt.ylabel('Average MSE')
plt.savefig(os.path.join(results_path, 'experiment_3_degree_vs_error.png'))

plt.clf()
plt.plot(deg_by_avg_error_df.loc[1:, 'degree'], deg_by_avg_error_df.loc[1:, 'error'])
plt.xlabel('Degree')
plt.ylabel('Average MSE')
plt.savefig(os.path.join(results_path, 'experiment_3_degree_vs_error_without_0.png'))

deg_1_df = pd.read_csv(os.path.join(model_path, '1.csv'))
good_columns = [col for col in list(deg_1_df.columns.values) if good_column(col)]
for col in good_columns:
    print(col)
    print(deg_1_df[col].mean())
