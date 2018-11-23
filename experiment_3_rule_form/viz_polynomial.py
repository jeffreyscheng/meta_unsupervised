import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

# acc by phi DONE
# acc by theta DONE
# poly appx
# vivj heatmap
# vi by hebb
# vj by hebb
# wij by hebb
# grad vs hebb


final_path = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                          'final_data')
pointwise_path = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                              'final_data',
                              'pointwise_mean_df.csv')
model_path = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]),
                          'final_data',
                          'degree_appx_model_sets')
results_path = os.path.join(os.sep.join(pointwise_path.split(os.sep)[:-2]), 'experiment_results')

hebbian_vs_grad_df = pd.read_csv(pointwise_path)
# hebbian_vs_grad_df = hebbian_vs_grad_df.loc[abs(hebbian_vs_grad_df['grad']) > 0.00001,]
hebbian_vs_grad_df['mean_Hebbian_update'] = hebbian_vs_grad_df['mean_Hebbian_update']
correct_columns = ['v_i', 'w_ij', 'v_j', 'grad']


plt.clf()
N = 100
hebbian_vs_grad_df = hebbian_vs_grad_df.sort_values(['grad'])
moving_avg_hebb = hebbian_vs_grad_df['mean_Hebbian_update'].rolling(50).mean()
print(len(moving_avg_hebb))
print(len(hebbian_vs_grad_df.loc[:, 'grad'].index))
plt.plot(hebbian_vs_grad_df.loc[:, 'grad'], moving_avg_hebb)
plt.xlim(-0.2, 0.2)
plt.xlabel('Gradient Update')
plt.ylabel('Smoothed Optimal-Hebb Update')
plt.savefig(os.path.join(results_path, 'experiment_3_grad_vs_Hebb.png'))
plt.savefig(os.path.join(results_path, 'experiment_3_grad_vs_Hebb.eps'))

plt.clf()


means, bin_edges, binnnumber = stats.binned_statistic(hebbian_vs_grad_df['v_i'], hebbian_vs_grad_df['mean_Hebbian_update'])
std, bin_edges, binnnumber = stats.binned_statistic(hebbian_vs_grad_df['v_i'], hebbian_vs_grad_df['mean_Hebbian_update'], 'std')
counts, bin_edges, binnnumber = stats.binned_statistic(hebbian_vs_grad_df['v_i'], hebbian_vs_grad_df['mean_Hebbian_update'], 'count')
plt.errorbar(bin_edges[:-1], means, yerr= std/np.sqrt(counts))
# plt.xlim(-0.05, 0.05)
plt.xlabel(r'Presynaptic Activity $v_i$')
plt.ylabel('Optimal-Hebb Update')
plt.title('Dependence on Presynaptic Activity')
plt.savefig(os.path.join(results_path, 'experiment_3_vi_vs_Hebb.png'))
plt.savefig(os.path.join(results_path, 'experiment_3_vi_vs_Hebb.eps'))

plt.clf()

means, bin_edges, binnnumber = stats.binned_statistic(hebbian_vs_grad_df['v_j'], hebbian_vs_grad_df['mean_Hebbian_update'])
std, bin_edges, binnnumber = stats.binned_statistic(hebbian_vs_grad_df['v_j'], hebbian_vs_grad_df['mean_Hebbian_update'], 'std')
counts, bin_edges, binnnumber = stats.binned_statistic(hebbian_vs_grad_df['v_j'], hebbian_vs_grad_df['mean_Hebbian_update'], 'count')
plt.errorbar(bin_edges[:-1], means, yerr= std/np.sqrt(counts))
# plt.xlim(-0.05, 0.05)
plt.xlabel(r'Postsynaptic Activity $v_j$')
plt.ylabel('Optimal-Hebb Update')
plt.title('Dependence on Postsynaptic Activity')
plt.savefig(os.path.join(results_path, 'experiment_3_vj_vs_Hebb.png'))
plt.savefig(os.path.join(results_path, 'experiment_3_vj_vs_Hebb.eps'))

#
# hebbian_vs_grad_df['vivj'] = hebbian_vs_grad_df['v_i'] * hebbian_vs_grad_df['v_j']
# plt.clf()
# means, bin_edges, binnnumber = stats.binned_statistic(hebbian_vs_grad_df['vivj'], hebbian_vs_grad_df['mean_Hebbian_update'])
# std, bin_edges, binnnumber = stats.binned_statistic(hebbian_vs_grad_df['vivj'], hebbian_vs_grad_df['mean_Hebbian_update'], 'std')
# counts, bin_edges, binnnumber = stats.binned_statistic(hebbian_vs_grad_df['vivj'], hebbian_vs_grad_df['mean_Hebbian_update'], 'count')
# plt.errorbar(bin_edges[:-1], means, yerr= std/np.sqrt(counts))
# # plt.xlim(-0.05, 0.05)
# plt.xlabel(r'$v_iv_j$')
# plt.ylabel('Optimal-Hebb Update')
# plt.title('Optimal-Hebb vs. True Hebb')
# plt.savefig(os.path.join(results_path, 'experiment_3_vivj_vs_Hebb.png'))
# plt.savefig(os.path.join(results_path, 'experiment_3_vivj_vs_Hebb.eps'))

plt.clf()

# .eps

means, bin_edges, binnnumber = stats.binned_statistic(hebbian_vs_grad_df['w_ij'], hebbian_vs_grad_df['mean_Hebbian_update'])
std, bin_edges, binnnumber = stats.binned_statistic(hebbian_vs_grad_df['w_ij'], hebbian_vs_grad_df['mean_Hebbian_update'], 'std')
counts, bin_edges, binnnumber = stats.binned_statistic(hebbian_vs_grad_df['w_ij'], hebbian_vs_grad_df['mean_Hebbian_update'], 'count')
plt.errorbar(bin_edges[:-1], means, yerr= std/np.sqrt(counts))
# plt.xlim(-0.05, 0.05)
plt.xlabel(r'Weight $w_{ij}$')
plt.ylabel('Optimal-Hebb Update')
plt.title('Dependence on the Synaptic Weight')
plt.savefig(os.path.join(results_path, 'experiment_3_wij_vs_Hebb.png'))
plt.savefig(os.path.join(results_path, 'experiment_3_wij_vs_Hebb.eps'))

plt.clf()
plt.scatter(hebbian_vs_grad_df['v_i'], hebbian_vs_grad_df['v_j'], c=hebbian_vs_grad_df['mean_Hebbian_update'], cmap="BuGn")
# plt.xlim(-0.05, 0.05)
plt.xlabel(r'Presynaptic Activity $v_i$')
plt.ylabel(r'Postsynaptic Activity $v_j$')
plt.legend(['Optimal-Hebb Update'])
plt.colorbar()
plt.savefig(os.path.join(results_path, 'experiment_3_vi_vj_Hebb.png'))
plt.savefig(os.path.join(results_path, 'experiment_3_vi_vj_Hebb.eps'))

plt.clf()
plt.tricontourf(hebbian_vs_grad_df['v_i'], hebbian_vs_grad_df['v_j'], hebbian_vs_grad_df['mean_Hebbian_update'], 20)
# plt.xlim(-0.05, 0.05)
plt.xlabel('Presynaptic Activity ' + r'$v_i$')
plt.ylabel(r'Postsynaptic Activity $v_j$')
plt.legend(['Optimal-Hebb Update'])
plt.colorbar()
plt.savefig(os.path.join(results_path, 'experiment_3_vi_vj_Hebb_c.png'))
plt.savefig(os.path.join(results_path, 'experiment_3_vi_vj_Hebb_c.eps'))

plt.clf()
plt.scatter(hebbian_vs_grad_df['w_ij'], hebbian_vs_grad_df['v_j'], c=hebbian_vs_grad_df['mean_Hebbian_update'], cmap="BuGn")
# plt.xlim(-0.05, 0.05)
plt.xlabel(r'$w_ij$')
plt.ylabel(r'$v_j$')
plt.legend()
plt.savefig(os.path.join(results_path, 'experiment_3_wij_vj_Hebb.png'))
plt.savefig(os.path.join(results_path, 'experiment_3_wij_vj_Hebb.eps'))

# F, p = stats.f_oneway(hebbian_vs_grad_df['v_i'], hebbian_vs_grad_df['v_j'], hebbian_vs_grad_df['w_ij'],
#                       hebbian_vs_grad_df['mean_Hebbian_update'])
# print(F)
# print(p)
# hebbian_vs_grad_df['vivj'] = hebbian_vs_grad_df['v_i'] * hebbian_vs_grad_df['v_j']
# a = hebbian_vs_grad_df.loc[:, ['v_i', 'v_j', 'w_ij', 'vivj','mean_Hebbian_update']].corr()
# print('v_i')
# print(a['v_i'])
#
# print('-------')
#
# print('v_j')
# print(a['v_j'])
#
# print('-------')
#
# print('w_ij')
# print(a['w_ij'])
#
# print('-------')
#
# print('mean_Hebbian_update')
# print(a['mean_Hebbian_update'])



def good_column(s):
    return s[0] == '(' or s == 'error'


deg_by_avg_error = []
deg_by_significance = []
list_of_deg_i_df = []

# create histograms for parameters
for deg in range(0, 10):
    deg_i_path = os.path.join(results_path, 'experiment_3_deg_' + str(deg))
    if not os.path.exists(deg_i_path):
        os.makedirs(deg_i_path)
    plt.clf()
    deg_i_df = pd.read_csv(os.path.join(model_path, str(deg) + '.csv'))
    good_columns = [col for col in list(deg_i_df.columns.values) if good_column(col)]
    deg_by_avg_error.append({'degree': deg, 'error': deg_i_df['error'].mean()})
    # num_significant = 0
    # for col in good_columns:
    #     deg_i_hist = deg_i_df.hist(column=[col])
    #     _, p_value = stats.ttest_1samp(deg_i_df.loc[:, col], 0)
    #     # print(p_value)
    #     if col != 'error':
    #         if p_value < 0.01:
    #             # print("Significant: degree " + str(deg) + ", col " + str(col))
    #             num_significant += 1
    #         # else:
    #         # print("Insignificant: degree " + str(deg) + ", col " + str(col))
    #     plt.savefig(os.path.join(deg_i_path, col + '_hist.png'))
    # deg_by_significance.append({'degree': deg, 'significance': num_significant / (len(good_columns) - 1)})
    # print("degree " + str(deg) + ", significant: " + str(num_significant) + " / " + str(len(good_columns) - 1))
    list_of_deg_i_df.append(deg_i_df)

# plot smooth histograms for
# heuristic_columns = ['(1, 0, 1)', '(2, 0, 0)']
# for col in heuristic_columns:
#     col_across_deg_df =

#
# plt.clf()
deg_by_avg_error_df = pd.DataFrame(deg_by_avg_error)
deg_by_avg_error_df.to_csv(os.sep.join([final_path, 'deg_by_avg_error_df.csv']))
print(deg_by_avg_error_df)
plt.plot(deg_by_avg_error_df['degree'], deg_by_avg_error_df['error'])
plt.xlabel('Degree')
plt.ylabel('Approximation Error')
plt.title('Optimal-Hebb is Complex')
plt.savefig(os.path.join(results_path, 'experiment_3_degree_vs_error.png'))
plt.savefig(os.path.join(results_path, 'experiment_3_degree_vs_error.eps'))

plt.clf()
plt.plot(deg_by_avg_error_df.loc[1:, 'degree'], deg_by_avg_error_df.loc[1:, 'error'])
plt.xlabel('Degree')
plt.ylabel('Average MSE')
plt.title('Optimal-Hebb is Complex')
plt.savefig(os.path.join(results_path, 'experiment_3_degree_vs_error_without_0.png'))
plt.savefig(os.path.join(results_path, 'experiment_3_degree_vs_error_without_0.eps'))

#
# plt.clf()
# deg_by_significance_df = pd.DataFrame(deg_by_significance)
# deg_by_significance_df.to_csv(os.sep.join([final_path, 'deg_by_significance_df.csv']))
# plt.plot(deg_by_significance_df['degree'], deg_by_significance_df['significance'])
# plt.xlabel('Degree')
# plt.ylabel('Significance Proportion')
# plt.savefig(os.path.join(results_path, 'experiment_3_degree_vs_significance.png'))
#
#
# def basic_augment_with_deg(df, i):
#     df['degree'] = i
#     return df.loc[:, ['degree', '(1, 0, 1)', '(2, 0, 0)']]
#
#
# for_tableau_coeff_list = [basic_augment_with_deg(list_of_deg_i_df[i], i) for i in range(2, 10)]
# for_tableau_coeff_df = pd.concat(for_tableau_coeff_list)
# for_tableau_coeff_df.to_csv(os.sep.join([final_path, 'tableau_coeff_df.csv']))
#
#
# def augment_with_deg(df, col, i):
#     df['deg'] = i
#     df = df.loc[:, [col]]
#     df.columns = ['Degree ' + str(i)]
#     return df
#
#
# list_of_hebb = [augment_with_deg(list_of_deg_i_df[i], '(1, 0, 1)', i) for i in range(2, 10)]
# list_of_oja = [augment_with_deg(list_of_deg_i_df[i], '(2, 0, 0)', i) for i in range(2, 10)]
# hebb_df = pd.concat(list_of_hebb, axis=1)
# oja_df = pd.concat(list_of_oja, axis=1)
# hebb_df.to_csv(os.sep.join([final_path, 'hebb_df.csv']))
# oja_df.to_csv(os.sep.join([final_path, 'hebb_df.csv']))
#
# hebb_bins = np.linspace(-1, 3, 100)
#
# # hebb
# plt.clf()
# for col in list(hebb_df):
#     plt.hist(hebb_df[col], None, alpha=0.5, label=col)
# plt.legend(loc='upper right')
# plt.xlabel('Coefficient of ' + r'$v_i\cdot v_j$')
# plt.ylabel('Frequency')
# plt.savefig(os.path.join(results_path, 'experiment_3_hebb_coefficient.png'))
#
# oja_bins = np.linspace(-1, 1, 100)
#
# # oja
# plt.clf()
# for col in list(oja_df):
#     plt.hist(oja_df[col], oja_bins, alpha=0.5, label=col)
# plt.legend(loc='upper right')
# plt.xlabel('Coefficient of ' + r'$v_i^2$')
# plt.ylabel('Frequency')
# plt.savefig(os.path.join(results_path, 'experiment_3_oja_coefficient.png'))
