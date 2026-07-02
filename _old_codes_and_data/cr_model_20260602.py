from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import fsolve

ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5', '4.2.batch6']
batch_colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
batch_labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
A_add_list = [0.0, 0.7, 1.4, 2.0]
A_add_markers = ['o', 's', '^', 'D']

data = {}
for id in ids:
    data[id] = {}
    data[id]['no3'] = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0)
    data[id]['no2'] = pd.read_csv(f'concentrations/{id}_no2_cons.csv', index_col=0)

def load_data_to_plot(ids, data, no3_or_no2, chl):
    data_to_plot = {}
    for id in ids:
        df = data[id][no3_or_no2]
        df = df[(df['Chloramphenicol'] == chl) & (df['Sample_type'] != 'Blank')]
        data_to_plot[id] = df
    return data_to_plot

data_to_plot = load_data_to_plot(ids, data, 'no2', chl=1)

def mean_bootstrap_ci(data, n_bootstrap, ci):
    # data has to be a 1D array
    if len(data) == 0:
        print("Warning: Empty data array provided to mean_bootstrap_ci. Returning NaN for mean and confidence interval.")
        return np.nan, np.nan, np.nan
    
    if len(data) == 1:
        return data[0], data[0], data[0]
    else:
        rng = np.random.default_rng()
        bootstrap_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
        mean = np.mean(bootstrap_means)
        lower_bound = np.percentile(bootstrap_means, (100 - ci) / 2)
        upper_bound = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)

        return mean, lower_bound, upper_bound

# Initial slopes
linear_fit_results = {}
for id in ids:
    df = data_to_plot[id]
    time = df.columns[:-4].astype(float)
    linear_fit_results[id] = pd.DataFrame(index=df.index, columns=['Initial_slope', 'Initial_intercept'])
    fit_indices = range(6)
    for well in df.index:
        y = df.loc[well].iloc[:-4].astype(float)
        slope, intercept = np.polyfit(time[fit_indices], y.iloc[fit_indices], 1)
        linear_fit_results[id].loc[well, 'Initial_slope'] = slope
        linear_fit_results[id].loc[well, 'Initial_intercept'] = intercept

# Area under the normalised cumulative consumption curve
nccc_stat = {}
for id in ids:
    df = data_to_plot[id]
    time = df.columns[:-4].astype(float)
    nccc_stat[id] = pd.DataFrame(index=df.index, columns=['AUC_norm_cons', 't_last', 'death_rate', 'curve_hits_zero'])
    for well in df.index:
        y = df.loc[well].iloc[:-4].astype(float)
        thrsh_end = 0.05
        zero_indices = np.where(y < thrsh_end)[0]
        if len(zero_indices) > 0:
            last_index = zero_indices[0]
            nccc_stat[id].loc[well, 'curve_hits_zero'] = True
        else:
            last_index = len(y) - 1
            nccc_stat[id].loc[well, 'curve_hits_zero'] = False
        y = y.iloc[:last_index + 1]
        time_norm = time[:last_index + 1] / time[last_index]
        y_norm = (y.iloc[0] - y) / (y.iloc[0] - y.iloc[-1])
        auc = np.trapezoid(y_norm, time_norm)
        def equation(x):
            if x == 0:
                return 0.5 - auc  # The mathematical limit as x approaches 0
            return 1 / (1 - np.exp(-x)) - (1 / x) - auc
        # Provide an initial guess, for example, 1.0
        x_guess = 1.0
        x_solution = fsolve(equation, x_guess)[0]

        nccc_stat[id].loc[well, 'AUC_norm_cons'] = auc
        nccc_stat[id].loc[well, 't_last'] = time[last_index]
        nccc_stat[id].loc[well, 'death_rate'] = x_solution / time[last_index] if time[last_index] > 0 else np.nan

# Figure 1
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(r'Raw $NO_3^-$ concentration $A(t)$ - drug condition' + '\n' + r'one curve per replicate | colour = batch')
subplot_titles = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
I_add_colors = ['blue', 'green', 'red', 'cyan']
I_add_list = [0.0, 0.7, 1.4, 2.0]
for i in range(6):
    ax = axes[i // 3, i % 3]
    ax.set_title(subplot_titles[i])
    if i // 3 == 1:
        ax.set_xlabel('Time (hr)')
    if i % 3 == 0:
        ax.set_ylabel(r'$NO_3^-$ (mM)')

    df = data_to_plot[ids[i]]        # time points
    for I_add, color in zip(I_add_list, I_add_colors):
        df_I = df[df['Nitrite_input'] == I_add].iloc[:, :-4]
        for well in df_I.index:
            time = df_I.columns.astype(float)
            y = df_I.loc[well].values.astype(float)
            ax.plot(time, y, color=color, linestyle='-', marker='.', alpha=0.5, label=f'$I_{{add}}$={I_add}' if well == df_I.index[0] else None)
    if i == 0:
        ax.legend(loc='upper right')
plt.savefig(f'plots/figure1_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
plt.show()


# # Figure 2
# no3_or_no2 = 'no3'
# chl = 1
# data_to_plot = {}
# for id in ids:
#     df = data[id][no3_or_no2]
#     df = df[(df['Chloramphenicol'] == chl) & (df['Sample_type'] != 'Blank')]
#     data_to_plot[id] = df
# y_min = min(df.iloc[:, :-4].min().min() for df in data_to_plot.values())
# y_max = max(df.iloc[:, :-4].max().max() for df in data_to_plot.values())

# fig, axes = plt.subplots(4, 4, figsize=(20, 20))
# fig.suptitle(r'Raw $NO_3^-$ concentration by condition ($A_{add}$, $I_{add}$) - colour = batch'
#                 + '\n' + r'rows = nominal $I_{add}$ | columns = nominal $A_{add}$'
#                 + '\n' + r'shared y-axis | thick = condition mean | thin = replicates')
# colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
# legends = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
# no3_add_values = [0.0, 0.7, 1.4, 2.0]
# no2_add_values = [0.0, 0.7, 1.4, 2.0]

# for i in range(4):
#     for j in range(4):
#         ax = axes[i, j]
#         ax.set_xlabel('Time (hr)') if i == 3 else None
#         ax.set_ylabel(r'$NO_3^-$ (mM)') if j == 0 else None
#         ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
#         no3_add = no3_add_values[j]
#         no2_add = no2_add_values[i]
#         if i == 0:
#             ax.set_title(f'$A_{{add}}$ = {no3_add}', fontsize=10)
#         if j == 0:
#             ax.text(-0.35, 0.5, f'$I_{{add}}$ = {no2_add}', 
#                    transform=ax.transAxes, fontsize=10, 
#                    rotation=90, va='center', ha='center')

#         for id, color, legend in zip(ids, colors, legends):
#             df = data_to_plot[id]
#             df = df.iloc[:, :-4][(df['Sample_type'] != 'Blank') & (df['Nitrate_input'] == no3_add) & (df['Nitrite_input'] == no2_add)]
#             x = df.columns.astype(float)
#             y_mean = df.mean(axis=0).values.astype(float)
#             ax.plot(x, y_mean, color=color, linestyle='-', linewidth=1.5, alpha=1, label=legend)
#             for well in df.index:
#                 y = df.loc[well].values.astype(float)
#                 ax.plot(x, y, color=color, linestyle='-', linewidth=0.5, alpha=0.5)
# handles, labels = axes[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10)
# plt.savefig(f'plots/figure2_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Figure 3
# no3_or_no2 = 'no3'
# chl = 1
# data_to_plot = {}
# for id in ids:
#     df = data[id][no3_or_no2]
#     df = df[(df['Chloramphenicol'] == chl) & (df['Sample_type'] != 'Blank')]
#     data_to_plot[id] = df.iloc[:, :-4]

# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Normalized cumulative consumption $\tilde{C}(t)$ - individual replicates'
#                  + '\n' + r'$t_{last}$ = first zero time point (or last measured if never zero) | diagonal = linear model')
# batch_labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
# colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
# for i in range(6):
#     ax = axes[i // 3, i % 3]
#     df = data_to_plot[ids[i]]
#     time = df.columns.astype(float)
#     count = 0
#     for well in df.index:
#         y = df.loc[well].values.astype(float)
#         thrsh_init = 0.3
#         if y[0] < thrsh_init:
#             continue
        
#         thrsh_end = 0.05
#         zero_indices = np.where(y < thrsh_end)[0]
#         if len(zero_indices) > 0:
#             last_index = zero_indices[0]
#         else:
#             last_index = len(y) - 1

#         y = y[:last_index + 1]
#         y = (y[0] - y) / (y[0] - y[-1]) 
#         x = time[:last_index + 1] / time[last_index]
        
#         ax.plot(x, y, color=colors[i], linestyle='-', alpha=0.5)
#         count += 1
#     ax.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=1.0, label='Linear')
#     ax.set_title(batch_labels[i] + f' (n={count})', color=colors[i])
#     if i == 0:
#         ax.legend(loc='upper left')
#     if i  % 3 == 0:
#         ax.set_ylabel(r'$\tilde{C}(t) = \frac{A(0) - A(t)}{A(0) - A(t_{last})}$')
#     if i // 3 == 1:
#         ax.set_xlabel(r'$\frac{t}{t_{last}}$')
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
# plt.savefig(f'plots/figure3_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Figure 4
# data_to_plot = load_data_to_plot(ids, data, 'no3', chl=1)
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'$I_0$ independance: all $I_0$ pairs, all relicate combinations - one panel per batch' 
#                  + '\n' + r'colour = nominal $A_{add}$ | diagonal = perfect agreement | all 3 x3 replicate pairs per ($A_{add}, I_{add}$ pair, batch)')
# colors = ['black', 'blue', 'orange', 'red']
# batch_labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
# for i in range(6):
#     ax = axes[i // 3, i % 3]
#     df = data_to_plot[ids[i]]
#     N = 0
#     se = 0
#     for A_add, color in zip([0.0, 0.7, 1.4, 2.0], colors):
#         df1 = df[df['Nitrate_input'] == A_add]
#         if A_add == 2.0:
#             I_add_pairs = [(0.0, 0.7), (0.0, 1.4), (0.7, 1.4)]
#         else:
#             I_add_pairs = [(0.0, 0.7), (0.0, 1.4), (0.0, 2.0), (0.7, 1.4), (0.7, 2.0), (1.4, 2.0)]
#         for I_add_pair in I_add_pairs:
#             df_x = df1[df1['Nitrite_input'] == I_add_pair[0]].iloc[:, :-4]
#             df_y = df1[df1['Nitrite_input'] == I_add_pair[1]].iloc[:, :-4]
#             if len(df_x.index) != 3 or len(df_y.index) != 3:
#                 print(f"Warning: Batch {ids[i]} - A_add={A_add} - I_add_pair={I_add_pair} have {len(df_x.index)} and {len(df_y.index)} replicates respectively, expected 3 each.")
#             for idx_x in df_x.index:
#                 for idx_y in df_y.index:
#                     x = df_x.loc[idx_x]
#                     y = df_y.loc[idx_y]
#                     ax.scatter(x, y, color=color, s=0.5, marker='o', alpha=0.3, label=f'$A_{{add}}$={A_add}' if (I_add_pair == (0.0, 0.7)) & (idx_x == df_x.index[0]) & (idx_y == df_y.index[0]) else None)
#                     se += np.sum((x - y) ** 2 / 2)
#                     N += len(x)
#     ax.legend(loc='upper left', ncols=2, fontsize=8) if i == 0 else None
#     ax.set_xlabel(r'A(t), $I_{add}$=a (mM)') if i // 3 == 1 else None
#     ax.set_ylabel(r'A(t), $I_{add}$=b (mM)') if i % 3 == 0 else None
#     ax.set_title(batch_labels[i], fontsize=10)
#     ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], color='gray', linestyle='--', linewidth=1, alpha=0.7)
#     ax.text(0.95, 0.05, f'RMSE = {np.sqrt(se/N):.3f} mM', 
#         transform=ax.transAxes, fontsize=9,
#         verticalalignment='bottom', horizontalalignment='right',
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.1))
# plt.savefig(f'plots/figure4_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Figure 5
# data_to_plot = load_data_to_plot(ids, data, 'no3', chl=1)
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Mean $NO_3^-$ curves averaged over replicates and $I_0$ - per batch'
#                     + '\n' + r'colour = nominal $A_{add}$ | band = $\pm$ SEM | faint = indicidual curves')    
# colors = ['gray', 'blue', 'orange', 'red']

# for i in range(6):
#     ax = axes[i // 3, i % 3]
#     df = data_to_plot[ids[i]]
#     for A_add, color in zip([0.0, 0.7, 1.4, 2.0], colors):
#         df_A = df[df['Nitrate_input'] == A_add]
#         x = df_A.columns[:-4].astype(float)
#         y_mean = df_A.iloc[:, :-4].mean(axis=0).values.astype(float)
#         y_sem = df_A.iloc[:, :-4].sem(axis=0).values.astype(float)
#         ax.plot(x, y_mean, color=color, linestyle='-', linewidth=2, label=f'$A_{{add}}$={A_add}')
#         ax.fill_between(x, y_mean - y_sem, y_mean + y_sem, color=color, alpha=0.2)
#         for idx in df_A.index:
#             y = df_A.loc[idx].iloc[:-4].values.astype(float)
#             ax.plot(x, y, color=color, linestyle='-', alpha=0.1)
#     ax.set_xlabel('Time (hr)') if i // 3 == 1 else None
#     ax.set_ylabel(r'$NO_3^-$ (mM)') if i % 3 == 0 else None
#     ax.set_title(f'Batch {i+1}', fontsize=10)
#     if i == 0:
#         ax.legend(loc='upper right', fontsize=8)
# fig.savefig(f'plots/figure5_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Figure 6
# data_to_plot = load_data_to_plot(ids, data, 'no3', chl=1)
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# fig.suptitle(r'Excess nitrate $A(0) - A_{add}$ by batch'
#                 + '\n' + r'colour/martker = nominal $A_{add}$ | diamond - batch median')
# ax.set_xticks(range(6))
# ax.set_xticklabels([f'Batch {j+1}' for j in range(6)])
# ax.set_ylabel(r'$A(0) - A_{add}$ (mM)')
# colors = ['gray', 'blue', 'orange', 'red']
# markers = ['o', 's', '^', 'D']
# for i in range(6):
#     df = data_to_plot[ids[i]]
#     median_val = np.median(df.iloc[:, :-4].iloc[:, 0].values.astype(float) - df['Nitrate_input'].values.astype(float))
#     ax.scatter([i], [median_val], color='white', marker='D', s=100, edgecolors='black', linewidths=1.5, zorder=5)
#     for A_add, color, marker in zip([0.0, 0.7, 1.4, 2.0], colors, markers):
#         df_A = df[df['Nitrate_input'] == A_add]
#         x_positions = [i] * len(df_A)
#         y_values = (df_A.iloc[:, :-4].iloc[:, 0].values.astype(float) - A_add)
#         ax.scatter(x_positions, y_values, color=color, marker=marker, alpha=0.3, s=50)
# handles = [
#     plt.Line2D([0], [0], marker=markers[0], color=colors[0], label='$A_{add}$=0.0 mM', linestyle='None'),
#     plt.Line2D([0], [0], marker=markers[1], color=colors[1], label='$A_{add}$=0.7 mM', linestyle='None'),
#     plt.Line2D([0], [0], marker=markers[2], color=colors[2], label='$A_{add}$=1.4 mM', linestyle='None'),
#     plt.Line2D([0], [0], marker=markers[3], color=colors[3], label='$A_{add}$=2.0 mM', linestyle='None'),
# ]
# ax.legend(handles=handles, loc='upper left')
# plt.savefig(f'plots/figure6_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Figure 7
# data_to_plot = load_data_to_plot(ids, data, 'no3', chl=1)
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Mean $NO_3^-$ curves with fitted linear slop $A(0) - st$')
#                     # + '\n' + r'batch 5 uses 8 time points | others use 6 | dotted = window boundary'
# colors = ['gray', 'blue', 'orange', 'red']
# batch_labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']

# for i in range(6):
#     ax = axes[i // 3, i % 3]
#     ax.set_title(batch_labels[i] + ' ( 6 pts)', fontsize=10)
#     # if ids[i] == '4.2.batch5':
#     #     ax.set_title(batch_labels[i] + ' ( 8 pts)', fontsize=10)
#     ax.set_xlabel('Time (hr)') if i // 3 == 1 else None
#     ax.set_ylabel(r'$NO_3^-$ (mM)') if i % 3 == 0 else None

#     for A_add, color in zip([0.0, 0.7, 1.4, 2.0], colors):
#         df = data_to_plot[ids[i]]
#         df_A = df[df['Nitrate_input'] == A_add]
#         x = df_A.columns[:-4].astype(float)
#         y_mean = df_A.iloc[:, :-4].mean(axis=0).values.astype(float)
        
#         fit_indices = range(6)
#         x_fit = x[fit_indices]
#         y_fit = y_mean[fit_indices]
#         slope, intercept = np.polyfit(x_fit, y_fit, 1)
#         y_line = intercept + slope * x_fit
#         ax.scatter(x, y_mean, color=color, marker='o', alpha=0.5, s=3, label=f'$A_{{add}}$={A_add}' if i == 0 else None)
#         ax.plot(x_fit, y_line, color=color, linestyle='-', linewidth=1)
#         ax.axvline(x=x_fit[-1], color='black', linestyle=':', linewidth=1)
#     ax.legend(loc='upper right') if i == 0 else None
# plt.savefig(f'plots/figure7_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Figure 8
# data_to_plot = load_data_to_plot(ids, data, 'no3', chl=1)
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Mean normalised cumulative consumption $\tilde{C}(t)$ per ($A_{add}$, batch)'
#                     + '\n' + r'averaged over replicates and $I_0$ | colour = nominal $A_{add}$ | diagonal = linear')
# colors = ['gray', 'blue', 'orange', 'red']
# for i in range(6):
#     ax = axes[i // 3, i % 3]
#     df = data_to_plot[ids[i]]
#     time = df.columns[:-4].astype(float)
#     for A_add, color in zip([0.0, 0.7, 1.4, 2.0], colors):
#         df_A = df[df['Nitrate_input'] == A_add]
#         y_mean = df_A.iloc[:, :-4].mean(axis=0).values.astype(float)
#         thrsh_end = 0.05
#         zero_indices = np.where(y_mean < thrsh_end)[0]
#         if len(zero_indices) > 0:
#             last_index = zero_indices[0]
#         else:
#             last_index = len(y_mean) - 1
#         y_mean = y_mean[:last_index + 1]
#         x_norm = time[:last_index + 1] / time[last_index]
#         y_norm = (y_mean[0] - y_mean) / (y_mean[0] - y_mean[-1])
#         ax.plot(x_norm, y_norm, color=color, linestyle='-', linewidth=2, label=f'$A_{{add}}$={A_add}')
#     ax.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=1.0)
#     ax.set_xlabel(r'$\frac{t}{t_{last}}$') if i // 3 == 1 else None
#     ax.set_ylabel(r'$\tilde{C}(t)$') if i % 3 == 0 else None
#     ax.set_title(f'Batch {i+1}', fontsize=10)
#     ax.set_xlim(-0.05, 1.05)
#     ax.set_ylim(-0.05, 1.05)
# handles, labels = axes[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10)
# plt.savefig(f'plots/figure8_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Figure 9
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.set_title(r's vs $\Phi$ - colour = batch | marker = nominal $A_{add}$ '
#                 + '\n' + r'bootstrap 68% CI | dashed = linear reference ($\Phi$ = 0.5)')
# ax.set_xlabel(r's (mM/hr)')
# ax.set_ylabel(r'$\Phi$ = Area under $\tilde{C}$')
# ax.set_xlim([0.0, 0.05])
# ax.set_ylim([0.4, 0.8])
# ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
# colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
# batch_labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
# markers = ['o', 's', '^', 'D']
# A_add_list = [0.0, 0.7, 1.4, 2.0]

# for id, color, batch_label in zip(ids, colors, batch_labels):
#     conc_df = data_to_plot[id]
#     s_df = - linear_fit_results[id]['Initial_slope']
#     phi_df = nccc_stat[id]['AUC_norm_cons']
#     for A_add, marker in zip(A_add_list, markers):
#         indices = conc_df[conc_df['Nitrate_input'] == A_add].index
#         s = s_df.loc[indices].astype(float).values
#         phi = phi_df.loc[indices].astype(float).values
#         s_mean, s_lb, s_ub = mean_bootstrap_ci(s, n_bootstrap=1000, ci=68)
#         phi_mean, phi_lb, phi_ub = mean_bootstrap_ci(phi, n_bootstrap=1000, ci=68)
#         ax.errorbar(s_mean, phi_mean, xerr=[[s_mean - s_lb], [s_ub - s_mean]], yerr=[[phi_mean - phi_lb], [phi_ub - phi_mean]],
#                     fmt=marker, color=color, alpha=0.8, markersize=6, label=batch_label if A_add == 0.0 else None)

# first_legend = ax.legend(loc='upper left', fontsize=9, title='Batch', ncols=2)
# ax.add_artist(first_legend)

# marker_handles = [plt.Line2D([0], [0], marker=m, color='black', label=f'$A_{{add}}$={a}', linestyle='None', markersize=6) 
#             for m, a in zip(markers, A_add_list)]
# ax.legend(handles=marker_handles, loc='lower right', fontsize=9, title=r'Nominal $A_{add}$', ncol=2)
# plt.savefig(f'plots/figure9_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Figure 9.1
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.set_title(r's vs $\Phi$ - colour = batch | marker = nominal $A_{add}$ | dashed = linear reference ($\Phi$ = 0.5)')
# ax.set_xlabel(r's (mM/hr)')
# ax.set_ylabel(r'$\Phi$ = Area under $\tilde{C}$')
# ax.set_ylim([0.2, 1.0])
# ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
# colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
# batch_labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
# markers = ['o', 's', '^', 'D']
# A_add_list = [0.0, 0.7, 1.4, 2.0]

# for id, color, batch_label in zip(ids, colors, batch_labels):
#     conc_df = data_to_plot[id]
#     s_df = - linear_fit_results[id]['Initial_slope']
#     phi_df = nccc_stat[id]['AUC_norm_cons']
#     for A_add, marker in zip(A_add_list, markers):
#         indices = conc_df[conc_df['Nitrate_input'] == A_add].index
#         s = s_df.loc[indices].astype(float)
#         phi = phi_df.loc[indices].astype(float)
#         ax.scatter(s, phi, color=color, marker=marker, alpha=0.5, s=50, label=batch_label if A_add == 0.0 else None)

# first_legend = ax.legend(loc='upper left', fontsize=9, title='Batch', ncols=2)
# ax.add_artist(first_legend)

# marker_handles = [plt.Line2D([0], [0], marker=m, color='black', label=f'$A_{{add}}$={a}', linestyle='None', markersize=6) 
#             for m, a in zip(markers, A_add_list)]
# ax.legend(handles=marker_handles, loc='lower right', fontsize=9, title=r'Nominal $A_{add}$', ncol=2)
# plt.savefig(f'plots/figure9.1_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Figure 10
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.set_title(r's vs excess nitrate - batch 1-2 faded'
#                 + '\n' + r'bootstrap 68% CI on both axes')
# ax.set_xlabel(r'Excess nitrate $A(0) - A_{add}$ (mM)')
# ax.set_ylabel(r's (mM/hr)')

# colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
# batch_labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
# markers = ['o', 's', '^', 'D']
# A_add_list = [0.0, 0.7, 1.4, 2.0]

# for id, color, batch_label in zip(ids, colors, batch_labels):
#     conc_df = data_to_plot[id]
#     s_df = - linear_fit_results[id]['Initial_slope']
#     excess_nitrate_df = conc_df.iloc[:, :-4].iloc[:, 0].astype(float) - conc_df['Nitrate_input'].astype(float)
#     for A_add, marker in zip(A_add_list, markers):
#         indices = conc_df[conc_df['Nitrate_input'] == A_add].index
#         s = s_df.loc[indices].astype(float).values
#         excess_nitrate = excess_nitrate_df.loc[indices].values
#         s_mean, s_lb, s_ub = mean_bootstrap_ci(s, n_bootstrap=1000, ci=68)
#         excess_mean, excess_lb, excess_ub = mean_bootstrap_ci(excess_nitrate, n_bootstrap=1000, ci=68)
#         alpha = 0.2 if batch_label in ['Batch 1', 'Batch 2'] else 0.8
#         ax.errorbar(excess_mean, s_mean, xerr=[[excess_mean - excess_lb], [excess_ub - excess_mean]], yerr=[[s_mean - s_lb], [s_ub - s_mean]],
#                     fmt=marker, color=color, alpha=alpha, markersize=6, label=batch_label if A_add == 0.0 else None)
# first_legend = ax.legend(loc='upper left', fontsize=9, title='Batch', ncols=2)
# ax.add_artist(first_legend)

# marker_handles = [plt.Line2D([0], [0], marker=m, color='black', label=f'$A_{{add}}$={a}', linestyle='None', markersize=6) 
#             for m, a in zip(markers, A_add_list)]
# ax.legend(handles=marker_handles, loc='lower right', fontsize=9, title=r'Nominal $A_{add}$', ncol=2)
# plt.savefig(f'plots/figure10_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Figure 11
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.set_title(r'$\Phi$ vs excess nitrate - batch 1-2 faded'
#                 + '\n' + r'bootstrap 68% CI on both axes | dashed = linear reference ($\Phi$ = 0.5)')
# ax.set_xlabel(r'Excess nitrate $A(0) - A_{add}$ (mM)')
# ax.set_ylabel(r'$\Phi$ = Area under $\tilde{C}$')
# ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
# ax.set_ylim([0.4, 1.0])

# for id, color, batch_label in zip(ids, batch_colors, batch_labels):
#     conc_df = data_to_plot[id]
#     phi_df = nccc_stat[id]['AUC_norm_cons']
#     excess_nitrate_df = conc_df.iloc[:, :-4].iloc[:, 0].astype(float) - conc_df['Nitrate_input'].astype(float)
#     for A_add, marker in zip(A_add_list, A_add_markers):
#         indices = conc_df[conc_df['Nitrate_input'] == A_add].index
#         phi = phi_df.loc[indices].astype(float).values
#         excess_nitrate = excess_nitrate_df.loc[indices].values
#         phi_mean, phi_lb, phi_ub = mean_bootstrap_ci(phi, n_bootstrap=1000, ci=68)
#         excess_mean, excess_lb, excess_ub = mean_bootstrap_ci(excess_nitrate, n_bootstrap=1000, ci=68)
#         alpha = 0.2 if batch_label in ['Batch 1', 'Batch 2'] else 0.8
#         ax.errorbar(excess_mean, phi_mean, xerr=[[excess_mean - excess_lb], [excess_ub - excess_mean]], yerr=[[phi_mean - phi_lb], [phi_ub - phi_mean]],
#                     fmt=marker, color=color, alpha=alpha, markersize=6, label=batch_label if A_add == 0.0 else None)
# first_legend = ax.legend(loc='upper left', fontsize=9, title='Batch', ncols=2)
# ax.add_artist(first_legend)
# marker_handles = [plt.Line2D([0], [0], marker=m, color='black', label=f'$A_{{add}}$={a}', linestyle='None', markersize=6) 
#             for m, a in zip(A_add_markers, A_add_list)]
# ax.legend(handles=marker_handles, loc='lower right', fontsize=9, title=r'Nominal $A_{add}$', ncol=2)
# plt.savefig(f'plots/figure11_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Figure 12
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.set_title(r'$\delta$ from  $\Phi$ inversion - curves that never hit zero only'
#                 + '\n' + r'bootstrap 68% CI | diamond = batch median')
# ax.set_xlabel('Batch (drying time course)')
# ax.set_xticks(range(6))
# ax.set_xticklabels(batch_labels)
# ax.set_ylabel(r'$\delta$ ($hr^{-1}$)')

# for i in range(6):
#     conc_df = data_to_plot[ids[i]]
#     nccc_stat_df = nccc_stat[ids[i]]
#     df = pd.concat([nccc_stat_df[['death_rate', 'curve_hits_zero']], conc_df[['Nitrate_input']]], axis=1)
#     delta = df[df['curve_hits_zero'] == False]['death_rate'].astype(float).values
#     delta = delta[~np.isnan(delta)]
#     median_delta = np.median(delta)
#     ax.scatter(i, median_delta, color='white', marker='D', s=100, edgecolors=batch_colors[i], linewidths=1.5, zorder=5, alpha=0.8)
#     for A_add, marker in zip(A_add_list, A_add_markers):
#         indices = df[(df['Nitrate_input'] == A_add) & (df['curve_hits_zero'] == False)].index
#         delta_A = df.loc[indices, ['death_rate']].astype(float).values
#         delta_A = delta_A[~np.isnan(delta_A)]
#         if len(delta_A) == 0:
#             continue
#         delta_mean, delta_lb, delta_ub = mean_bootstrap_ci(delta_A, n_bootstrap=1000, ci=68)
#         time = i + (A_add - 1.0) * 0.1
#         ax.errorbar(time, delta_mean, yerr=[[delta_mean - delta_lb], [delta_ub - delta_mean]],
#                     fmt=marker, color=batch_colors[i], alpha=0.8, markersize=6)
# handles = [plt.Line2D([0], [0], marker=m, color='black', label=f'$A_{{add}}$={a}', linestyle='None', markersize=6) 
#             for m, a in zip(A_add_markers, A_add_list)]
# ax.legend(handles=handles, loc='upper right', fontsize=9, title=r'Nominal $A_{add}$', ncols=2)
# plt.savefig(f'plots/figure12_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Figure 12.1
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.set_title(r'$\delta$ from  $\Phi$ inversion'
#                 + '\n' + r'bootstrap 68% CI | diamond = batch median')
# ax.set_xlabel('Batch (drying time course)')
# ax.set_xticks(range(6))
# ax.set_xticklabels(batch_labels)
# ax.set_ylabel(r'$\delta$ ($hr^{-1}$)')

# for i in range(6):
#     conc_df = data_to_plot[ids[i]]
#     nccc_stat_df = nccc_stat[ids[i]]
#     df = pd.concat([nccc_stat_df[['death_rate', 'curve_hits_zero']], conc_df[['Nitrate_input']]], axis=1)
#     delta = df['death_rate'].astype(float).values
#     delta = delta[~np.isnan(delta)]
#     median_delta = np.median(delta)
#     ax.scatter(i, median_delta, color='white', marker='D', s=100, edgecolors=batch_colors[i], linewidths=1.5, zorder=5, alpha=0.8)
#     for A_add, marker in zip(A_add_list, A_add_markers):
#         indices = df[(df['Nitrate_input'] == A_add)].index
#         delta_A = df.loc[indices, ['death_rate']].astype(float).values
#         delta_A = delta_A[~np.isnan(delta_A)]
#         if len(delta_A) == 0:
#             continue
#         delta_mean, delta_lb, delta_ub = mean_bootstrap_ci(delta_A, n_bootstrap=1000, ci=68)
#         ax.errorbar(i, delta_mean, yerr=[[delta_mean - delta_lb], [delta_ub - delta_mean]],
#                     fmt=marker, color=batch_colors[i], alpha=0.8, markersize=6)
# handles = [plt.Line2D([0], [0], marker=m, color='black', label=f'$A_{{add}}$={a}', linestyle='None', markersize=6) 
#             for m, a in zip(A_add_markers, A_add_list)]
# ax.legend(handles=handles, loc='upper right', fontsize=9, title=r'Nominal $A_{add}$', ncols=2)
# plt.savefig(f'plots/figure12.1_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Figure 13
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.set_title(r'$\delta$ vs s -- curves that never hit zero only'
#                 + '\n' + r'bootstrap 68% CI | colour = batch | marker = nominal $A_{add}$')
# ax.set_xlabel(r's (mM/hr)')
# ax.set_ylabel(r'$\delta$ ($hr^{-1}$)')

# for i in range(6):
#     conc_df = data_to_plot[ids[i]]
#     linear_fit_df = linear_fit_results[ids[i]]
#     nccc_stat_df = nccc_stat[ids[i]]
#     df = pd.concat([linear_fit_df['Initial_slope'], nccc_stat_df[['death_rate', 'curve_hits_zero']], conc_df['Nitrate_input']], axis=1)
#     df = df[df['curve_hits_zero'] == False]
#     for A_add, markers in zip(A_add_list, A_add_markers):
#         df_A = df[df['Nitrate_input'] == A_add]
#         s = -df_A['Initial_slope'].astype(float).values
#         s = s[~np.isnan(s)]
#         delta = df_A['death_rate'].astype(float).values
#         delta = delta[~np.isnan(delta)]

#         if len(s) == 0 or len(delta) == 0:
#             continue
#         s_mean, s_lb, s_ub = mean_bootstrap_ci(s, n_bootstrap=1000, ci=68)
#         delta_mean, delta_lb, delta_ub = mean_bootstrap_ci(delta, n_bootstrap=1000, ci=68)
#         ax.errorbar(s_mean, delta_mean, xerr=[[s_mean - s_lb], [s_ub - s_mean]], yerr=[[delta_mean - delta_lb], [delta_ub - delta_mean]],
#                     fmt=markers, color=batch_colors[i], alpha=0.8, markersize=6, label=batch_labels[i] if A_add == 2.0 else None)
# first_legend = ax.legend(loc='upper left', title='Batch', ncols=2)
# ax.add_artist(first_legend)
# marker_handles = [plt.Line2D([0], [0], marker=m, color='black', label=f'$A_{{add}}$={a}', linestyle='None', markersize=6) 
#             for m, a in zip(A_add_markers, A_add_list)]
# ax.legend(handles=marker_handles, loc='lower right', title=r'Nominal $A_{add}$', ncols=2)
# plt.savefig(f'plots/figure13_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Figure 14
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.set_title(r's for batches | bootstrap 68% CI | diamond = batch median')
# ax.set_xlabel('Batch (drying time course)')
# ax.set_xticks(range(6))
# ax.set_xticklabels(batch_labels)
# ax.set_ylabel(r's (mM/hr)')

# for i in range(6):
#     conc_df = data_to_plot[ids[i]]
#     linear_fit_df = linear_fit_results[ids[i]]
#     df = pd.concat([linear_fit_df['Initial_slope'], conc_df['Nitrate_input']], axis=1)
#     median = np.median(-df['Initial_slope'].astype(float).values)
#     ax.scatter(i, median, color='white', marker='D', s=100, edgecolors=batch_colors[i], linewidths=1.5, zorder=5, alpha=0.8)
#     for A_add, marker in zip(A_add_list, A_add_markers):
#         df_A = df[df['Nitrate_input'] == A_add]
#         s = -df_A['Initial_slope'].astype(float).values
#         s = s[~np.isnan(s)]
#         if len(s) == 0:
#             continue
#         s_mean, s_lb, s_ub = mean_bootstrap_ci(s, n_bootstrap=1000, ci=68)
#         x = i + (A_add - 1.0) * 0.1
#         ax.errorbar(x, s_mean, yerr=[[s_mean - s_lb], [s_ub - s_mean]],
#                     fmt=marker, color=batch_colors[i], alpha=0.8, markersize=6)
# handles = [plt.Line2D([0], [0], marker=m, color='black', label=f'$A_{{add}}$={a}', linestyle='None', markersize=6)
#             for m, a in zip(A_add_markers, A_add_list)]
# ax.legend(handles=handles, loc='upper right', fontsize=9, title=r'Nominal $A_{add}$', ncols=2)
# plt.savefig(f'plots/figure14_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()