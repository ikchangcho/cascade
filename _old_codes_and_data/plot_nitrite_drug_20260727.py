from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D

def linear_func(x, m):
    return m * x

ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5', '4.2.batch6']
batch_colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
batch_labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
add_conc = [0.0, 0.7, 1.4, 2.0]
markers = ['o', 's', '^', 'D']
colors = ['blue', 'green', 'red', 'cyan']

data_dict = {}
for id in ids:
    data_dict[id] = {}
    data_dict[id]['no3_conc'] = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0).iloc[:, :-4]
    data_dict[id]['no2_conc'] = pd.read_csv(f'concentrations/{id}_no2_conc.csv', index_col=0).iloc[:, :-4]
    data_dict[id]['no3_cons'] = pd.read_csv(f'concentrations/{id}_no3_cons.csv', index_col=0).iloc[:, :-4]
    data_dict[id]['no2_cons'] = pd.read_csv(f'concentrations/{id}_no2_cons.csv', index_col=0).iloc[:, :-4]
    data_dict[id]['metadata'] = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0).iloc[:, -4:]

mask_chl_blank_dict = {}
for id in ids:
    mask_chl_blank_dict[id] = (data_dict[id]['metadata']['Chloramphenicol'] == 1.0) & (data_dict[id]['metadata']['Sample_type'] != 'Blank')

for id in ids:
    data_dict[id]['no2_cons_rate'] = pd.DataFrame(index=data_dict[id]['no2_cons'].index)
    
    mask = mask_chl_blank_dict[id]
    for index in data_dict[id]['no3_conc'].loc[mask].index:
        for n in range(2, len(data_dict[id]['no2_cons'].columns) + 1):
            x = data_dict[id]['no2_cons'].columns[:n].values.astype(float)
            y = data_dict[id]['no2_cons'].loc[index].iloc[:n].values.astype(float)
            slope, _ = curve_fit(linear_func, x, y)
            data_dict[id]['no2_cons_rate'].loc[index, f'first_{n}_points'] = slope

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Total nitrite consumption (CHL+) vs $I_{add}$ | diamond = mean', fontsize=16)
for i, id in enumerate(ids):
    ax = axes[i // 3, i % 3]
    ax.set_xlabel(r'$I_{add}$ (mM)' if i >= 3 else '', fontsize=12)
    ax.set_xticks(add_conc)
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylabel('Total nitrite consumption (mM)' if i % 3 == 0 else '', fontsize=12)
    ax.set_ylim(-0.2, 1.2)
    mask = mask_chl_blank_dict[id]
    
    total_no2_consumption = data_dict[id]['no2_cons'].loc[mask].iloc[:, -1]
    df_meta = data_dict[id]['metadata'].loc[mask]
    
    median_consumption = []
    for no2_add in add_conc:
        mask_no2 = df_meta['Nitrite_input'] == no2_add
        consumption = total_no2_consumption.loc[mask_no2]
        median_consumption.append(consumption.median())
    ax.scatter(add_conc, median_consumption, marker='D', color='white', edgecolor=batch_colors[i],
               linewidths=1.5, alpha=0.7, s=100, zorder=3)
    
    for no2_add in add_conc:
        mask_no2 = df_meta['Nitrite_input'] == no2_add
        consumption = total_no2_consumption.loc[mask_no2]
        ax.scatter([no2_add] * len(consumption), consumption, color=batch_colors[i], alpha=0.5, s=50, label=f'$I_{{add}}$ = {no2_add} mM' if i == 0 else "")

        batch_median = total_no2_consumption.median()
        ax.set_title(f'{batch_labels[i]} (Median: {batch_median:.2f} mM)', fontsize=12)

plt.tight_layout()
plt.savefig(f'plots/total_nitrite_consumption_vs_I_add_CHL+_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
plt.show()

# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle('Total nitrite consumption (CHL+) vs $A_{add}$', fontsize=16)
# for i, id in enumerate(ids):
#     ax = axes[i // 3, i % 3]
#     ax.set_xlabel(r'$A_{add}$ (mM)' if i >= 3 else '', fontsize=12)
#     ax.set_xticks(add_conc)
#     ax.set_xlim(-0.1, 2.1)
#     ax.set_ylabel('Total nitrite consumption (mM)' if i % 3 == 0 else '', fontsize=12)
#     mask = mask_chl_blank_dict[id]
    
#     total_no2_consumption = data_dict[id]['no2_cons'].loc[mask].iloc[:, -1]
#     df_meta = data_dict[id]['metadata'].loc[mask]
    
#     mean_consumption = []
#     sem_consumption = []
#     for no3_add in add_conc:
#         mask_no3 = df_meta['Nitrate_input'] == no3_add
#         consumption = total_no2_consumption.loc[mask_no3]
#         mean_consumption.append(consumption.mean())
#         sem_consumption.append(consumption.sem())
#     ax.errorbar(add_conc, mean_consumption, yerr=sem_consumption, fmt='o', color=batch_colors[i], alpha=0.7,
#                 capsize=5, markersize=8, elinewidth=1.5)
#     for no3_add in add_conc:
#         mask_no3 = df_meta['Nitrate_input'] == no3_add
#         consumption = total_no2_consumption.loc[mask_no3]
#         ax.scatter([no3_add] * len(consumption), consumption, color=batch_colors[i], alpha=0.2, s=50, label=f'$A_{{add}}$ = {no3_add} mM' if i == 0 else "")

#         batch_median = total_no2_consumption.median()
#         ax.set_title(f'{batch_labels[i]} (Median: {batch_median:.2f} mM)', fontsize=12)

# plt.tight_layout()
# plt.savefig(f'plots/total_nitrite_consumption_vs_A_add_CHL+_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()

# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Nitrite consumption rate for first n = 6, 9, 11 time points (CHL+) vs $I_{add}$', fontsize=16)
# for i, id in enumerate(ids):
#     ax = axes[i // 3, i % 3]
#     ax.set_title(f'{batch_labels[i]}', fontsize=12)
#     ax.set_xlabel(r'$I_{add}$ (mM)' if i >= 3 else '', fontsize=12)
#     ax.set_xlim(-0.1, 2.1)
#     ax.set_ylabel('Nitrite consumption rate (mM/h)' if i % 3 == 0 else '', fontsize=12)
#     ax.set_xticks(add_conc)
#     mask = mask_chl_blank_dict[id]
    
#     for k, n in enumerate([6, 9, 11]):
#         marker = markers[k]
#         color = colors[k]
#         df_meta = data_dict[id]['metadata'].loc[mask]
#         df_slope = data_dict[id]['no2_cons_rate'].loc[mask, f'first_{n}_points']

#         slope_mean = []
#         slope_sem = []
#         for j, no2_add in enumerate(add_conc):
#             mask_no2 = df_meta['Nitrite_input'] == no2_add
#             slopes = df_slope.loc[mask_no2]
#             slope_mean.append(slopes.mean())
#             slope_sem.append(slopes.sem())
#         ax.errorbar(add_conc, slope_mean, yerr=slope_sem, fmt=marker, color=color, alpha=0.5,
#                     label=f'n = {n}', capsize=5, markersize=8, elinewidth=1.5)    
#     ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
#     ax.legend() if i == 0 else None
# plt.tight_layout()
# plt.savefig(f'plots/4.2.chl1_no2_cons_rate_vs_I_add_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# for n in [3, 6, 9, 11]:
#     fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#     ax.set_title(f'Nitrite consumption rate for first {n} time points (CHL+)' + '\n' + r'marker = $I_{add}$ | color = batch | diamond = batch median', fontsize=14)
#     ax.set_xticks(range(6))
#     ax.set_xticklabels([f'{batch_label}' for batch_label in batch_labels])
#     ax.set_ylabel('Nitrite consumption rate (mM/h)')
#     custom_lines = [Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=8, label=f'$I_{{add}}$ = {conc} mM') for marker, conc in zip(markers, add_conc)]
#     legend = ax.legend(handles=custom_lines)
#     ax.add_artist(legend)
#     for i, id in enumerate(ids):
#         mask = mask_chl_blank_dict[id]
#         for index in data_dict[id]['no2_cons_rate'].loc[mask].index:
#             slope = data_dict[id]['no2_cons_rate'].loc[index, f'first_{n}_points']
#             no2_add = data_dict[id]['metadata'].loc[index, 'Nitrite_input']
#             marker = markers[add_conc.index(no2_add)]
#             ax.scatter(i + 0.1 * (add_conc.index(no2_add) - 1.0), slope, color=batch_colors[i], marker=marker, alpha=0.5)
#             slope_median = data_dict[id]['no2_cons_rate'].loc[mask, f'first_{n}_points'].median()
#             ax.scatter(i, slope_median, color='white', edgecolors='black', marker='D', s=100)
#     ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
#     plt.savefig(f'plots/4.2.chl1_no2_cons_rate_first_{n}_time_points_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
#     plt.show()

# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Nitrite consumption rate for first n time points (CHL+) | grouped by $I_{add}$', fontsize=16)
# for i, id in enumerate(ids):
#     ax = axes[i // 3, i % 3]
#     ax.set_title(f'{batch_labels[i]}', fontsize=12)
#     ax.set_xlabel('Number of time points (n)' if i >= 3 else '', fontsize=12)
#     ax.set_ylabel('Nitrite consumption rate (mM/h)' if i % 3 == 0 else '', fontsize=12)
#     ax.set_xlim(1.5, len(data_dict[id]['no2_cons_rate'].columns) + 0.5)
#     x = np.arange(2, len(data_dict[id]['no2_cons_rate'].columns) + 2, dtype=float)
#     ax.set_xticks(x)
#     mask = mask_chl_blank_dict[id]
#     df = data_dict[id]['no2_cons_rate'].loc[mask]
#     for j, no2_add in enumerate(add_conc):
#         mask_no2_add = data_dict[id]['metadata']['Nitrite_input'] == no2_add
#         df_no2_add = df.loc[mask_no2_add]
#         ax.errorbar(x + (no2_add - 1.0) * 0.2, df_no2_add.mean(), yerr=df_no2_add.sem(), fmt=markers[j], color=colors[j], linestyle='-'
#                     , label=r'$I_{add}$ =' f'{no2_add} mM', capsize=5, markersize=8, elinewidth=1.5)
#     ax.legend() if i == 0 else None
#     ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)

# plt.tight_layout()
# plt.savefig(f'plots/nitrite_consumption_rate_first_n_time_points_CHL+_grouped_by_I_add_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Nitrite consumption rate for first n time points (CHL+) | grouped by $A_{add}$', fontsize=16)
# for i, id in enumerate(ids):
#     ax = axes[i // 3, i % 3]
#     ax.set_title(f'{batch_labels[i]}', fontsize=12)
#     ax.set_xlabel('Number of time points (n)' if i >= 3 else '', fontsize=12)
#     ax.set_ylabel('Nitrite consumption rate (mM/h)' if i % 3 == 0 else '', fontsize=12)
#     ax.set_xlim(1.5, len(data_dict[id]['no2_cons_rate'].columns) + 0.5)
#     x = np.arange(2, len(data_dict[id]['no2_cons_rate'].columns) + 2, dtype=float)
#     ax.set_xticks(x)
#     mask = mask_chl_blank_dict[id]
#     df = data_dict[id]['no2_cons_rate'].loc[mask]
#     for j, no3_add in enumerate(add_conc):
#         mask_no3_add = data_dict[id]['metadata']['Nitrate_input'] == no3_add
#         df_no3_add = df.loc[mask_no3_add]
#         ax.errorbar(x + (no3_add - 1.0) * 0.2, df_no3_add.mean(), yerr=df_no3_add.sem(), fmt=markers[j], color=colors[j], linestyle='-'
#                     , label=r'$A_{add}$ =' f'{no3_add} mM', capsize=5, markersize=8, elinewidth=1.5)
#     ax.legend() if i == 0 else None
#     ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)

# plt.tight_layout()
# plt.savefig(f'plots/nitrite_consumption_rate_first_n_time_points_CHL+_grouped_by_A_add_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()
