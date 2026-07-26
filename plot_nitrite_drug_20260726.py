from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import fsolve
from matplotlib.lines import Line2D
import warnings

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
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            data_dict[id]['no2_cons_rate'].loc[index, f'first_{n}_points'] = slope



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

# plt.tight_layout()
# plt.savefig(f'plots/nitrite_consumption_rate_first_n_time_points_CHL+_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
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

# plt.tight_layout()
# plt.savefig(f'plots/nitrite_consumption_rate_first_n_time_points_CHL+_grouped_by_A_add_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()
