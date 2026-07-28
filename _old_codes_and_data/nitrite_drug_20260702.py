from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import fsolve
from matplotlib.lines import Line2D

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

# # Normalized cumulative consumption
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Normalized cumulative consumption of nitrite $\Delta B \equiv \Delta I + \Delta A$'
#                  + '\n' + r'individual replicates | $t_{last}$ = last measured time | diagonal = linear model')

# for i in range(6):
#     ax = axes[i // 3, i % 3]    
#     id = ids[i]
#     df = data_dict[id]['no2_cons'].loc[mask_chl_blank_dict[id]]
#     time = df.columns.values.astype(float)
#     count = 0
#     for well in df.index:
#         B = df.loc[well].values.astype(float)
#         if B[-1] < 0.1:         # exclude wells that consumed less than 0.3 mM
#             continue
#         y = B / B[-1]
#         x = time / time[-1]
        
#         ax.plot(x, y, color=batch_colors[i], linestyle='-', alpha=0.5)
#         count += 1
#         ax.set_title(batch_labels[i] + f'(n={count})')
#     ax.set_xlabel(r'$\frac{t}{t_{last}}$') if i // 3 == 1 else None
#     ax.set_ylabel(r'$\tilde{C}(t) = \frac{B(0) - B(t)}{B(0) - B(t_{last})}$') if i % 3 == 0 else None
#     ax.set_xlim(0, 1)
#     ax.set_ylim(-0.1, 1.1)
#     ax.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=1.0, label='Linear')
#     if i == 0:
#         ax.legend()

# plt.savefig(f'plots/nitrite_norm_cum_cons_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()


# # Nitrite consumption
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Nitrite cumulative consumption, drug conditions | colour = nominal $A_{add}$', fontsize=16)
# for i in range(6):
#     ax = axes[i // 3, i % 3]
#     ax.set_title(batch_labels[i])
#     ax.set_xlabel('Time (hr)') if i // 3 == 1 else None
#     ax.set_ylabel(r'$A(0) - A(t) + I(0) - I(t)$ (mM)') if i % 3 == 0 else None
#     if i == 0:
#         custom_lines = [Line2D([0], [0], color=colors[0]),
#                         Line2D([0], [0], color=colors[1]),
#                         Line2D([0], [0], color=colors[2]),
#                         Line2D([0], [0], color=colors[3])]
#         ax.legend(custom_lines, ['0.0 mM', '0.7 mM', '1.4 mM', '2.0 mM'], title=r'Nominal $A_{add}$')

#     id = ids[i]
#     df = data_dict[id]['no2_cons'].loc[mask_chl_blank_dict[id]]
#     meta_df = data_dict[id]['metadata'].loc[mask_chl_blank_dict[id]]
#     times = df.columns.values.astype(float)
#     for index in df.index:
#         no2_cons = df.loc[index].values.astype(float)
#         no3_add = meta_df.loc[index, 'Nitrate_input']
#         ax.plot(times, no2_cons, linestyle='-', color=colors[add_conc.index(no3_add)], alpha = 0.3)
# plt.savefig(f'plots/4.2.chl1_B(t)-B(0)_color_A_add_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)
# plt.show()

# Nitrite consumption colored by I_add
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(r'Nitrite cumulative consumption, drug conditions | colour = nominal $I_{add}$', fontsize=16)
for i in range(6):
    ax = axes[i // 3, i % 3]
    ax.set_title(batch_labels[i])
    ax.set_xlabel('Time (hr)') if i // 3 == 1 else None
    ax.set_ylabel(r'$A(0) - A(t) + I(0) - I(t)$ (mM)') if i % 3 == 0 else None
    if i == 0:
        custom_lines = [Line2D([0], [0], color=colors[0]),
                        Line2D([0], [0], color=colors[1]),
                        Line2D([0], [0], color=colors[2]),
                        Line2D([0], [0], color=colors[3])]
        ax.legend(custom_lines, ['0.0 mM', '0.7 mM', '1.4 mM', '2.0 mM'], title=r'Nominal $I_{add}$')

    id = ids[i]
    df = data_dict[id]['no2_cons'].loc[mask_chl_blank_dict[id]]
    meta_df = data_dict[id]['metadata'].loc[mask_chl_blank_dict[id]]
    times = df.columns.values.astype(float)
    for index in df.index:
        no2_cons = df.loc[index].values.astype(float)
        no2_add = meta_df.loc[index, 'Nitrite_input']
        ax.plot(times, no2_cons, linestyle='-', color=colors[add_conc.index(no2_add)], alpha=0.3)
plt.savefig(f'plots/4.2.chl1_B(t)-B(0)_color_I_add_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)
plt.show()