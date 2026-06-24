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


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(r'Nitrate concentration subtracted by initial concentrations, drug conditions | colour = nominal $A_{add}$', fontsize=16)
for i in range(6):
    ax = axes[i // 3, i % 3]
    ax.set_title(batch_labels[i])
    ax.set_xlabel('Time (hr)') if i // 3 == 1 else None
    ax.set_ylabel(r'$A(t) - A(0)$ (mM)') if i % 3 == 0 else None

    if i == 0:
        custom_lines = [Line2D([0], [0], color=colors[0]),
                        Line2D([0], [0], color=colors[1]),
                        Line2D([0], [0], color=colors[2]),
                        Line2D([0], [0], color=colors[3])]
        ax.legend(custom_lines, ['0.0 mM', '0.7 mM', '1.4 mM', '2.0 mM'], title=r'Nominal $A_{add}$')

    id = ids[i]
    no3_conc_df = data_dict[id]['no3_conc'].loc[mask_chl_blank_dict[id]]
    meta_df = data_dict[id]['metadata'].loc[mask_chl_blank_dict[id]]
    times = no3_conc_df.columns.values.astype(float)
    for index in no3_conc_df.index:
        no3_conc = no3_conc_df.loc[index].values.astype(float)
        no3_cons = no3_conc - no3_conc[0]  
        no3_add = meta_df.loc[index, 'Nitrate_input']
        ax.plot(times, no3_cons, linestyle='-', color=colors[add_conc.index(no3_add)], alpha = 0.3)
plt.savefig(f'plots/4.2.chl1_A(t)-A(0)_color_A_add_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(r'Nitrate concentration subtracted by initial concentrations, drug conditions'
                + '\n' + r'colour = nominal $A_{add}$ | line = average over each $A_{add}$| band = $\pm$ SEM')    
for i in range(6):
    ax = axes[i // 3, i % 3]
    ax.set_title(batch_labels[i])
    ax.set_xlabel('Time (hr)') if i // 3 == 1 else None
    ax.set_ylabel(r'$A(t) - A(0)$ (mM)') if i % 3 == 0 else None

    if i == 0:
        custom_lines = [Line2D([0], [0], color=colors[0]),
                        Line2D([0], [0], color=colors[1]),
                        Line2D([0], [0], color=colors[2]),
                        Line2D([0], [0], color=colors[3])]
        ax.legend(custom_lines, ['0.0 mM', '0.7 mM', '1.4 mM', '2.0 mM'], title=r'Nominal $A_{add}$')

    id = ids[i]
    no3_conc_df = data_dict[id]['no3_conc'].loc[mask_chl_blank_dict[id]]
    no3_cons_df = no3_conc_df.sub(no3_conc_df.iloc[:, 0], axis=0)
    meta_df = data_dict[id]['metadata'].loc[mask_chl_blank_dict[id]]
    times = no3_conc_df.columns.values.astype(float)
    for no3_add in add_conc:
        mask = (meta_df['Nitrate_input'] == no3_add)
        mean_no3_cons = no3_cons_df.loc[mask].mean().values.astype(float)
        sem_no3_cons = no3_cons_df.loc[mask].sem().values.astype(float)
        
        ax.plot(times, mean_no3_cons, linestyle='-', color=colors[add_conc.index(no3_add)], alpha = 0.8)
        ax.fill_between(times, mean_no3_cons - sem_no3_cons, mean_no3_cons + sem_no3_cons, color=colors[add_conc.index(no3_add)], alpha=0.2)
plt.savefig(f'plots/4.2.chl1_A(t)-A(0)_color_A_add_line_band_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(r'Nitrate concentration subtracted by initial concentrations, drug conditions | colour = nominal $I_{add}$', fontsize=16)
for i in range(6):
    ax = axes[i // 3, i % 3]
    ax.set_title(batch_labels[i])
    ax.set_xlabel('Time (hr)') if i // 3 == 1 else None
    ax.set_ylabel(r'$A(t) - A(0)$ (mM)') if i % 3 == 0 else None

    if i == 0:
        custom_lines = [Line2D([0], [0], color=colors[0]),
                        Line2D([0], [0], color=colors[1]),
                        Line2D([0], [0], color=colors[2]),
                        Line2D([0], [0], color=colors[3])]
        ax.legend(custom_lines, ['0.0 mM', '0.7 mM', '1.4 mM', '2.0 mM'], title=r'Nominal $I_{add}$')

    id = ids[i]
    no3_conc_df = data_dict[id]['no3_conc'].loc[mask_chl_blank_dict[id]]
    meta_df = data_dict[id]['metadata'].loc[mask_chl_blank_dict[id]]
    times = no3_conc_df.columns.values.astype(float)
    for index in no3_conc_df.index:
        no3_conc = no3_conc_df.loc[index].values.astype(float)
        no3_cons = no3_conc - no3_conc[0]  
        no2_add = meta_df.loc[index, 'Nitrite_input']
        ax.plot(times, no3_cons, linestyle='-', color=colors[add_conc.index(no2_add)], alpha = 0.3)
plt.savefig(f'plots/4.2.chl1_A(t)-A(0)_color_I_add_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(r'Nitrate concentration subtracted by initial concentrations, drug conditions'
                + '\n' + r'colour = nominal $I_{add}$ | line = average over each $A_{add}$| band = $\pm$ SEM')    
for i in range(6):
    ax = axes[i // 3, i % 3]
    ax.set_title(batch_labels[i])
    ax.set_xlabel('Time (hr)') if i // 3 == 1 else None
    ax.set_ylabel(r'$A(t) - A(0)$ (mM)') if i % 3 == 0 else None

    if i == 0:
        custom_lines = [Line2D([0], [0], color=colors[0]),
                        Line2D([0], [0], color=colors[1]),
                        Line2D([0], [0], color=colors[2]),
                        Line2D([0], [0], color=colors[3])]
        ax.legend(custom_lines, ['0.0 mM', '0.7 mM', '1.4 mM', '2.0 mM'], title=r'Nominal $I_{add}$')

    id = ids[i]
    no3_conc_df = data_dict[id]['no3_conc'].loc[mask_chl_blank_dict[id]]
    no3_cons_df = no3_conc_df.sub(no3_conc_df.iloc[:, 0], axis=0)
    meta_df = data_dict[id]['metadata'].loc[mask_chl_blank_dict[id]]
    times = no3_conc_df.columns.values.astype(float)
    for no2_add in add_conc:
        mask = (meta_df['Nitrite_input'] == no2_add)
        mean_no3_cons = no3_cons_df.loc[mask].mean().values.astype(float)
        sem_no3_cons = no3_cons_df.loc[mask].sem().values.astype(float)
        
        ax.plot(times, mean_no3_cons, linestyle='-', color=colors[add_conc.index(no2_add)], alpha = 0.8)
        ax.fill_between(times, mean_no3_cons - sem_no3_cons, mean_no3_cons + sem_no3_cons, color=colors[add_conc.index(no2_add)], alpha=0.2)
plt.savefig(f'plots/4.2.chl1_A(t)-A(0)_color_I_add_line_band_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)
plt.show()


# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Nitrate concentration subtracted by grouped mean of $A(0)$, drug conditions | colour = nominal $A_{add}$', fontsize=16)
# for i in range(6):
#     ax = axes[i // 3, i % 3]
#     ax.set_title(batch_labels[i])
#     ax.set_xlabel('Time (hr)') if i // 3 == 1 else None
#     ax.set_ylabel(r'$A(t) - \overline{A(0)}$') if i % 3 == 0 else None

#     if i == 0:
#         custom_lines = [Line2D([0], [0], color=colors[0]),
#                         Line2D([0], [0], color=colors[1]),
#                         Line2D([0], [0], color=colors[2]),
#                         Line2D([0], [0], color=colors[3])]
#         ax.legend(custom_lines, ['0.0 mM', '0.7 mM', '1.4 mM', '2.0 mM'], title=r'Nominal $A_{add}$')

#     id = ids[i]
#     no3_conc_df = data_dict[id]['no3_conc'].loc[mask_chl_blank_dict[id]]
#     meta_df = data_dict[id]['metadata'].loc[mask_chl_blank_dict[id]]
    
#     no3_conc_norm_df = no3_conc_df
#     for no3_add in add_conc:
#         mask = (meta_df['Nitrate_input'] == no3_add)
#         mean_no3_value = no3_conc_df.loc[mask].iloc[:, 0].mean()
#         no3_conc_norm_df.loc[mask] = no3_conc_df.loc[mask] - mean_no3_value
    
#     times = no3_conc_norm_df.columns.values.astype(float)
#     for index in no3_conc_norm_df.index:
#         no3_conc = no3_conc_norm_df.loc[index].values.astype(float)
#         no3_add = meta_df.loc[index, 'Nitrate_input']
#         ax.plot(times, no3_conc, linestyle='-', color=colors[add_conc.index(no3_add)], alpha = 0.3)
# plt.savefig(f'plots/4.2.chl1_A(t)-A(0)_color_A_add_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)
# plt.show()


# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Nitrate concentration subtracted by grouped mean of $A(0)$, drug conditions'
#                 + '\n' + r'colour = nominal $A_{add}$ | line = average over each $A_{add}$| band = $\pm$ SEM')    
# for i in range(6):
#     ax = axes[i // 3, i % 3]
#     ax.set_title(batch_labels[i])
#     ax.set_xlabel('Time (hr)') if i // 3 == 1 else None
#     ax.set_ylabel(r'$A(t) - \overline{A(0)}$') if i % 3 == 0 else None

#     if i == 0:
#         custom_lines = [Line2D([0], [0], color=colors[0]),
#                         Line2D([0], [0], color=colors[1]),
#                         Line2D([0], [0], color=colors[2]),
#                         Line2D([0], [0], color=colors[3])]
#         ax.legend(custom_lines, ['0.0 mM', '0.7 mM', '1.4 mM', '2.0 mM'], title=r'Nominal $A_{add}$')

#     id = ids[i]
#     no3_conc_df = data_dict[id]['no3_conc'].loc[mask_chl_blank_dict[id]]
#     meta_df = data_dict[id]['metadata'].loc[mask_chl_blank_dict[id]]
    
#     no3_conc_norm_df = no3_conc_df
#     times = no3_conc_norm_df.columns.values.astype(float)
#     for no3_add in add_conc:
#         mask = (meta_df['Nitrate_input'] == no3_add)
#         mean_no3_value = no3_conc_df.loc[mask].iloc[:, 0].mean()
#         no3_conc_norm_df.loc[mask] = no3_conc_df.loc[mask] - mean_no3_value
#         mean_no3_conc_norm = no3_conc_norm_df.loc[mask].mean().values.astype(float)
#         sem_no3_conc_norm = no3_conc_norm_df.loc[mask].sem().values.astype(float)
        
#         ax.plot(times, mean_no3_conc_norm, linestyle='-', color=colors[add_conc.index(no3_add)], alpha = 0.8)
#         ax.fill_between(times, mean_no3_conc_norm - sem_no3_conc_norm, mean_no3_conc_norm + sem_no3_conc_norm, color=colors[add_conc.index(no3_add)], alpha=0.2)
# plt.savefig(f'plots/4.2.chl1_A(t)-A(0)_color_A_add_line_band_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)
# plt.show()
    

# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Nitrate concentration subtracted by grouped mean of $A(0)$, drug conditions | colour = nominal $I_{add}$', fontsize=16)
# for i in range(6):
#     ax = axes[i // 3, i % 3]
#     ax.set_title(batch_labels[i])
#     ax.set_xlabel('Time (hr)') if i // 3 == 1 else None
#     ax.set_ylabel(r'$A(t) - \overline{A(0)}$') if i % 3 == 0 else None

#     if i == 0:
#         custom_lines = [Line2D([0], [0], color=colors[0]),
#                         Line2D([0], [0], color=colors[1]),
#                         Line2D([0], [0], color=colors[2]),
#                         Line2D([0], [0], color=colors[3])]
#         ax.legend(custom_lines, ['0.0 mM', '0.7 mM', '1.4 mM', '2.0 mM'], title=r'Nominal $I_{add}$')

#     id = ids[i]
#     no3_conc_df = data_dict[id]['no3_conc'].loc[mask_chl_blank_dict[id]]
#     meta_df = data_dict[id]['metadata'].loc[mask_chl_blank_dict[id]]
    
#     no3_conc_norm_df = no3_conc_df
#     for no3_add in add_conc:
#         mask = (meta_df['Nitrate_input'] == no3_add)
#         mean_no3_value = no3_conc_df.loc[mask].iloc[:, 0].mean()
#         no3_conc_norm_df.loc[mask] = no3_conc_df.loc[mask] - mean_no3_value
    
#     times = no3_conc_norm_df.columns.values.astype(float)
#     for index in no3_conc_norm_df.index:
#         no3_conc = no3_conc_norm_df.loc[index].values.astype(float)
#         no2_add = meta_df.loc[index, 'Nitrite_input']
#         ax.plot(times, no3_conc, linestyle='-', color=colors[add_conc.index(no2_add)], alpha = 0.3)
# plt.savefig(f'plots/4.2.chl1_A(t)-A(0)_color_I_add_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)
# plt.show()


# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Nitrate concentration subtracted by grouped mean of $A(0)$, drug conditions'
#                 + '\n' + r'colour = nominal $I_{add}$ | line = average over each $A_{add}$| band = $\pm$ SEM')    
# for i in range(6):
#     ax = axes[i // 3, i % 3]
#     ax.set_title(batch_labels[i])
#     ax.set_xlabel('Time (hr)') if i // 3 == 1 else None
#     ax.set_ylabel(r'$A(t) - \overline{A(0)}$') if i % 3 == 0 else None

#     if i == 0:
#         custom_lines = [Line2D([0], [0], color=colors[0]),
#                         Line2D([0], [0], color=colors[1]),
#                         Line2D([0], [0], color=colors[2]),
#                         Line2D([0], [0], color=colors[3])]
#         ax.legend(custom_lines, ['0.0 mM', '0.7 mM', '1.4 mM', '2.0 mM'], title=r'Nominal $I_{add}$')

#     id = ids[i]
#     no3_conc_df = data_dict[id]['no3_conc'].loc[mask_chl_blank_dict[id]]
#     meta_df = data_dict[id]['metadata'].loc[mask_chl_blank_dict[id]]
    
#     no3_conc_norm_df = no3_conc_df
#     times = no3_conc_norm_df.columns.values.astype(float)
#     for no3_add in add_conc:
#         mask = (meta_df['Nitrate_input'] == no3_add)
#         mean_no3_value = no3_conc_df.loc[mask].iloc[:, 0].mean()
#         no3_conc_norm_df.loc[mask] = no3_conc_df.loc[mask] - mean_no3_value
    
#     for no2_add in add_conc:
#         mask = (meta_df['Nitrite_input'] == no2_add)
#         mean_no3_conc_norm = no3_conc_norm_df.loc[mask].mean().values.astype(float)
#         sem_no3_conc_norm = no3_conc_norm_df.loc[mask].sem().values.astype(float)
        
#         ax.plot(times, mean_no3_conc_norm, linestyle='-', color=colors[add_conc.index(no2_add)], alpha = 0.8)
#         ax.fill_between(times, mean_no3_conc_norm - sem_no3_conc_norm, mean_no3_conc_norm + sem_no3_conc_norm, color=colors[add_conc.index(no2_add)], alpha=0.2)

# plt.savefig(f'plots/4.2.chl1_A(t)-A(0)_color_I_add_line_band_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)
# plt.show()
