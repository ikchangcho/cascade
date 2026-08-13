from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
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
    mask_chl_blank_dict[id] = (data_dict[id]['metadata']['Chloramphenicol'] == 0.0) & (data_dict[id]['metadata']['Sample_type'] != 'Blank')

count_dict = {}
for id in ids:
    count_dict[id] = {}
    mask = mask_chl_blank_dict[id]
    df = data_dict[id]['no3_cons'].loc[mask]
    data_dict[id]['no3_cons_norm_auc'] = pd.DataFrame(index=df.index)
    for index in df.index:
        no3_cons = df.loc[index].values
        time = df.columns.values.astype(float)
        auc = np.trapezoid(no3_cons, time)
        norm_auc = auc / time[-1]
        data_dict[id]['no3_cons_norm_auc'].loc[index, 'total'] = norm_auc
    data_dict[id]['no2_cons_norm_auc'] = pd.DataFrame(index=df.index)
    for index in df.index:
        no2_cons = data_dict[id]['no2_cons'].loc[index].values
        time = data_dict[id]['no2_cons'].columns.values.astype(float)
        auc = np.trapezoid(no2_cons, time)
        norm_auc = auc / time[-1]
        data_dict[id]['no2_cons_norm_auc'].loc[index, 'total'] = norm_auc

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i, id in enumerate(ids):
    df = data_dict[id]['no3_cons_norm_auc']
    for index in df.index:
        y = df.loc[index, 'total']
        no3_add = data_dict[id]['metadata'].loc[index, 'Nitrate_input']
        marker = markers[add_conc.index(no3_add)]
        ax.scatter(i + 0.1 * (add_conc.index(no3_add) - 1.0), y, color=batch_colors[i], marker=marker, alpha=0.5)
        y_med = df['total'].median()
        ax.scatter(i, y_med, color='white', edgecolors='black', marker='D', s=100, alpha = 0.5)
ax.set_title(f'Nitrate consumption normalized AUC (CHL-)' + '\n' + r'colour = batch | marker = $A_{add}$ | diamond = batch median', fontsize=14)
ax.set_xticks(range(6))
ax.set_xticklabels([f'{batch_label}' for batch_label in batch_labels])
ax.set_ylabel(f'Normalized AUC (mM)')
custom_lines = [Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=8, label=f'$A_{{add}}$ = {conc} mM') for marker, conc in zip(markers, add_conc)]
legend = ax.legend(handles=custom_lines)
ax.add_artist(legend)
plt.savefig(f'plots/4.2.chl0_no3_cons_norm_auc_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i, id in enumerate(ids):
    df = data_dict[id]['no2_cons_norm_auc']
    for index in df.index:
        y = df.loc[index, 'total']
        no2_add = data_dict[id]['metadata'].loc[index, 'Nitrite_input']
        marker = markers[add_conc.index(no2_add)]
        ax.scatter(i + 0.1 * (add_conc.index(no2_add) - 1.0), y, color=batch_colors[i], marker=marker, alpha=0.5)
        y_med = df['total'].median()
        ax.scatter(i, y_med, color='white', edgecolors='black', marker='D', s=100, alpha = 0.5)
ax.set_title(f'Nitrite consumption normalized AUC (CHL-)' + '\n' + r'colour = batch | marker = $A_{add}$ | diamond = batch median', fontsize=14)
ax.set_xticks(range(6))
ax.set_xticklabels([f'{batch_label}' for batch_label in batch_labels])
ax.set_ylabel(f'Normalized AUC (mM)')
custom_lines = [Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=8, label=f'$A_{{add}}$ = {conc} mM') for marker, conc in zip(markers, add_conc)]
legend = ax.legend(handles=custom_lines)
ax.add_artist(legend)
plt.savefig(f'plots/4.2.chl0_no2_cons_norm_auc_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')

