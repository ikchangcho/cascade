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

epsilon = 0.05
no2_accum_dict = {}
for id in ids:
    no2_accum_dict[id] = []
    for index in data_dict[id]['no3_conc'].loc[mask_chl_blank_dict[id]].index:
        no3_conc = data_dict[id]['no3_conc'].loc[index].values.astype(float)
        no2_conc = data_dict[id]['no2_conc'].loc[index].values.astype(float)
        
        first_below_epsilon = np.argmax(no3_conc < epsilon) if np.any(no3_conc < epsilon) else -1
        no2_accum = no2_conc[-1] - no2_conc[first_below_epsilon] if first_below_epsilon != -1 else np.nan
        no2_accum_dict[id].append(no2_accum)
no2_accum_median = {id: np.nanmedian(no2_accum_dict[id]) for id in ids}
no2_accum_std = {id: np.nanstd(no2_accum_dict[id]) for id in ids}
no2_accum_size = {id: np.sum(~np.isnan(no2_accum_dict[id])) for id in ids}

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_title('Accumulated nitrite after nitrate depletion (CHL+)')
ax.set_xticks(range(6))
xlabels = [f'{batch_label}' + '\n' + f'(n={no2_accum_size[id]})' for batch_label, id in zip(batch_labels, ids)]
ax.set_xticklabels(xlabels)
ax.set_ylabel('Accumulated nitrite (mM)')

for i, id in enumerate(ids):
    # ax.scatter(i, no2_accum_median[id], color='white', edgecolor='black', s=100, zorder=3, marker='D', alpha=0.5)
    for no2_accum in no2_accum_dict[id]:
        if not np.isnan(no2_accum):
            ax.scatter(i, no2_accum, color=batch_colors[i], alpha=0.5)
ax.axhline(0, color='black', linestyle='--', linewidth=1, zorder=2)
plt.savefig(f'plots/accumulated_nitrite_after_nitrate_depletion_CHL+_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')