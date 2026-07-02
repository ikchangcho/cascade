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

# Normalized cumulative consumption
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(r'Normalized cumulative consumption of nitrite $\Delta B \equiv \Delta I + \Delta A$ - individual replicates'
                 + '\n' + r'$t_{last}$ = last measured time | diagonal = linear model')

for i in range(6):
    ax = axes[i // 3, i % 3]
    ax.set_title(batch_labels[i])
    ax.set_xlabel(r'$\frac{t}{t_{last}}$') if i // 3 == 1 else None
    ax.set_ylabel(r'$\tilde{C}(t) = \frac{B(0) - B(t)}{B(0) - B(t_{last})}$') if i % 3 == 0 else None
    if i == 0:
        custom_lines = [Line2D([0], [0], color='black', linestyle='--')]
        ax.legend(custom_lines, ['Linear'])


    id = ids[i]
    df = data_dict[id]['no2_cons'].loc[mask_chl_blank_dict[id]]
    time = df.columns.values.astype(float)
    count = 0
    for well in df.index:
        y = df.loc[well].values.astype(float)
        



    df = data_dict[id]['no2_cons'].loc[mask_chl_blank_dict]
    time = df.columns.astype(float)
    count = 0
    for well in df.index:
        y = df.loc[well].values.astype(float)
        thrsh_init = 0.3
        if y[0] < thrsh_init:
            continue
        
        thrsh_end = 0.05
        zero_indices = np.where(y < thrsh_end)[0]
        if len(zero_indices) > 0:
            last_index = zero_indices[0]
        else:
            last_index = len(y) - 1

        y = y[:last_index + 1]
        y = (y[0] - y) / (y[0] - y[-1]) 
        x = time[:last_index + 1] / time[last_index]
        
        ax.plot(x, y, color=colors[i], linestyle='-', alpha=0.5)
        count += 1
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=1.0, label='Linear')
    ax.set_title(batch_labels[i] + f' (n={count})', color=colors[i])
    if i == 0:
        ax.legend(loc='upper left')
    if i  % 3 == 0:
        ax.set_ylabel(r'$\tilde{C}(t) = \frac{A(0) - A(t)}{A(0) - A(t_{last})}$')
    if i // 3 == 1:
        ax.set_xlabel(r'$\frac{t}{t_{last}}$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
plt.savefig(f'plots/figure3_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
plt.show()