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

epsilon = 0.05
for id in ids:
    data_dict[id]['metadata']['nitrate_depletion_index'] = np.nan
    data_dict[id]['metadata']['nitrite_cons_slope_until_depletion'] = np.nan
    mask = mask_chl_blank_dict[id]
    for index in data_dict[id]['no3_conc'].loc[mask].index:
        no3_conc = data_dict[id]['no3_conc'].loc[index].values.astype(float)
        no2_conc = data_dict[id]['no2_conc'].loc[index].values.astype(float)
        
        first_index_below_epsilon = np.argmax(no3_conc < epsilon) if np.any(no3_conc < epsilon) else len(no3_conc)
        data_dict[id]['metadata'].loc[index, 'nitrate_depletion_index'] = first_index_below_epsilon

# n = 10
# for id in ids:
#     smallest_positive_numbers = data_dict[id]['metadata']['nitrate_depletion_index']
#     smallest_positive_numbers = smallest_positive_numbers[smallest_positive_numbers > 0].nsmallest(n)
#     print(f"Smallest {n} positive numbers for {id}:")
#     print(smallest_positive_numbers)

        if first_index_below_epsilon > 4:
            regression_length = min(5, first_index_below_epsilon)
            x = data_dict[id]['no2_cons'].columns[:regression_length].values.astype(float)
            y = data_dict[id]['no2_cons'].loc[index].iloc[:regression_length].values.astype(float)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            data_dict[id]['metadata'].loc[index, 'nitrite_cons_slope_until_depletion'] = slope

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_title('Nitrite consumption rate for first five time points (CHL+)' + '\n' + r'marker = $A_{add}$ | color = batch')
ax.set_xticks(range(6))
ax.set_xticklabels([f'{batch_label}' for batch_label in batch_labels])
ax.set_ylabel('Nitrite consumption rate (mM/h)')
custom_lines = [Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=8, label=f'$A_{{add}}$ = {conc} mM') for marker, conc in zip(markers, add_conc)]
legend = ax.legend(handles=custom_lines)
ax.add_artist(legend)


for i, id in enumerate(ids):
    mask = mask_chl_blank_dict[id]
    for index in data_dict[id]['no3_conc'].loc[mask].index:
        slope = data_dict[id]['metadata'].loc[index, 'nitrite_cons_slope_until_depletion']
        if not np.isnan(slope):
            no3_add = data_dict[id]['metadata'].loc[index, 'Nitrate_input']
            marker = markers[add_conc.index(no3_add)]
            ax.scatter(i + 0.1 * (add_conc.index(no3_add) - 1.0), slope, color=batch_colors[i], marker=marker, alpha=0.5)
plt.savefig(f'plots/nitrite_consumption_rate_first_five_time_points_CHL+_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
            


