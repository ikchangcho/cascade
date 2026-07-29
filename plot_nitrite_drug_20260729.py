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
    
n = 6  # Number of time points to consider for linear regression
count_dict = {}
for id in ids:
    count_dict[id] = {}
    df = data_dict[id]['no2_cons'].loc[mask_chl_blank_dict[id]]
    data_dict[id]['no2_cons_rate'] = pd.DataFrame(index=df.index)
    for index in df.index:
        y = df.loc[index].values
        x = df.columns.values.astype(float)
        if np.sum(y < 0) < 5:
            slope, _ = curve_fit(linear_func, x, y)
            data_dict[id]['no2_cons_rate'].loc[index, 'slope'] = slope[0]
            rmse = np.sqrt(np.mean((y - linear_func(x, slope[0]))**2))
            data_dict[id]['no2_cons_rate'].loc[index, 'rmse'] = rmse
            count_dict[id]['no2_cons_rate'] = np.sum(~np.isnan(data_dict[id]['no2_cons_rate']['slope']))

    data_dict[id][f'no2_cons_rate_first{n}'] = pd.DataFrame(index=df.index)
    for index in df.index:
        y = data_dict[id]['no2_cons'].loc[index].values[:n]
        x = data_dict[id]['no2_cons'].columns.values.astype(float)[:n]
        if np.sum(y < 0) < 3:
            slope, _ = curve_fit(linear_func, x, y)
            data_dict[id][f'no2_cons_rate_first{n}'].loc[index, 'slope'] = slope[0]
            rmse = np.sqrt(np.mean((y - linear_func(x, slope[0]))**2))
            data_dict[id][f'no2_cons_rate_first{n}'].loc[index, 'rmse'] = rmse        
            count_dict[id][f'no2_cons_rate_first{n}'] = np.sum(~np.isnan(data_dict[id][f'no2_cons_rate_first{n}']['slope']))
    
    data_dict[id][f'no2_cons_rate_last{n}'] = pd.DataFrame(index=df.index)
    for index in df.index:
        y_last = data_dict[id]['no2_cons'].loc[index].values[-n:]
        x_last = data_dict[id]['no2_cons'].columns.values.astype(float)[-n:]
        if np.sum(y_last < 0) < 4:
            slope_last, intercept_last, _, _, _ = stats.linregress(x_last, y_last)
            data_dict[id][f'no2_cons_rate_last{n}'].loc[index, 'slope'] = slope_last
            rmse = np.sqrt(np.mean((y_last - (slope_last * x_last + intercept_last))**2))
            data_dict[id][f'no2_cons_rate_last{n}'].loc[index, 'rmse'] = rmse
            count_dict[id][f'no2_cons_rate_last{n}'] = np.sum(~np.isnan(data_dict[id][f'no2_cons_rate_last{n}']['slope']))


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i, id in enumerate(ids):
    mask = mask_chl_blank_dict[id]
    df = data_dict[id]['no2_cons'].loc[mask]
    for index in df.index:
        y = df.loc[index].iloc[-1]
        no2_add = data_dict[id]['metadata'].loc[index, 'Nitrite_input']
        marker = markers[add_conc.index(no2_add)]
        ax.scatter(i + 0.1 * (add_conc.index(no2_add) - 1.0), y, color=batch_colors[i], marker=marker, alpha=0.5)
        y_med = df.iloc[:, -1].median()
        ax.scatter(i, y_med, color='white', edgecolors='black', marker='D', s=100, alpha = 0.7)
ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.set_title(f'Nitrite total consumption (CHL+)' + '\n' + r'colour = batch | marker = $I_{add}$ | diamond = batch median', fontsize=14)
ax.set_xticks(range(6))
ax.set_xticklabels([f'{batch_label}' for batch_label in batch_labels])
ax.set_ylabel(f'Total consumption (mM)')
custom_lines = [Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=8, label=f'$I_{{add}}$ = {conc} mM') for marker, conc in zip(markers, add_conc)]
legend = ax.legend(handles=custom_lines)
ax.add_artist(legend)
plt.savefig(f'plots/4.2.chl1_no2_cons_total_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i, id in enumerate(ids):
    mask = mask_chl_blank_dict[id]
    df = data_dict[id][f'no2_cons_rate_first{n}'].loc[mask, 'slope']
    for index in df.index:
        y = df.loc[index]
        no2_add = data_dict[id]['metadata'].loc[index, 'Nitrite_input']
        marker = markers[add_conc.index(no2_add)]
        ax.scatter(i + 0.1 * (add_conc.index(no2_add) - 1.0), y, color=batch_colors[i], marker=marker, alpha=0.5)
        y_med = df.median()
        ax.scatter(i, y_med, color='white', edgecolors='black', marker='D', s=100, alpha = 0.7)
ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.set_title(f'Nitrite consumption rate for the first {n} time points (CHL+)' + '\n' + r'colour = batch | marker = $I_{add}$ | diamond = batch median', fontsize=14)
ax.set_xticks(range(6))
ax.set_xticklabels([f'{batch_label}\n(N={batch_count})' for (batch_label, batch_count) in zip(batch_labels, [count_dict[id][f'no2_cons_rate_first{n}'] for id in ids])])
ax.set_ylabel(f'Consumption rate (mM/h)')
custom_lines = [Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=8, label=f'$I_{{add}}$ = {conc} mM') for marker, conc in zip(markers, add_conc)]
legend = ax.legend(handles=custom_lines)
ax.add_artist(legend)
plt.savefig(f'plots/4.2.chl1_no2_cons_rate_first{n}_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i, id in enumerate(ids):
    mask = mask_chl_blank_dict[id]
    df = data_dict[id][f'no2_cons_rate_last{n}'].loc[mask, 'slope']
    for index in df.index:
        y = df.loc[index]
        no2_add = data_dict[id]['metadata'].loc[index, 'Nitrite_input']
        marker = markers[add_conc.index(no2_add)]
        ax.scatter(i + 0.1 * (add_conc.index(no2_add) - 1.0), y, color=batch_colors[i], marker=marker, alpha=0.5)
        y_med = df.median()
        ax.scatter(i, y_med, color='white', edgecolors='black', marker='D', s=100, alpha=0.7)
ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.set_title(f'Nitrite consumption rate for the last {n} time points (CHL+)' + '\n' + r'colour = batch | marker = $I_{add}$ | diamond = batch median', fontsize=14)
ax.set_xticks(range(6))
ax.set_xticklabels([f'{batch_label}\n(N={batch_count})' for (batch_label, batch_count) in zip(batch_labels, [count_dict[id][f'no2_cons_rate_last{n}'] for id in ids])])
ax.set_ylabel(f'Consumption rate (mM/h)')
custom_lines = [Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=8, label=f'$I_{{add}}$ = {conc} mM') for marker, conc in zip(markers, add_conc)]
legend = ax.legend(handles=custom_lines)
ax.add_artist(legend)
plt.savefig(f'plots/4.2.chl1_no2_cons_rate_last{n}_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i, id in enumerate(ids):
    mask = mask_chl_blank_dict[id]
    df = data_dict[id]['no2_cons_rate'].loc[mask, 'slope']
    for index in df.index:
        y = df.loc[index]
        no2_add = data_dict[id]['metadata'].loc[index, 'Nitrite_input']
        marker = markers[add_conc.index(no2_add)]
        ax.scatter(i + 0.1 * (add_conc.index(no2_add) - 1.0), y, color=batch_colors[i], marker=marker, alpha=0.5)
        y_med = df.median()
        ax.scatter(i, y_med, color='white', edgecolors='black', marker='D', s=100, alpha=0.7)
ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.set_title('Nitrite consumption rate for the whole time series (CHL+)\ncolour = batch | marker = $I_{add}$ | diamond = batch median', fontsize=14)
ax.set_xticks(range(6))
ax.set_xticklabels([f'{batch_label}\n(N={batch_count})' for (batch_label, batch_count) in zip(batch_labels, [count_dict[id]['no2_cons_rate'] for id in ids])])
ax.set_ylabel('Consumption rate (mM/h)')
custom_lines = [Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=8, label=f'$I_{{add}}$ = {conc} mM') for marker, conc in zip(markers, add_conc)]
legend = ax.legend(handles=custom_lines)
ax.add_artist(legend)
plt.savefig(f'plots/4.2.chl1_no2_cons_rate_whole_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')


# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# for i, id in enumerate(ids):
#     mask = mask_chl_blank_dict[id]
#     df_whole = data_dict[id]['no2_cons_rate'].loc[mask, 'rmse']
#     df_first = data_dict[id][f'no2_cons_rate_first{n}'].loc[mask, 'rmse']
#     df_last = data_dict[id][f'no2_cons_rate_last{n}'].loc[mask, 'rmse']

#     ax.errorbar(i - 0.1, df_whole.mean(), yerr=df_whole.sem(), fmt='o', color=batch_colors[i], alpha=1.0)
#     ax.errorbar(i, df_first.mean(), yerr=df_first.sem(), fmt='s', color=batch_colors[i], alpha=1.0)
#     ax.errorbar(i + 0.1, df_last.mean(), yerr=df_last.sem(), fmt='^', color=batch_colors[i], alpha=1.0)
# ax.set_title('RMSE of linear regression for nitrite consumption rate (CHL+)\nerror bar = mean ± sem | colour = batch | marker = time points', fontsize=14)
# ax.set_xticks(range(6))
# ax.set_xticklabels([f'{batch_label}' for batch_label in batch_labels])
# ax.set_ylabel('RMSE (mM)', fontsize=12)
# custom_lines = [Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=8, label=label) for marker, label in zip(['o', 's', '^'], ['Whole time series', f'First {n} points', f'Last {n} points'])]
# ax.add_artist(ax.legend(handles=custom_lines))

# plt.savefig(f'plots/4.2.chl1_no2_cons_rate_rmse_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')



# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# for i, id in enumerate(ids):
#     mask = mask_chl_blank_dict[id]
#     df = data_dict[id][f'no2_cons_rate_first{n}'].loc[mask, 'slope'] - data_dict[id][f'no2_cons_rate_last{n}'].loc[mask, 'slope']
#     for index in df.index:
#         y = df.loc[index]
#         no2_add = data_dict[id]['metadata'].loc[index, 'Nitrite_input']
#         marker = markers[add_conc.index(no2_add)]
#         ax.scatter(i + 0.1 * (add_conc.index(no2_add) - 1.0), y, color=batch_colors[i], marker=marker, alpha=0.5)
#         y_med = df.median()
#         ax.scatter(i, y_med, color='white', edgecolors='black', marker='D', s=100)
# ax.set_title(f'Difference on consumption rates between the first and last {n} time points\ncolour = batch | marker = $I_{{add}}$ | diamond = batch median', fontsize=14)
# ax.set_xticks(range(6))
# ax.set_xticklabels([f'{batch_label}' for batch_label in batch_labels])
# ax.set_ylabel(f'Consumption rate difference (mM/h)')
# custom_lines = [Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=8, label=f'$I_{{add}}$ = {conc} mM') for marker, conc in zip(markers, add_conc)]
# legend = ax.legend(handles=custom_lines)
# ax.add_artist(legend)

# plt.savefig(f'plots/4.2.chl1_no2_cons_rate_diff_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Linear regression fitting
# row_labels = [
#     '$A_{add}$ = 2.0 mM\n$I_{add}$ = 1.4 mM',
#     '$A_{add}$ = 1.4 mM\n$I_{add}$ = 2.0 mM',
#     '$A_{add}$ = 1.4 mM\n$I_{add}$ = 1.4 mM',
#     '$A_{add}$ = 1.4 mM\n$I_{add}$ = 0.7 mM',
#     '$A_{add}$ = 0.7 mM\n$I_{add}$ = 1.4 mM',
#     '$A_{add}$ = 1.4 mM\n$I_{add}$ = 0.0 mM',
#     '$A_{add}$ = 0.0 mM\n$I_{add}$ = 1.4 mM'
# ]

# # Create the figure and subplots
# fig, axes = plt.subplots(7, 6, figsize=(18, 21))
# fig.text(0.5, 0.93, 
#          'Linear regression of nitrite consumption, for first and last six time points\n' \
#          '\nsolid lines: first six time points | dashed lines: last six time points\n' \
#          'regression was not performed if more than half of the points were negative'
#          , ha='center', fontsize=20)

# # Add column labels as text
# for i, label in enumerate(batch_labels):
#     fig.text(0.15 + i * 0.135, 0.90, label, fontsize=15)

# # Add row labels as text
# for i, label in enumerate(row_labels):
#     fig.text(0.91, 0.82 - i * 0.112, label, fontsize=15)

# # Plot data for each subplot
# for col_idx, id in enumerate(ids):
#     for row_idx, row_label in enumerate(row_labels):
#         ax = axes[row_idx, col_idx]
#         A_add = float(row_label.split('$A_{add}$ = ')[1].split(' mM')[0])
#         I_add = float(row_label.split('$I_{add}$ = ')[1].split(' mM')[0])

#         # Filter metadata to find matching indices
#         metadata = data_dict[id]['metadata'][mask_chl_blank_dict[id]]
#         indices = metadata[
#             (metadata['Nitrate_input'] == A_add) &
#             (metadata['Nitrite_input'] == I_add)
#         ].index

#         if len(indices) != 3:
#             print(f"Warning: Expected 3 matching indices for {id} with A_add={A_add} and I_add={I_add}, but found {len(indices)}.")

#         # Plot the data for each matching index
#         for idx, marker, color in zip(indices, ['o', 's', '^'], ['blue', 'green', 'red']):
#             time = data_dict[id]['no2_cons'].columns.values.astype(float)
#             time_series = data_dict[id]['no2_cons'].loc[idx]
#             ax.scatter(
#                 time, time_series.values,
#                 marker=marker, color=color, alpha=0.5)
            
#             n = 6
#             # Perform linear regression for the first n time points, fix the intercept at 0
#             x = time[:n]
#             y = time_series.values[:n]
#             if np.sum(y < 0) < 3:
#                 slope, _ = curve_fit(linear_func, x, y)
#                 ax.plot(time, linear_func(time, slope[0]), color=color, linewidth=1.0, alpha=0.7)

#             # Perform linear regression for the last n time points
#             x_last = time[-n:]
#             y_last = time_series.values[-n:]
#             if np.sum(y_last < 0) < 4:
#                 slope_last, intercept_last, _, _, _ = stats.linregress(x_last, y_last)
#                 ax.plot(time, slope_last * time + intercept_last, color=color, linestyle='--', linewidth=1.0, alpha=0.7)

#             # Shade the entire area below y = 0
#         ax.fill_between(time, ax.get_ylim()[0], 0, color='gray', alpha=0.5)
#         ax.set_ylabel(r'$- \Delta I - \Delta A$ (mM)', fontsize=12) if col_idx == 0 else None
#         ax.set_xlabel('Time (h)', fontsize=12) if row_idx == 6 else None

# plt.savefig(f'plots/4.2.chl1_no2_cons_linear_regression_half_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)
# plt.show()