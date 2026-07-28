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
    
row_labels = [
    '$A_{add}$ = 2.0 mM\n$I_{add}$ = 1.4 mM',
    '$A_{add}$ = 1.4 mM\n$I_{add}$ = 2.0 mM',
    '$A_{add}$ = 1.4 mM\n$I_{add}$ = 1.4 mM',
    '$A_{add}$ = 1.4 mM\n$I_{add}$ = 0.7 mM',
    '$A_{add}$ = 0.7 mM\n$I_{add}$ = 1.4 mM',
    '$A_{add}$ = 1.4 mM\n$I_{add}$ = 0.0 mM',
    '$A_{add}$ = 0.0 mM\n$I_{add}$ = 1.4 mM'
]


for n in [3, 6, 9]:
    # Create the figure and subplots
    fig, axes = plt.subplots(7, 6, figsize=(18, 21))
    fig.text(0.5, 0.95, f'Linear Regression of Nitrite Consumption on Selected Conditions (First {n} Time Points)', ha='center', fontsize=20)

    # Add column labels as text
    for i, label in enumerate(batch_labels):
        fig.text(0.15 + i * 0.135, 0.90, label, fontsize=15)

    # Add row labels as text
    for i, label in enumerate(row_labels):
        fig.text(0.91, 0.82 - i * 0.112, label, fontsize=15)

    # Plot data for each subplot
    for row_idx, row_label in enumerate(row_labels):
        for col_idx, id in enumerate(ids):
            ax = axes[row_idx, col_idx]
            A_add = float(row_label.split('$A_{add}$ = ')[1].split(' mM')[0])
            I_add = float(row_label.split('$I_{add}$ = ')[1].split(' mM')[0])

            # Filter metadata to find matching indices
            metadata = data_dict[id]['metadata'][mask_chl_blank_dict[id]]
            indices = metadata[
                (metadata['Nitrate_input'] == A_add) &
                (metadata['Nitrite_input'] == I_add)
            ].index

            if len(indices) != 3:
                print(f"Warning: Expected 3 matching indices for {id} with A_add={A_add} and I_add={I_add}, but found {len(indices)}.")

            # Plot the data for each matching index
            for idx, marker, color in zip(indices, ['o', 's', '^'], ['blue', 'green', 'red']):
                time = data_dict[id]['no2_cons'].columns.values.astype(float)
                time_series = data_dict[id]['no2_cons'].loc[idx]
                ax.scatter(
                    time, time_series.values,
                    marker=marker, color=color, alpha=0.5)
                
                # Perform linear regression for the first n time points
                slope, intercept, r_value, p_value, std_err = stats.linregress(time[:n], time_series.values[:n])
                regression_line = slope * time + intercept
                ax.plot(
                    time, regression_line,
                    color=color, linestyle='--', alpha=0.8)

            ax.set_ylabel(r'$- \Delta I - \Delta A$ (mM)', fontsize=12) if col_idx == 0 else None
            ax.set_xlabel('Time (h)', fontsize=12) if row_idx == 6 else None

    plt.savefig(f'plots/4.2.chl1_no2_cons_linear_regression_{n}_time_points_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300)