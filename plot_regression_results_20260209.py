import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

datetime_array = [
    datetime(2024, 11, 24, 12, 56),
    datetime(2024, 12, 1, 13, 0),
    datetime(2024, 12, 8, 9, 35),
    datetime(2024, 12, 19, 11, 0),
    datetime(2025, 1, 14, 11, 15)]
time = [0]
for i in range(1, len(datetime_array)):
    time_diff = datetime_array[i] - datetime_array[0]
    time.append(time_diff.total_seconds() / 3600 / 24)

water_contents =  [98.9, 62.5, 34.4, 7.44, 7.10, 5.16]

# Load all batch data files
dfs = [pd.read_csv(f'fitting_results/4.2.batch{i}_linear_regression_mean_var.csv') for i in range(1, 6)]

# Filter and select columns for all batches
chl = 1
key = 'NO2 Slope'
dfs = [dfs[i][dfs[i]['Chloramphenicol'] == chl][['Nitrite_input', 'Nitrate_input', f'{key}_mean', f'{key}_var']] for i in range(len(dfs))]

mean_plot = dfs[0][['Nitrite_input', 'Nitrate_input']]
std_plot = dfs[0][['Nitrite_input', 'Nitrate_input']].pow(0.5)
for i in range(len(dfs)):
    mean_plot[time[i]] = dfs[i][f'{key}_mean']
    std_plot[time[i]] = dfs[i][f'{key}_var']

x = time
y = mean_plot
y_max = max(y.iloc[:, 2:].max())
y_min = min(y.iloc[:, 2:].min())

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()
for i in range(len(y)):
    ax = axes[i+1]
    ax.plot(x, y.iloc[i, 2:], 'o-')
    ax.errorbar(x, y.iloc[i, 2:], yerr=np.sqrt(std_plot.iloc[i, 2:]), fmt='o-', capsize=3)
    ax.set_ylim(y_min, y_max)
fontsize = 20
fig.text(0.55, 0.05, 'Time (days)', ha='center', fontsize=fontsize)
fig.text(0.145, 0.9, f'A(0) = 2.0 mM', fontsize=fontsize)
fig.text(0.35, 0.9, f'A(0) = 1.4 mM', fontsize=fontsize)
fig.text(0.55, 0.9, f'A(0) = 0.7 mM', fontsize=fontsize)
fig.text(0.75, 0.9, f'A(0) = 0.0 mM', fontsize=fontsize)
fig.text(0.08, 0.5, f'{unit}', va='center', rotation='vertical', fontsize=fontsize)
fig.text(0.91, 0.77, f'I(0) =\n2.0 mM', fontsize=fontsize)
fig.text(0.91, 0.575, f'I(0) =\n1.4 mM', fontsize=fontsize)
fig.text(0.91, 0.37, f'I(0) =\n0.7 mM', fontsize=fontsize)
fig.text(0.91, 0.165, f'I(0) =\n0.0 mM', fontsize=fontsize)
fig.suptitle(f'Mean values of {key} with error bars', fontsize=fontsize+4)
plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
plt.show()