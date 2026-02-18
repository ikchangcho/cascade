import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def plot_regression_results(x, y, std_plot, x_label, y_label, x_fn, color, key, chl, show_plot=False, save_plot=False):
    y_max = max(y.iloc[:, 2:].max())
    y_min = min(y.iloc[:, 2:].min())

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    for i in range(len(y)):
        ax = axes[i+1]
        ax.plot(x, y.iloc[i, 2:], 'o-', color=color)
        ax.errorbar(x, y.iloc[i, 2:], yerr=np.sqrt(std_plot.iloc[i, 2:]), fmt='o-', capsize=3, color=color)
        ax.set_ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))
    fontsize = 20
    fig.text(0.55, 0.05, f'{x_label}', ha='center', fontsize=fontsize)
    fig.text(0.145, 0.9, f'A(0) = 2.0 mM', fontsize=fontsize)
    fig.text(0.35, 0.9, f'A(0) = 1.4 mM', fontsize=fontsize)
    fig.text(0.55, 0.9, f'A(0) = 0.7 mM', fontsize=fontsize)
    fig.text(0.75, 0.9, f'A(0) = 0.0 mM', fontsize=fontsize)
    fig.text(0.08, 0.5, f'{y_label}', va='center', rotation='vertical', fontsize=fontsize)
    fig.text(0.91, 0.77, f'I(0) =\n2.0 mM', fontsize=fontsize)
    fig.text(0.91, 0.575, f'I(0) =\n1.4 mM', fontsize=fontsize)
    fig.text(0.91, 0.37, f'I(0) =\n0.7 mM', fontsize=fontsize)
    fig.text(0.91, 0.165, f'I(0) =\n0.0 mM', fontsize=fontsize)
    fig.suptitle(f'Mean values of {key} (chl{chl})', fontsize=fontsize+4)
    plt.tight_layout(rect=[0.11, 0.1, 0.9, 0.9])
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(f'plots/4.2.chl{chl}_{key}_vs_{x_fn}.png', dpi=300)
        print(f'Plot saved as plots/4.2.chl{chl}_{key}_vs_{x_fn}.png')
    plt.close()

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

water_contents = [98.9, 62.5, 34.4, 7.10, 5.16]

# Load all batch data files
number_of_batches = 5
linear_regression_results_dfs = [pd.read_csv(f'fitting_results/4.2.batch{i}_conc_linear_regression_results_mean_var.csv') for i in range(1, number_of_batches + 1)]
linear_regression_results_dfs = [df.sort_values(['Chloramphenicol', 'Nitrite_input', 'Nitrate_input'], ascending=False) for df in linear_regression_results_dfs]
poly_regression_results_dfs = [pd.read_csv(f'fitting_results/4.2.batch{i}.chl0_polynomial_regression_results_mean_var.csv') for i in range(1, number_of_batches + 1)]
poly_regression_results_dfs = [df.sort_values(['Chloramphenicol', 'Nitrite_input', 'Nitrate_input'], ascending=False) for df in poly_regression_results_dfs]

for x, x_label, x_fn in [[time, 'Time (days)', 'time'], [water_contents, 'Water content (%whc)', 'water_content']]:
    for key, y_label, color in [['no2_init_rate', 'NO2 reduction rate (mM/hour)', 'red'], ['no3_init_rate', 'NO3 reduction rate (mM/hour)', 'blue']]:
        for chl in [0, 1]:
            dfs = [linear_regression_results_dfs[i][linear_regression_results_dfs[i]['Chloramphenicol'] == chl][['Nitrite_input', 'Nitrate_input', f'{key}_mean', f'{key}_var']] for i in range(len(linear_regression_results_dfs))]
            mean_plot = dfs[0][['Nitrite_input', 'Nitrate_input']].copy()
            std_plot = dfs[0][['Nitrite_input', 'Nitrate_input']].pow(0.5).copy()
            for i in range(len(dfs)):
                mean_plot[time[i]] = dfs[i][f'{key}_mean']
                std_plot[time[i]] = dfs[i][f'{key}_var']
            plot_regression_results(x, mean_plot, std_plot, x_label, y_label, x_fn, color, key, chl, show_plot=False, save_plot=True)
    # for key, y_label in [['no2_second_coef', 'mM/hour^2'], ['no2_first_coef', 'mM/hour'], ['no2_rate_first_half', 'mM/hour'], ['no2_rate_second_half', 'mM/hour'], ['no3_second_coef', 'mM/hour^2'], ['no3_first_coef', 'mM/hour'], ['no3_rate_first_half', 'mM/hour'], ['no3_rate_second_half', 'mM/hour']]:
    #     chl = 0
    #     dfs = [poly_regression_results_dfs[i][poly_regression_results_dfs[i]['Chloramphenicol'] == chl][['Nitrite_input', 'Nitrate_input', f'{key}_mean', f'{key}_var']] for i in range(len(poly_regression_results_dfs))]
    #     mean_plot = dfs[0][['Nitrite_input', 'Nitrate_input']].copy()
    #     std_plot = dfs[0][['Nitrite_input', 'Nitrate_input']].pow(0.5).copy()
    #     for i in range(len(dfs)):
    #         mean_plot[time[i]] = dfs[i][f'{key}_mean']
    #         std_plot[time[i]] = dfs[i][f'{key}_var']
    #     plot_regression_results(x, mean_plot, std_plot, x_label, y_label, x_fn, key, chl, show_plot=False, save_plot=True)
        