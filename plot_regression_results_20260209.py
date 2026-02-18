import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List

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

def calculate_mean_and_var(
        filepath: str,
        groupby_cols: List[str] = ['Nitrite_input', 'Nitrate_input', 'Chloramphenicol'],
        drop_cols: List[str] = ['Sample_type'],
        ):
    regression_results_df = pd.read_csv(f'{filepath}.csv', index_col=0)
    # Consider only rows where all drop_cols are NaN (i.e., exclude rows with specific Sample_type)
    regression_results_df = regression_results_df[regression_results_df[drop_cols].isnull().all(axis=1)]
    mean_df = regression_results_df.drop(drop_cols, axis=1).groupby(groupby_cols, as_index=False).mean()
    var_df = regression_results_df.drop(drop_cols, axis=1).groupby(groupby_cols, as_index=False).var()
    mean_and_var = mean_df.merge(var_df, on=groupby_cols, suffixes=('_mean', '_var'))
    mean_and_var = mean_and_var.sort_values(['Chloramphenicol', 'Nitrate_input', 'Nitrite_input'], ascending=False)
    mean_and_var.to_csv(f"{filepath}_mean_var.csv", index=False)
    print(f"Mean and variance of regression results saved to {filepath}_mean_var.csv")

def heatmap_for_col(
        input_fn: str,
        chl: int,
        col_label: str,
        conv_factor: float,
        output_fn: str,
        x_axis: str = 'Nitrate_input',
        y_axis: str = 'Nitrite_input',
        show_plot: bool = False
        ):
    regression_results_df = pd.read_csv(input_fn)

    # Function to create pivot table
    def create_pivot_table(df, chl, col_label, conv_factor, suffix):
        return df.query(f'Chloramphenicol == {chl}').pivot(
            index=y_axis, 
            columns=x_axis, 
            values=f'{col_label}_{suffix}'
        ).sort_index(ascending=False).sort_index(axis=1, ascending=False) * conv_factor
    
    pivot_table_mean = create_pivot_table(regression_results_df, chl, col_label, conv_factor, suffix='mean')
    pivot_table_std = create_pivot_table(regression_results_df, chl, col_label, conv_factor, suffix='var').pow(0.5)
    pivot_table_annot = pivot_table_mean.round(2).astype(str) + "\n±" + pivot_table_std.round(2).astype(str)

    sns.heatmap(pivot_table_mean, annot=pivot_table_annot, fmt='', cmap='PiYG')
    plt.title(f'{output_fn} x {conv_factor}', fontsize=16)
    plt.xlabel('Nitrate Input (mM)', fontsize=14)
    plt.ylabel('Nitrite Input (mM)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f'plots/{output_fn}_heatmap.png', dpi=300, bbox_inches='tight')    
    print(f'Saved plots/{output_fn}_heatmap.png')
    if show_plot:
        plt.show()
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


model_fit_results_df = pd.concat([pd.read_csv(f'fitting_results/4.2.batch{i}.chl0_model3_four_cond.csv') for i in range(1, number_of_batches + 1)])
model_fit_results_df = model_fit_results_df.iloc[:, 1:]
for col in model_fit_results_df.columns:
    if col in ['K_A', 'K_I']:
        plt.plot(water_contents, model_fit_results_df.iloc[:][col], 'o-', label=col)
plt.xlabel('Water content (%whc)')
plt.ylabel('Affinity (mM)')
plt.legend()
plt.show()


# for x, x_label, x_fn in [[time, 'Time (days)', 'time'], [water_contents, 'Water content (%whc)', 'water_content']]:
#     for key, y_label, color in [['no2_init_rate', 'NO2 reduction rate (mM/hour)', 'red'], ['no3_init_rate', 'NO3 reduction rate (mM/hour)', 'blue']]:
#         for chl in [0, 1]:
#             dfs = [linear_regression_results_dfs[i][linear_regression_results_dfs[i]['Chloramphenicol'] == chl][['Nitrite_input', 'Nitrate_input', f'{key}_mean', f'{key}_var']] for i in range(len(linear_regression_results_dfs))]
#             mean_plot = dfs[0][['Nitrite_input', 'Nitrate_input']].copy()
#             std_plot = dfs[0][['Nitrite_input', 'Nitrate_input']].pow(0.5).copy()
#             for i in range(len(dfs)):
#                 mean_plot[time[i]] = dfs[i][f'{key}_mean']
#                 std_plot[time[i]] = dfs[i][f'{key}_var']
#             plot_regression_results(x, mean_plot, std_plot, x_label, y_label, x_fn, color, key, chl, show_plot=False, save_plot=True)
    # for key, y_label in [['no2_second_coef', 'mM/hour^2'], ['no2_first_coef', 'mM/hour'], ['no2_rate_first_half', 'mM/hour'], ['no2_rate_second_half', 'mM/hour'], ['no3_second_coef', 'mM/hour^2'], ['no3_first_coef', 'mM/hour'], ['no3_rate_first_half', 'mM/hour'], ['no3_rate_second_half', 'mM/hour']]:
    #     chl = 0
    #     dfs = [poly_regression_results_dfs[i][poly_regression_results_dfs[i]['Chloramphenicol'] == chl][['Nitrite_input', 'Nitrate_input', f'{key}_mean', f'{key}_var']] for i in range(len(poly_regression_results_dfs))]
    #     mean_plot = dfs[0][['Nitrite_input', 'Nitrate_input']].copy()
    #     std_plot = dfs[0][['Nitrite_input', 'Nitrate_input']].pow(0.5).copy()
    #     for i in range(len(dfs)):
    #         mean_plot[time[i]] = dfs[i][f'{key}_mean']
    #         std_plot[time[i]] = dfs[i][f'{key}_var']
    #     plot_regression_results(x, mean_plot, std_plot, x_label, y_label, x_fn, key, chl, show_plot=False, save_plot=True)


# # Create mean and variance file
# calculate_mean_and_var(filepath=f'fitting_results/{id}_conc_linear_regression_results')

# # Create heatmaps
# input_fn = f'fitting_results/{id}.chl0_polynomial_regression_results_mean_var.csv'
# chl = 0
# for col_label in ['no2_second_coef', 'no2_first_coef', 'no2_rate_first_half', 'no2_rate_second_half', 'no3_second_coef', 'no3_first_coef', 'no3_rate_first_half', 'no3_rate_second_half']:
#     if col_label in ['no2_second_coef', 'no3_second_coef']:
#         conv_factor = 1000
#     else:
#         conv_factor = 24
#     heatmap_for_col(input_fn=input_fn, chl=chl, col_label=col_label, conv_factor=conv_factor,
#         output_fn=f'{id}.chl{chl}_{col_label}',show_plot=False)

# input_fn = f'fitting_results/{id}_conc_linear_regression_results_mean_var.csv'
# conv_factor=24
# for col_label in ['no2_init_rate', 'no3_init_rate']:
#     for chl in [0, 1]:
#         heatmap_for_col(input_fn=input_fn, chl=chl, col_label=col_label, conv_factor=conv_factor,
#             output_fn=f'{id}.chl{chl}_{col_label}',show_plot=False)