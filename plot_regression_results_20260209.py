import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns

def plot_regression_results_for_all_batches(x, y, std_plot, x_label, y_label, x_fn, color, key, chl, show_plot=False, save_plot=False):
    # Calculate y limits excluding outliers
    y_data = y.iloc[:, 2:]
    y_flat = y_data.values.flatten()
    y_flat = y_flat[~np.isnan(y_flat)]  # Remove NaN values
    
    upper_bound, lower_bound = np.percentile(y_flat, [85, 15])
    iqr = upper_bound - lower_bound
    outlier_threshold = upper_bound + 1.5 * iqr
    
    # Get min/max excluding outliers
    y_non_outliers = y_flat[y_flat <= outlier_threshold]
    y_max = y_non_outliers.max()
    y_min = y_non_outliers.min()

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    for i in range(len(y)):
        ax = axes[i+1]
        y_row = y.iloc[i, 2:]
        std_row = std_plot.iloc[i, 2:]
        
        # Plot normal points
        ax.plot(x, y_row, 'o-', color=color)
        ax.errorbar(x, y_row, yerr=np.sqrt(std_row), fmt='o-', capsize=3, color=color)
        
        # Mark outliers with arrows pointing up
        for j, (x_val, y_val) in enumerate(zip(x, y_row)):
            if pd.notna(y_val) and y_val > outlier_threshold:
                # Draw an upward arrow at the top of the plot
                ax.annotate(f'{y_val:.2f}', 
                           xy=(x_val, y_max), 
                           xytext=(x_val, y_max + 0.05 * abs(y_max)),
                           ha='center', 
                           fontsize=8,
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                           color='red')
        
        ax.set_ylim(y_min - 0.1 * abs(y_min), y_max + 0.15 * abs(y_max))  # Extra space for annotations
        ax.set_xlim(x.min(), x.max())

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

time = np.array(time)
water_contents = np.array([98.9, 62.5, 34.4, 7.10, 5.16])

# Load all batch data files
ids = [f'4.2.batch{i}' for i in range(1, 6)]

# for id in ids:
#     calculate_mean_and_var(f'fitting_results/{id}.chl0_half_life_reciprocal')

# linear_regression_results_dfs = [pd.read_csv(f'fitting_results/{id}_conc_linear_regression_results_mean_var.csv') for id in ids]
# linear_regression_results_dfs = [df.sort_values(['Chloramphenicol', 'Nitrite_input', 'Nitrate_input'], ascending=False) for df in linear_regression_results_dfs]
# poly_regression_results_dfs = [pd.read_csv(f'fitting_results/{id}.chl0_polynomial_regression_results_mean_var.csv') for id in ids]
# poly_regression_results_dfs = [df.sort_values(['Chloramphenicol', 'Nitrite_input', 'Nitrate_input'], ascending=False) for df in poly_regression_results_dfs]
# half_life_reciprocal_dfs = [pd.read_csv(f'fitting_results/{id}.chl0_half_life_reciprocal_mean_var.csv') for id in ids]
# half_life_reciprocal_dfs = [df.sort_values(['Chloramphenicol', 'Nitrite_input', 'Nitrate_input'], ascending=False) for df in half_life_reciprocal_dfs]


# for x, x_label, x_fn in [[time, 'Time (days)', 'time'], [water_contents, 'Water content (%whc)', 'water_content']]:
#     for key, y_label, color in [['no2_half_life_reciprocal', '1 / NO2 half life (1/hour)', 'red'], ['no3_half_life_reciprocal', '1 / NO3 half life (1/hour)', 'blue']]:
#         for chl in [0]:
#             dfs = [half_life_reciprocal_dfs[i][half_life_reciprocal_dfs[i]['Chloramphenicol'] == chl][['Nitrite_input', 'Nitrate_input', f'{key}_mean', f'{key}_var']] for i in range(len(half_life_reciprocal_dfs))]
#             mean_plot = dfs[0][['Nitrite_input', 'Nitrate_input']].copy()
#             std_plot = dfs[0][['Nitrite_input', 'Nitrate_input']].pow(0.5).copy()
#             for i in range(len(dfs)):
#                 mean_plot[time[i]] = dfs[i][f'{key}_mean']
#                 std_plot[time[i]] = dfs[i][f'{key}_var']
#             plot_regression_results_for_all_batches(x, mean_plot, std_plot, x_label, y_label, x_fn, color, key, chl, show_plot=False, save_plot=True)

# Create heatmaps
for id in ids:
    input_fn = f'fitting_results/{id}.chl0_half_life_reciprocal_mean_var.csv'
    chl = 0
    for col_label in ['no2_half_life_reciprocal', 'no3_half_life_reciprocal']:
        heatmap_for_col(input_fn=input_fn, chl=chl, col_label=col_label, conv_factor=24,
            output_fn=f'{id}.chl{chl}_{col_label}',show_plot=False)

# input_fn = f'fitting_results/{id}_conc_linear_regression_results_mean_var.csv'
# conv_factor=24
# for col_label in ['no2_init_rate', 'no3_init_rate']:
#     for chl in [0, 1]:
#         heatmap_for_col(input_fn=input_fn, chl=chl, col_label=col_label, conv_factor=conv_factor,
#             output_fn=f'{id}.chl{chl}_{col_label}',show_plot=False)