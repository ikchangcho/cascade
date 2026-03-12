import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns

def plot_regression_results_in_four_by_five_grid(x, x_label, dfs_to_plot, y_label, color, show_plot=False, output_fn='', fontsize=15):
    lengths = [len(df) for df in dfs_to_plot]
    if len(set(lengths)) != 1:
        print(f"Dataframe lengths: {lengths}")
        raise ValueError(f"All dataframes in dfs_to_plot must have the same length, but got lengths: {lengths}")
    
    mean_dfs_to_plot = [df.loc[:, f'{y_label}_mean'] for df in dfs_to_plot]
    var_dfs_to_plot = [df.loc[:, f'{y_label}_var'] for df in dfs_to_plot]

    fig, axes = plt.subplots(4, 5, figsize=(25, 15))
    axes = axes.flatten()
    for i in range(1, 20):
        ax = axes[i]
        q, r = divmod(i, 5)
        mean, var = [], []
        
        if r == 0:
            for j in range(2, len(dfs_to_plot)):
                mean.append(mean_dfs_to_plot[j].iloc[4*q + r - 1])
                var.append(var_dfs_to_plot[j].iloc[4*q + r - 1])
            x_plot = x[2:len(dfs_to_plot)]
        elif r == 4:
            for j in range(2):
                mean.append(mean_dfs_to_plot[j].iloc[4*q + r - 2])
                var.append(var_dfs_to_plot[j].iloc[4*q + r - 2])
            x_plot = x[:2]
        else:
            for j in range(2):
                mean.append(mean_dfs_to_plot[j].iloc[4*q + r - 2])
                var.append(var_dfs_to_plot[j].iloc[4*q + r - 2])
            for j in range(2, len(dfs_to_plot)):
                mean.append(mean_dfs_to_plot[j].iloc[4*q + r - 1])
                var.append(var_dfs_to_plot[j].iloc[4*q + r - 1])
            x_plot = x

        ax.plot(x_plot, mean, 'o-', color=color)
        ax.errorbar(x_plot, mean, yerr=np.sqrt(var), fmt='o-', capsize=3, color=color)
        ax.set_xlim(x.min() - 0.05 * (x.max() - x.min()), x.max() + 0.05 * (x.max() - x.min()))
    
    fig.text(0.55, 0.05, f'{x_label}', ha='center', fontsize=fontsize)
    fig.text(0.08, 0.5, f'{y_label}', va='center', rotation='vertical', fontsize=fontsize)
    fig.suptitle(f'{output_fn}', fontsize=fontsize+4)
    
    if output_fn != '':
        fig.suptitle(f'{output_fn}', fontsize=20, fontweight='bold')
        plt.savefig(f'plots/{output_fn}.png', dpi=300, bbox_inches='tight')
        print(f'Saved plots/{output_fn}.png')
    if show_plot:
        plt.show()
    

def plot_regression_results_for_all_batches(x, y, std_plot, x_label, y_label, x_fn, color, key, chl, fontsize = 15, show_plot=False, save_plot=False):
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
        ax.set_xlim(x.min() - 0.01 * (x.max() - x.min()), x.max() + 0.01 * (x.max() - x.min()))

    fig.text(0.55, 0.05, f'{x_label}', ha='center', fontsize=fontsize)
    fig.text(0.145, 0.9, f'A_add = 2.0 mM', fontsize=fontsize)
    fig.text(0.35, 0.9, f'A_add = 1.4 mM', fontsize=fontsize)
    fig.text(0.55, 0.9, f'A_add = 0.7 mM', fontsize=fontsize)
    fig.text(0.75, 0.9, f'A_add = 0.0 mM', fontsize=fontsize)
    fig.text(0.08, 0.5, f'{y_label}', va='center', rotation='vertical', fontsize=fontsize)
    fig.text(0.91, 0.77, f'I_add =\n2.0 mM', fontsize=fontsize)
    fig.text(0.91, 0.575, f'I_add =\n1.4 mM', fontsize=fontsize)
    fig.text(0.91, 0.37, f'I_add =\n0.7 mM', fontsize=fontsize)
    fig.text(0.91, 0.165, f'I_add =\n0.0 mM', fontsize=fontsize)
    fig.suptitle(f'Mean values of {key} (chl{chl})', fontsize=fontsize+4)
    #plt.tight_layout(rect=[0.11, 0.1, 0.9, 0.9])
    
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(f'plots/4.2.chl{chl}_{key}_vs_{x_fn}.png', dpi=300)
        print(f'Plot saved as plots/4.2.chl{chl}_{key}_vs_{x_fn}.png')
    plt.close()

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
    datetime(2025, 11, 24, 12, 56),
    datetime(2025, 12, 1, 13, 0),
    datetime(2025, 12, 8, 9, 35),
    datetime(2025, 12, 19, 11, 0),
    datetime(2026, 1, 14, 11, 15),
    datetime(2026, 2, 23, 11, 37)]
time = [0]
for i in range(1, len(datetime_array)):
    time_diff = datetime_array[i] - datetime_array[0]
    time.append(time_diff.total_seconds() / 3600 / 24)

time = np.array(time)
water_contents = np.array([98.9, 62.5, 34.4, 7.10, 5.16, 4.74])
ids = [f'4.2.batch{i}' for i in range(1, 7)]

dfs_to_plot = [pd.read_csv(f'fitting_results/{id}_data_for_phase_diagram.csv') for id in ids]

y_label = 'frac_chl0_chl1'
colors = ['red', 'darkorange', 'green', 'blue', 'purple', 'black']
y_lim = (-5, 6)

fig, axes = plt.subplots(2, 1)
for i, df in enumerate(dfs_to_plot):
    for j, y in enumerate(df[y_label]):
        init_no3 = float((df['chl1_init_no3'].iloc[j] + df['chl0_init_no3'].iloc[j]) / 
                        np.max(df['chl1_init_no3'] + df['chl0_init_no3'])) * 0.9 + 0.1
        axes[0].plot(
            time[i], y,
            color=colors[i],
            marker='o',
            alpha=init_no3,
            markersize=8
        )
        axes[0].set_ylim(y_lim[0], y_lim[1])

        init_no2 = float((df['chl1_init_no2'].iloc[j] + df['chl0_init_no2'].iloc[j]) / 
                        np.max(df['chl1_init_no2'] + df['chl0_init_no2'])) * 0.9 + 0.1
        axes[1].plot(
            time[i], y,
            color=colors[i],
            marker='o',
            alpha=init_no2,
            markersize=8
        )
        axes[1].set_ylim(y_lim[0], y_lim[1])
axes[1].set_xlabel('Time (days)')

sm1 = plt.cm.ScalarMappable(cmap='binary', norm=plt.Normalize(vmin=np.min(df['chl1_init_no3'] + df['chl0_init_no3'])/2, vmax=np.max(df['chl1_init_no3'] + df['chl0_init_no3'])/2))
sm1.set_array([])
fig.colorbar(sm1, ax=axes[0], label='A(0) (mM)')
sm2 = plt.cm.ScalarMappable(cmap='binary', norm=plt.Normalize(vmin=np.min(df['chl1_init_no2'] + df['chl0_init_no2'])/2, vmax=np.max(df['chl1_init_no2'] + df['chl0_init_no2'])/2))
sm2.set_array([])
fig.colorbar(sm2, ax=axes[1], label='I(0) (mM)')
fig.suptitle(r'$\frac{A^-_{cons}-I^-_{cons}}{A^+_{cons}-I^+_{cons}}$', fontsize = 15)

plt.show()