import sys
import os
import pandas as pd
import numpy as np
import glob
import pickle
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from datetime import datetime
import importlib
import griess as gr
import bmgdata as bd
import denitfit as dn

# Load the data
meta_chl1 = pd.read_csv(f'20250113/sample_metadata.csv', index_col=0).dropna(how='all')
no2_chl1_evap = pd.read_csv(f'20250113/no2_chl+_evap.csv', index_col=0)
no3_chl1_evap = pd.read_csv(f'20250113/no3_chl+_evap.csv', index_col=0)
no2_chl1_cons = pd.read_csv(f'20250113/no2_chl+_cons.csv', index_col=0)
no3_chl1_cons = pd.read_csv(f'20250113/no3_chl+_cons.csv', index_col=0)
meta_chl0 = pd.read_csv(f'20250114/sample_metadata.csv', index_col=0).dropna(how='all')
no2_chl0_evap = pd.read_csv(f'20250114/no2_chl-_evap.csv', index_col=0)
no3_chl0_evap = pd.read_csv(f'20250114/no3_chl-_evap.csv', index_col=0)
no2_chl0_cons = pd.read_csv(f'20250114/no2_chl-_cons.csv', index_col=0)
no3_chl0_cons = pd.read_csv(f'20250114/no3_chl-_cons.csv', index_col=0)

# Plot all the data of no2 and no3 with a given order
def plot_all(meta_data, no2_data, no3_data, title, labels, filename, num_of_replicates=3):
    times = no2_data.columns.astype(float).tolist()
    # Determine the global y-axis limits
    all_values = pd.concat([no2_data, no3_data])
    y_min = all_values.min().min()
    y_max = all_values.max().max()
    #times = no2_data.columns.astype(float).tolist()

    # List of row names in the order you want to display them
    rows_to_plot = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
                    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12',
                    'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12',
                    'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09',
                    'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12',
                    'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12',
                    'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12',
                    'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'D10', 'D11', 'D12', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12']

    # Number of rows and columns for the subplots grid
    nrows = 7
    ncols = 5
    plots_per_figure = nrows * ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

    for ii in range(0, len(rows_to_plot), num_of_replicates):
        rows_chunk = rows_to_plot[ii:ii + num_of_replicates]
        ax = axes[ii // 3]
        initial_no2 = meta_data.loc[rows_chunk[0], 'Nitrite_input']
        initial_no3 = meta_data.loc[rows_chunk[0], 'Nitrate_input']
        initial_carbon = meta_data.loc[rows_chunk[0], 'Carbon_input']
        if initial_carbon > 0:
            ax.set_title(f'I={initial_no2}, A={initial_no3}, C={initial_carbon}', fontsize=20)
        else:
            ax.set_title(f'I={initial_no2}, A={initial_no3}', fontsize=20)

        ax.set_ylim(y_min, y_max)
        for row in rows_chunk:
            ax.plot(times, no2_data.loc[row], 'r.-')
            ax.plot(times, no3_data.loc[row], 'b.-')

    fig.text(0.5, 0.05, 'Time (hours)', ha='center', fontsize=30)
    fig.text(0.04, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize=30)
    fig.suptitle(title, fontsize=40, y=0.95)
    handles = [plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=labels[0]),
               plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=labels[1])]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.85, 0.95), fontsize=20)
    plt.subplots_adjust(hspace=0.3, bottom=0.1, left=0.1)

    # Save the figure
    plt.savefig(f'{filename}.png')
    plt.close()
    print(f'{filename}.png saved')

# plot_all(meta_chl1, no2_chl1_evap, no3_chl1_evap, f'NO2 (I), NO3 (A) Concentration (CHL+)', ['I', 'A'], '20250113/plots/no2_no3_chl+_evap')
# plot_all(meta_chl1, no2_chl1_cons, no3_chl1_cons, f'NO2 (I), NO3 (A) Consumption (CHL+)', [f'$—\Delta I(t) -\Delta A(t)$', f'$-\Delta A(t)$'], '20250113/plots/no2_no3_chl+_cons')
# plot_all(meta_chl0, no2_chl0_evap, no3_chl0_evap, f'NO2 (I), NO3 (A) Concentration (CHL-)', ['I', 'A'], '20250114/plots/no2_no3_chl-_evap')
# plot_all(meta_chl0, no2_chl0_cons, no3_chl0_cons, f'NO2 (I), NO3 (A) Consumption (CHL-)', [f'$—\Delta I(t) -\Delta A(t)$', f'$-\Delta A(t)$'], '20250114/plots/no2_no3_chl-_cons')


# Plot a given data within one figure
def overlap_plots(data, title, filename, colors, labels, num_of_replicates=3):
    times = data.columns.astype(float).tolist()
    num_of_conditions = len(data) // num_of_replicates
    fig, ax = plt.subplots(figsize=(10, 8))
    for ii in range(len(data)):
        ax.plot(times, data.iloc[ii], '.-', label=labels[ii//3], color=colors[ii//3])
    # for ii, row in enumerate(data.index):
    #     ax.plot(times, data.loc[row], '.-', label=labels[ii//num_of_replicates], color=colors[ii//num_of_replicates])
    handles = [plt.Line2D([0], [0], color=colors[i], marker='.', linestyle='-', label=labels[i]) for i in range(num_of_conditions)]
    ax.legend(handles=handles, fontsize=15)
    ax.set_xlabel('Time (hours)', fontsize=20)
    ax.set_ylabel('Concentration (mM)', fontsize=20)
    ax.grid(True)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.savefig(f'{filename}.png')
    plt.close()
    print(f'{filename}.png saved')

# # I(0) conditions
# no2_conditions = [
#     (['A01', 'A02', 'A03', 'B04', 'B05', 'B06', 'C07', 'C08', 'C09', 'E01', 'E02', 'E03', 'F04', 'F05', 'F06'], '2.0'),
#     (['A04', 'A05', 'A06', 'B07', 'B08', 'B09', 'C10', 'C11', 'C12', 'E04', 'E05', 'E06', 'F07', 'F08', 'F09'], '1.5'),
#     (['A07', 'A08', 'A09', 'B10', 'B11', 'B12', 'D01', 'D02', 'D03', 'E07', 'E08', 'E09', 'F10', 'F11', 'F12'], '1.0'),
#     (['A10', 'A11', 'A12', 'C01', 'C02', 'C03', 'D04', 'D05', 'D06', 'E10', 'E11', 'E12', 'G01', 'G02', 'G03'], '0.5'),
#     (['B01', 'B02', 'B03', 'C04', 'C05', 'C06', 'D07', 'D08', 'D09', 'F01', 'F02', 'F03', 'G04', 'G05', 'G06'], '0.0')
# ]
# # A(0) conditions
# no3_conditions = [
#     (['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03'], '2.0'),
#     (['B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06'], '1.5'),
#     (['C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09'], '1.0'),
#     (['E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12','F01', 'F02', 'F03'], '0.5'),
#     (['F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06'], '0.0')
# ]
# colors = ["#b30000", "#4421af", "#0d88e6", "#5ad45a", "#ebdc78"]
# no2_legends = ['A(0)=2.0', 'A(0)=1.5', 'A(0)=1.0', 'A(0)=0.5', 'A(0)=0.0']
# no3_legends = ['I(0)=2.0', 'I(0)=1.5', 'I(0)=1.0', 'I(0)=0.5', 'I(0)=0.0']

# for rows, no2 in no2_conditions:
#     overlap_plots(no2_chl1_evap.loc[rows], f'NO2 (I) Concentration from I(0) = {no2} (CHL+)', f'20250113/plots/no2_{no2}_chl+_evap', colors, no2_legends)
#     overlap_plots(no2_chl1_cons.loc[rows], f'NO2 (I) Consumption from I(0) = {no2} (CHL+)', f'20250113/plots/no2_{no2}_chl+_cons', colors, no2_legends)
#     overlap_plots(no2_chl0_evap.loc[rows], f'NO2 (I) Concentration from I(0) = {no2} (CHL-)', f'20250114/plots/no2_{no2}_chl-_evap', colors, no2_legends)
#     overlap_plots(no2_chl0_cons.loc[rows], f'NO2 (I) Consumption from I(0) = {no2} (CHL-)', f'20250114/plots/no2_{no2}_chl-_cons', colors, no2_legends)
#     overlap_plots(no3_chl1_cons.loc[rows], f'NO3 (A) Consumption when I(0) = {no2} (CHL+)', f'20250113/plots/no3_chl+_cons_no2_{no2}', colors, no2_legends)
#     overlap_plots(no3_chl0_cons.loc[rows], f'NO3 (A) Consumption when I(0) = {no2} (CHL-)', f'20250114/plots/no3_chl-_cons_no2_{no2}', colors, no2_legends)
    
# for rows, no3 in no3_conditions:
#     overlap_plots(no3_chl1_evap.loc[rows], f'NO3 (A) Concentration from A(0) = {no3} (CHL+)', f'20250113/plots/no3_{no3}_chl+_evap', colors, no3_legends)
#     overlap_plots(no3_chl1_cons.loc[rows], f'NO3 (A) Consumption from A(0) = {no3} (CHL+)', f'20250113/plots/no3_{no3}_chl+_cons', colors, no3_legends)
#     overlap_plots(no3_chl0_evap.loc[rows], f'NO3 (A) from A(0) = {no3} Concentration (CHL-)', f'20250114/plots/no3_{no3}_chl-_evap', colors, no3_legends)
#     overlap_plots(no3_chl0_cons.loc[rows], f'NO3 (A) from A(0) = {no3} Consumption (CHL-)', f'20250114/plots/no3_{no3}_chl-_cons', colors, no3_legends)
#     overlap_plots(no2_chl1_cons.loc[rows], f'NO2 (I) Consumption when A(0) = {no3} (CHL+)', f'20250113/plots/no2_chl+_cons_no3_{no3}', colors, no3_legends)
#     overlap_plots(no2_chl0_cons.loc[rows], f'NO2 (I) Consumption when A(0) = {no3} (CHL-)', f'20250114/plots/no2_chl-_cons_no3_{no3}', colors, no3_legends)

# carbon_conditions = [
#     (['A01', 'A02', 'A03', 'G07', 'G08', 'G09'], 2.0, 2.0),
#     (['F04', 'F05', 'F06', 'H01', 'H02', 'H03'], 2.0, 0.0),
#     (['B01', 'B02', 'B03', 'G10', 'G11', 'G12'], 0.0, 2.0),
#     (['F10', 'F11', 'F12', 'H04', 'H05', 'H06'], 1.0, 0.0),
#     (['G04', 'G05', 'G06', 'D10', 'D11', 'D12'], 0.0, 0.0),
# ]
# colors = ['r', 'orange', 'b', 'g']
# legends = ['I', f'$I_C$' , 'A', f'$A_C$']

# for rows, no2, no3 in carbon_conditions:
#     overlap_plots(pd.concat([no2_chl1_evap.loc[rows], no3_chl1_evap.loc[rows]]), 
#                   f'NO2 (I), NO3 (A) Concentration from I(0) = {no2} and A(0) = {no3}\nwith and without 1 C-mM Succinate (CHL+)', 
#                   f'20250113/plots/no2_{no2}_no3_{no3}_chl+_evap_carbon', colors, legends)
#     overlap_plots(pd.concat([no2_chl1_cons.loc[rows], no3_chl1_cons.loc[rows]]),
#                     f'NO2 (I), NO3 (A) Consumption from I(0) = {no2} and A(0) = {no3}\nwith and without 1 C-mM Succinate (CHL+)', 
#                     f'20250113/plots/no2_{no2}_no3_{no3}_chl+_cons_carbon', colors, legends)
#     overlap_plots(pd.concat([no2_chl0_evap.loc[rows], no3_chl0_evap.loc[rows]]),
#                     f'NO2 (I), NO3 (A) Concentration from I(0) = {no2} and A(0) = {no3}\nwith and without 1 C-mM Succinate (CHL-)', 
#                     f'20250114/plots/no2_{no2}_no3_{no3}_chl-_evap_carbon', colors, legends)
#     overlap_plots(pd.concat([no2_chl0_cons.loc[rows], no3_chl0_cons.loc[rows]]),
#                     f'NO2 (I), NO3 (A) Consumption from I(0) = {no2} and A(0) = {no3}\nwith and without 1 C-mM Succinate (CHL-)', 
#                     f'20250114/plots/no2_{no2}_no3_{no3}_chl-_cons_carbon', colors, legends)


# Creage phaes diagram of I(t) and A(t)
def create_phase_diagram(meta_data, no2_data, no3_data, rows, title, filename, num_of_replicates=3):
    plt.figure(figsize=(10, 8))
    times = no2_data.columns.astype(float).round(1).tolist()
    num_of_conditions = len(rows) // num_of_replicates    
    markers = ['o', 's', 'v', '^', '<', '>', 'p', 'P', '*', 'X', 'D', 'd']
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    #initial_no2_array = meta_data.loc[rows, 'Nitrite_input'].values
    #initial_no3_array = meta_data.loc[rows, 'Nitrate_input'].values
    for ii, row in enumerate(rows):
        marker = markers[ii % len(markers)]
        color = colors[ii // num_of_replicates % len(colors)]
        initial_no2 = meta_data.loc[row, 'Nitrite_input']
        initial_no3 = meta_data.loc[row, 'Nitrate_input']
        plt.plot(no2_data.loc[row], no3_data.loc[row], marker=marker, linestyle='-', color=color, label=f'({initial_no2}, {initial_no3})')
    #handles = [plt.Line2D([0], [0], marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], label=f'({initial_no2_array[i * num_of_replicates]}, {initial_no3_array[i * num_of_replicates]})') for i in range(num_of_conditions)]
    plt.legend(title='(I(0), A(0)) (mM)', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.xlabel('I (mM)', fontsize=15)
    plt.ylabel('A (mM)', fontsize=15)
    plt.title(f'{title}\nTimes: {times}', fontsize=15)
    plt.grid(True)
    plt.savefig(f'{filename}.png', bbox_inches='tight')
    plt.close()
    print(f'{filename} saved')

# One replicate for each of every condition
chl1_subsets = [
    (['A01', 'A02', 'A03', 'B04', 'B05', 'B06', 'C07', 'C08', 'C09', 'E01', 'E02', 'E03', 'F04', 'F05', 'F06'], 'CHL+, I(0)=2.0', '20250113/plots/phase_chl+_evap_no2_2.0'),
    (['A04', 'A05', 'A06', 'B07', 'B08', 'B09', 'C10', 'C11', 'C12', 'E04', 'E05', 'E06', 'F07', 'F08', 'F09'], 'CHL+, I(0)=1.5', '20250113/plots/phase_chl+_evap_no2_1.5'),
    (['A07', 'A08', 'A09', 'B10', 'B11', 'B12', 'D01', 'D02', 'D03', 'E07', 'E08', 'E09', 'F10', 'F11', 'F12'], 'CHL+, I(0)=1.0', '20250113/plots/phase_chl+_evap_no2_1.0'),
    (['A10', 'A11', 'A12', 'C01', 'C02', 'C03', 'D04', 'D05', 'D06', 'E10', 'E11', 'E12', 'G01', 'G02', 'G03'], 'CHL+, I(0)=0.5', '20250113/plots/phase_chl+_evap_no2_0.5'),
    (['B01', 'B02', 'B03', 'C04', 'C05', 'C06', 'D07', 'D08', 'D09', 'F01', 'F02', 'F03', 'G04', 'G05', 'G06'], 'CHL+, I(0)=0.0', '20250113/plots/phase_chl+_evap_no2_0.0'),
    (['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03'], 'CHL+, A(0)=2.0', '20250113/plots/phase_chl+_evap_no3_2.0'),
    (['B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06'], 'CHL+, A(0)=1.5', '20250113/plots/phase_chl+_evap_no3_1.5'),
    (['C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09'], 'CHL+, A(0)=1.0', '20250113/plots/phase_chl+_evap_no3_1.0'),
    (['E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12','F01', 'F02', 'F03'], 'CHL+, A(0)=0.5', '20250113/plots/phase_chl+_evap_no3_0.5'),
    (['A01', 'A02', 'A03', 'G07', 'G08', 'G09', 'F04', 'F05', 'F06', 'H01', 'H02', 'H03', 'B01', 'B02', 'B03', 'G10', 'G11', 'G12', 'F10', 'F11', 'F12', 'H04', 'H05', 'H06', 'G04', 'G05', 'G06', 'D10', 'D11', 'D12'], 'CHL+, with and without Carbon', '20250113/plots/phase_chl+_evap_carbon')
]
chl0_subsets = [
    (['A01', 'A02', 'A03', 'B04', 'B05', 'B06', 'C07', 'C08', 'C09', 'E01', 'E02', 'E03', 'F04', 'F05', 'F06'], 'CHL-, I(0)=2.0', '20250114/plots/phase_chl-_evap_no2_2.0'),
    (['A04', 'A05', 'A06', 'B07', 'B08', 'B09', 'C10', 'C11', 'C12', 'E04', 'E05', 'E06', 'F07', 'F08', 'F09'], 'CHL-, I(0)=1.5', '20250114/plots/phase_chl-_evap_no2_1.5'),
    (['A07', 'A08', 'A09', 'B10', 'B11', 'B12', 'D01', 'D02', 'D03', 'E07', 'E08', 'E09', 'F10', 'F11', 'F12'], 'CHL-, I(0)=1.0', '20250114/plots/phase_chl-_evap_no2_1.0'),
    (['A10', 'A11', 'A12', 'C01', 'C02', 'C03', 'D04', 'D05', 'D06', 'E10', 'E11', 'E12', 'G01', 'G02', 'G03'], 'CHL-, I(0)=0.5', '20250114/plots/phase_chl-_evap_no2_0.5'),
    (['B01', 'B02', 'B03', 'C04', 'C05', 'C06', 'D07', 'D08', 'D09', 'F01', 'F02', 'F03', 'G04', 'G05', 'G06'], 'CHL-, I(0)=0.0', '20250114/plots/phase_chl-_evap_no2_0.0'),
    (['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03'], 'CHL-, A(0)=2.0', '20250114/plots/phase_chl-_evap_no3_2.0'),
    (['B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06'], 'CHL-, A(0)=1.5', '20250114/plots/phase_chl-_evap_no3_1.5'),
    (['C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09'], 'CHL-, A(0)=1.0', '20250114/plots/phase_chl-_evap_no3_1.0'),
    (['E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12','F01', 'F02', 'F03'], 'CHL-, A(0)=0.5', '20250114/plots/phase_chl-_evap_no3_0.5'),
    (['A01', 'A02', 'A03', 'G07', 'G08', 'G09', 'F04', 'F05', 'F06', 'H01', 'H02', 'H03', 'B01', 'B02', 'B03', 'G10', 'G11', 'G12', 'F10', 'F11', 'F12', 'H04', 'H05', 'H06', 'G04', 'G05', 'G06', 'D10', 'D11', 'D12'], 'CHL-, with and without Carbon', '20250114/plots/phase_chl-_evap_carbon')
]

create_phase_diagram(meta_chl1, no2_chl1_evap, no3_chl1_evap, no2_chl1_evap.index[0:76:3], 'CHL+, one replicate for each of every condition', '20250113/plots/phase_chl+_evap_all', 1)
for rows, title, filename in chl1_subsets:
    create_phase_diagram(meta_chl1, no2_chl1_evap, no3_chl1_evap, rows, title, filename)
create_phase_diagram(meta_chl0, no2_chl0_evap, no3_chl0_evap, no2_chl0_evap.index[0:76:3], 'CHL-, one replicate for each of every condition', '20250114/plots/phase_chl-_evap_all', 1)
for rows, title, filename in chl0_subsets:
    create_phase_diagram(meta_chl0, no2_chl0_evap, no3_chl0_evap, rows, title, filename)