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

# Function to create a figure for a chunk of rows
def time_series_plot(meta_data, no2_data, no3_data, title, filename, num_of_replicates=3):
    # Determine the global y-axis limits
    all_values = pd.concat([no2_data, no3_data])
    y_min = all_values.min().min()
    y_max = all_values.max().max()
    times = no2_data.columns.astype(float).tolist()

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
            ax.plot(times, no2_data.loc[row], 'r.-', label='NO2 (I)')
            ax.plot(times, no3_data.loc[row], 'b.-', label='NO3 (A)')

    fig.text(0.5, 0.05, 'Time (hours)', ha='center', fontsize=30)
    fig.text(0.04, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize=30)
    fig.suptitle(title, fontsize=40, y=0.95)
    handles = [plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label='NO2 (I)'),
               plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label='NO3 (A)')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.85, 0.95), fontsize=20)
    plt.subplots_adjust(hspace=0.3, bottom=0.1, left=0.1)

    # Save the figure
    plt.savefig(f'{output_folder}/{filename}.png')
    plt.close()
    print(f'Plots saved in folder: {output_folder}')


def create_phase_diagram(no2_data, no3_data, title, filename):
    plt.figure(figsize=(10, 8))
    for row in no2_data.index[::3]:
        plt.plot(no2_data.loc[row], no3_data.loc[row], 'o-', label=row)

    plt.xlabel('NO2 Concentration (mM)', fontsize=15)
    plt.ylabel('NO3 Concentration (mM)', fontsize=15)
    plt.title(f'{title}', fontsize=20)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True)
    plt.savefig(f'{output_folder}/{filename}.png')
    plt.close()
    print(f'Plots saved in folder: {output_folder}')


# Create a new folder to save the PNG files
filepath = '20250114'
output_folder = f'{filepath}/plots'
os.makedirs(output_folder, exist_ok=True)

# Load the data
meta_data = pd.read_csv(f'{filepath}/sample_metadata.csv', index_col=0).dropna(how='all')
no2_data = pd.read_csv(f'{filepath}/no2_time_series_evap.csv', index_col=0)
no3_data = pd.read_csv(f'{filepath}/no3_time_series_evap.csv', index_col=0)
times = no2_data.columns.astype(float).tolist()
title = '2025-01-14 CHL-'
filename = 'test'

# create_phase_diagram(no2_data, no3_data, title, filename)
time_series_plot(meta_data, no2_data, no3_data, title, filename)