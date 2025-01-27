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
def create_figure(no2_data, no3_data, rows_chunk, figure_index, filename):
    # Determine the global y-axis limits
    all_values = pd.concat([no2_data, no3_data])
    y_min = all_values.min().min()
    y_max = all_values.max().max()
    times = no2_data.columns.astype(float).tolist()

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

    for i in range(0, len(rows_chunk), 3):
        if i // 3 < len(axes):  # Ensure we don't exceed the number of subplots
            ax = axes[i // 3]
            for j in range(3):
                if i + j < len(rows_chunk):
                    row = rows_chunk[i + j]
                    ax.plot(times, no2_data.loc[row], 'r.-', label='NO2')
                    ax.plot(times, no3_data.loc[row], 'b.-', label='NO3')
            ax.set_title(f'{rows_chunk[i:i+3]}')
            ax.set_ylim(y_min, y_max)
            ax.legend().set_visible(False)  # Hide individual legends

    # Add a single legend for the entire figure
    handles = [plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label='NO2'),
               plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label='NO3')]
    fig.legend(handles=handles, loc='upper right', fontsize=20)

    # Add a single set of x and y labels for the entire figure
    fig.text(0.5, 0.04, 'Time (hours)', ha='center', fontsize=30)
    fig.text(0.04, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize=30)

    # Adjust layout to prevent overlap and set custom spacing
    plt.subplots_adjust(hspace=0.4, bottom=0.1, left=0.1)

    # Save the figure
    plt.savefig(f'{output_folder}/{filename}_{figure_index}.png')
    plt.close()


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

no2_data = pd.read_csv(f'{filepath}/no2_time_series_evap.csv', index_col=0)
no3_data = pd.read_csv(f'{filepath}/no3_time_series_evap.csv', index_col=0)
times = no2_data.columns.astype(float).tolist()
title = 'CHL-'
filename = 'no2_vs_no3_chl-'

create_phase_diagram(no2_data, no3_data, title, filename)


# # List of row names in the order you want to display them
# rows_to_plot = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
#                 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12',
#                 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12',
#                 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09',
#                 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12',
#                 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12',
#                 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12',
#                 'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'D10', 'D11', 'D12', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12']

# # Number of rows and columns for the subplots grid
# nrows = 7
# ncols = 5
# plots_per_figure = nrows * ncols

# for i in range(0, len(rows_to_plot), plots_per_figure * 3):
#     rows_chunk = rows_to_plot[i:i + plots_per_figure * 3]
#     create_figure(no2_data, no3_data, rows_chunk=rows_chunk, figure_index=i // (plots_per_figure * 3), filename=filename)

# print(f'Plots saved in folder: {output_folder}')