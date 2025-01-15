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

# set filepath for data
filepath = '202503'

# create x axis from time points
datetime_array = [
    datetime(2024, 1, 1, 17, 35),       # T0
    datetime(2024, 1, 1, 23, 5),       # T1
    datetime(2024, 1, 2, 11, 10),       # T2
    datetime(2024, 1, 2, 15, 40),       # T3
    datetime(2024, 1, 2, 22, 35),       # T4
    datetime(2024, 1, 3, 8, 00),       # T5
    datetime(2024, 1, 3, 16, 15),       # T6
    datetime(2024, 1, 3, 21, 40),       # T7
    datetime(2024, 1, 4, 9, 40)]       # T8

times = [0]
for i in range(1, len(datetime_array)):
    time_diff = datetime_array[i] - datetime_array[0]
    times.append(time_diff.total_seconds() / 3600)

# load file names
std_meta_fn = f'{filepath}/standard_metadata.csv'
std_no2_540_fn = glob.glob(f"{filepath}/*_Ik_NO2_standard_540*")[0]
std_no2_900_fn = glob.glob(f"{filepath}/*_Ik_NO2_standard_900*")[0]
std_no2no3_540_fn = glob.glob(f"{filepath}/*_Ik_NO2NO3_standard_540*")[0]
std_no2no3_900_fn = glob.glob(f"{filepath}/*_Ik_NO2NO3_standard_900*")[0]

meta_fn = f'{filepath}/sample_metadata.csv'
no2_540_fns = sorted(glob.glob(f'{filepath}/*_Ik_NO2_tp*_540*'))
no2_900_fns = sorted(glob.glob(f'{filepath}/*_Ik_NO2_tp*_900*'))
no2no3_540_fns = sorted(glob.glob(f'{filepath}/*_Ik_NO2NO3_tp*_540*'))
no2no3_900_fns = sorted(glob.glob(f'{filepath}/*_Ik_NO2NO3_tp*_900*'))

# fitted parameters
[[no2_blank, no2no3_blank], g_fit, v_fit, no3_fit] = gr.fit_griess(meta_fn = std_meta_fn, no2_540_fn=std_no2_540_fn, no2_900_fn=std_no2_900_fn, no2no3_540_fn = std_no2no3_540_fn, no2no3_900_fn = std_no2no3_900_fn)

# create times series dataframe
no2_time_series_dic = {}
no3_time_series_dic = {}
col_num = 0
for no2_540_fn, no2_900_fn, no2no3_540_fn, no2no3_900_fn in zip(no2_540_fns, no2_900_fns, no2no3_540_fns, no2no3_900_fns):
    df = gr.get_concentration(no2_blank=no2_blank, no2no3_blank=no2no3_blank, g_fit=g_fit, v_fit=no3_fit,
                          meta_fn=meta_fn, no2_fn=None, no2_540_fn=no2_540_fn, no2_900_fn=no2_900_fn,
                          no2no3_fn=None, no2no3_540_fn=no2no3_540_fn, no2no3_900_fn=no2no3_900_fn)
    no2_time_series_dic[col_num] = df['NO2_mM'].copy()
    no3_time_series_dic[col_num] = df['NO3_mM'].copy()
    col_num += 1

no2_time_series = pd.DataFrame(no2_time_series_dic)
no3_time_series = pd.DataFrame(no3_time_series_dic)
meta = pd.read_csv(meta_fn, index_col=0).dropna(how='all')
meta.index.name = None  # Remove the name of the index

# Create a new folder to save the PNG files
output_folder = f'{filepath}/plots'
os.makedirs(output_folder, exist_ok=True)

# List of row names in the order you want to display them
rows_to_plot = ['A01', 'A03', 'A05', 'A07', 'A09', 'A11', 'A02', 'A04', 'A06', 'A08', 'A10', 'A12',
                'B01', 'B03', 'B05', 'B07', 'B09', 'B11', 'B02', 'B04', 'B06', 'B08', 'B10', 'B12',
                'C01', 'C03', 'C05', 'C07', 'C09', 'C11', 'C02', 'C04', 'C06', 'C08', 'C10', 'C12',
                'D01', 'D03', 'D05', 'D07', 'D09', 'D11', 'D02', 'D04', 'D06', 'D08', 'D10', 'D12',
                'E01', 'E03', 'E05', 'E07', 'E09', 'E11', 'E02', 'E04', 'E06', 'E08', 'E10', 'E12',
                'F01', 'F03', 'F05', 'F07', 'F09', 'F11', 'F02', 'F04', 'F06', 'F08', 'F10', 'F12',
                'G01', 'G03', 'G05', 'G07', 'G09', 'G11', 'G02', 'G04', 'G06', 'G08', 'G10', 'G12',
                'H01', 'H03', 'H05', 'H07', 'H09', 'H11', 'H02', 'H04', 'H06', 'H08', 'H10', 'H12']

# Number of rows and columns for the subplots grid
nrows = 4
ncols = 4
plots_per_figure = nrows * ncols

# Determine the global y-axis limits
all_values = pd.concat([no2_time_series, no3_time_series])
y_min = all_values.min().min()
y_max = all_values.max().max()

# Function to create a figure for a chunk of rows
def create_figure(rows_chunk, figure_index):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

    for i in range(0, len(rows_chunk), 3):
        if i // 3 < len(axes):  # Ensure we don't exceed the number of subplots
            ax = axes[i // 3]
            for j in range(3):
                if i + j < len(rows_chunk):
                    row = rows_chunk[i + j]
                    ax.plot(times, no2_time_series.loc[row], 'r.-', label='NO2')
                    ax.plot(times, no3_time_series.loc[row], 'b.-', label='NO3')
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
    plt.savefig(f'{output_folder}/time_series_overlap_{figure_index}.png')
    plt.close()

# Split the rows into chunks and create figures for each chunk
for i in range(0, len(rows_to_plot), plots_per_figure * 3):
    rows_chunk = rows_to_plot[i:i + plots_per_figure * 3]
    create_figure(rows_chunk, i // (plots_per_figure * 3))

print(f'Plots saved in folder: {output_folder}')