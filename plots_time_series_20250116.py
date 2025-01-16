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
filepath = '20250113'
datetime_array = [
    datetime(2024, 12, 24, 23, 30),       # T0
    datetime(2024, 12, 25, 8, 0),       # T1
    datetime(2024, 12, 25, 17, 38),       # T2
    datetime(2024, 12, 26, 17, 43),       # T3
    datetime(2024, 12, 27, 17, 50),       # T4
    datetime(2024, 12, 28, 18, 10),       # T5
    datetime(2024, 12, 29, 17, 30),       # T6
    datetime(2024, 12, 30, 17, 16),       # T7
    datetime(2024, 12, 31, 17, 30)]       # T8

# # set filepath for data
# filepath = '20250114'
# datetime_array = [
#     datetime(2025, 1, 1, 14, 30),       # T0
#     datetime(2025, 1, 1, 15, 35),       # T1
#     datetime(2025, 1, 1, 16, 55),       # T2
#     datetime(2025, 1, 1, 20, 00),       # T3
#     datetime(2025, 1, 1, 22, 58),       # T4
#     datetime(2025, 1, 2, 8, 9),       # T5
#     datetime(2025, 1, 2, 13, 10),       # T6
#     datetime(2025, 1, 2, 17, 56),       # T7
#     datetime(2025, 1, 2, 22, 57),       # T8
#     datetime(2025, 1, 3, 8, 5)]       # T9


times = [0]
for i in range(1, len(datetime_array)):
    time_diff = datetime_array[i] - datetime_array[0]
    times.append(time_diff.total_seconds() / 3600)

# load file names
std_meta_fn = f'{filepath}/standard_metadata.csv'
std_no2_540_fn = glob.glob(f"{filepath}/*_Ik_STD_NO2_540*")[0]
std_no2_900_fn = glob.glob(f"{filepath}/*_Ik_STD_NO2_900*")[0]
std_no2no3_540_fn = glob.glob(f"{filepath}/*_Ik_STD_NO2NO3_540*")[0]
std_no2no3_900_fn = glob.glob(f"{filepath}/*_Ik_STD_NO2NO3_900*")[0]

meta_fn = f'{filepath}/sample_metadata.csv'
no2_540_fns = sorted(glob.glob(f'{filepath}/*_Ik_NO2_*_tp*_540*'))
no2_900_fns = sorted(glob.glob(f'{filepath}/*_Ik_NO2_*_tp*_900*'))
no2no3_540_fns = sorted(glob.glob(f'{filepath}/*_Ik_NO2NO3_*_tp*_540*'))
no2no3_900_fns = sorted(glob.glob(f'{filepath}/*_Ik_NO2NO3_*_tp*_900*'))

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

no2_time_series.to_csv(f'{filepath}/no2_time_series.csv')
no3_time_series.to_csv(f'{filepath}/no3_time_series.csv')

def no2_evap_correction(rows):
    # Extract the rows from no2_time_series using .loc
    selected_rows = no2_time_series.loc[rows]
    
    # Calculate the average of the selected rows
    avg_array = selected_rows.mean(axis=0)
    
    # Normalize the array by dividing by the first value
    normalization_factors = avg_array / avg_array.iloc[0]
    
    # Apply the normalization factors to each column of no2_time_series
    no2_time_series_evap = no2_time_series / normalization_factors
    
    return no2_time_series_evap

def no3_evap_correction(rows):
    # Extract the rows from no3_time_series using .loc
    selected_rows = no3_time_series.loc[rows]
    
    # Calculate the average of the selected rows
    avg_array = selected_rows.mean(axis=0)
    
    # Normalize the array by dividing by the first value
    normalization_factors = avg_array / avg_array.iloc[0]
    
    # Apply the normalization factors to each column of no3_time_series
    no3_time_series_evap = no3_time_series / normalization_factors
    
    return no3_time_series_evap

# Function to create a figure for a chunk of rows
def create_figure(no2_data, no3_data, rows_chunk, figure_index, filename):
    # Determine the global y-axis limits
    all_values = pd.concat([no2_data, no3_data])
    y_min = all_values.min().min()
    y_max = all_values.max().max()

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15))
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

no2_time_series_evap = no2_evap_correction(['H10', 'H11'])
no3_time_series_evap = no3_evap_correction(['H07', 'H08', 'H09'])

no2_time_series_evap.to_csv(f'{filepath}/no2_time_series_evap.csv')
no3_time_series_evap.to_csv(f'{filepath}/no3_time_series_evap.csv')

# Create a new folder to save the PNG files
output_folder = f'{filepath}/plots'
os.makedirs(output_folder, exist_ok=True)

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

for i in range(0, len(rows_to_plot), plots_per_figure * 3):
    rows_chunk = rows_to_plot[i:i + plots_per_figure * 3]
    create_figure(no2_data=no2_time_series_evap, no3_data=no3_time_series_evap, rows_chunk=rows_chunk, figure_index=i // (plots_per_figure * 3), filename='time_series_evap')

print(f'Plots saved in folder: {output_folder}')