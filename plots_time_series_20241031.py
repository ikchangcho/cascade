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
filepath = 'data_20240914'

# create x axis from time points
datetime_array = [
    datetime(2024, 1, 1, 21, 50),       # T0
    datetime(2024, 1, 1, 22, 35),       # T1
    datetime(2024, 1, 2, 10, 40),       # T2
    datetime(2024, 1, 2, 15, 40),       # T3
    datetime(2024, 1, 2, 19, 55),       # T4
    datetime(2024, 1, 3, 10, 30),       # T5
    datetime(2024, 1, 3, 15, 20),       # T6
    datetime(2024, 1, 3, 20, 19)]

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
no2_540_fns = sorted(glob.glob(f'{filepath}/*_Ik_NO2_time*_540*'))
no2_900_fns = sorted(glob.glob(f'{filepath}/*_Ik_NO2_time*_900*'))
no2no3_540_fns = sorted(glob.glob(f'{filepath}/*_Ik_NO2NO3_time*_540*'))
no2no3_900_fns = sorted(glob.glob(f'{filepath}/*_Ik_NO2NO3_time*_900*'))

# fitted parameters
[[no2_blank,no2no3_blank], g_fit, v_fit, no3_fit] = gr.fit_griess(meta_fn = std_meta_fn, no2_540_fn=std_no2_540_fn, no2_900_fn=std_no2_900_fn, no2no3_540_fn = std_no2no3_540_fn, no2no3_900_fn = std_no2no3_900_fn)

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

# Evaporation correction
# Step 1: Identify rows where 'Sample_type' is 'Nitrite_Blank' or 'Nitrate_Blank'
nitrite_blank_rows = meta[meta['Sample_type'] == 'Nitrite_Blank'].index
nitrate_blank_rows = meta[meta['Sample_type'] == 'Nitrate_Blank'].index

# Step 2: Normalize the rows in no2_time_series and no3_time_series
def normalize_rows(df, rows):
    normalized_arrays = []
    for row in rows:
        normalized_array = df.loc[row] / df.loc[row].iloc[0]
        normalized_arrays.append(normalized_array)
    return normalized_arrays

no2_normalized_arrays = normalize_rows(no2_time_series, nitrite_blank_rows)
no3_normalized_arrays = normalize_rows(no3_time_series, nitrate_blank_rows)

# Step 3: Calculate the average of the normalized rows
no2_average_normalized = np.mean(no2_normalized_arrays, axis=0)
no3_average_normalized = np.mean(no3_normalized_arrays, axis=0)

# Step 4: Take the reciprocal of the average array and multiply it with each row of the original DataFrames
no2_correction_factor = 1 / no2_average_normalized
no3_correction_factor = 1 / no3_average_normalized

no2_time_series_evap = no2_time_series.apply(lambda row: row * no2_correction_factor, axis=1)
no3_time_series_evap = no3_time_series.apply(lambda row: row * no3_correction_factor, axis=1)

# Create a new folder to save the PNG files
output_folder = f'{filepath}/plots'
os.makedirs(output_folder, exist_ok=True)

# Loop through each row of the DataFrames
for row in no2_time_series_evap.index:
    plt.figure()
    
    # Plot no2_time_series_evap in red
    plt.plot(times, no2_time_series_evap.loc[row], 'r.-', label='NO2')
    
    # Plot no3_time_series_evap in blue
    plt.plot(times, no3_time_series_evap.loc[row], 'b.-', label='NO3')
    
    # Add title and labels
    plt.title(f'{row}')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mM)')
    plt.legend()
    
    # Save the plot as a PNG file
    plt.savefig(os.path.join(output_folder, f'{row}.png'))
    plt.close()

print(f'Plots saved in folder: {output_folder}')

# List of row names in the order you want to display them
rows_to_plot = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
                'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12']

# Number of rows and columns for the subplots grid
nrows = 4
ncols = 6

# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15))
axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

# Loop through the list of row names and plot each one in the corresponding subplot
for i, row in enumerate(rows_to_plot):
    if i < len(axes):  # Ensure we don't exceed the number of subplots
        ax = axes[i]
        ax.plot(times, no2_time_series_evap.loc[row], 'r.-', label='NO2')
        ax.plot(times, no3_time_series_evap.loc[row], 'b.-', label='NO3')
        ax.set_title(f'{row}')
        ax.legend().set_visible(False)  # Hide individual legends

# Add a single legend for the entire figure
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize=14)

# Add a single set of x and y labels for the entire figure
fig.text(0.5, 0.04, 'Time (hours)', ha='center', fontsize=16)
fig.text(0.04, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize=16)

# Adjust layout to prevent overlap and set custom spacing
plt.subplots_adjust(hspace=0.4, bottom=0.1, left=0.1)
plt.savefig(f'{output_folder}/time_series.png')

# Show the figure
plt.show()