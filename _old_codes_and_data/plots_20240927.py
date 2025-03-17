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


def six_col_plot(no2_time_series, no3_time_series, meta_df, plot_fn='plot.png'):
    num_rows = len(no2_time_series)
    num_cols = 6
    num_plots = num_rows
    num_rows_needed = (num_plots + num_cols - 1) // num_cols

    # Reset the index of meta_df so "Well" values (A01, A02, etc.) are accessible as single-level index
    meta_df = meta_df.reset_index(level=0)  # Reset the multi-level index

    fig, axes = plt.subplots(nrows=num_rows_needed, ncols=num_cols, figsize=(18, 4 * num_rows_needed), sharex=False)
    axes = axes.flatten()

    for i in range(num_plots):
        # Dynamically adjust the marker size
        marker_size = max(3, 8 - num_rows_needed)  # Adjust marker size based on the number of rows

        # Extract information from meta_df for the title
        well = no2_time_series.index[i]  # Get the 'Well' value (row index in no2_time_series)

        # Use .loc[] to access the row corresponding to 'well' in the updated meta_df
        row = meta_df.loc[meta_df['Well'] == well]
        nitrite_input = row['Nitrite_input'].values[0]
        nitrate_input = row['Nitrate_input'].values[0]
        ammonium_input = row['Ammonium_input'].values[0]  # Assuming this column exists in meta_df
        sample_type = row['Sample_type'].values[0]

        # Split title into two lines
        title_str = f'NO2={nitrite_input}, NO3={nitrate_input}\nNH4={ammonium_input}, {sample_type}'

        # Plot NO2 and NO3 for each row with dynamic marker size
        axes[i].plot(time, no2_time_series.iloc[i, :], label='NO2', marker='o', markersize=marker_size)
        axes[i].plot(time, no3_time_series.iloc[i, :], label='NO3', marker='x', markersize=marker_size)

        # Make title 1.5x larger
        axes[i].set_title(title_str, fontsize=10 * 1.5)

        # Calculate legend font size based on subplot size
        legend_fontsize = max(6, 10 - num_rows_needed)  # Adjust dynamically
        axes[i].legend(fontsize=legend_fontsize)

    # Adjust space to ensure global x and y labels are visible
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2, hspace=0.5, wspace=0.4)

    # Use tight_layout to automatically adjust the layout with a bit more padding
    plt.tight_layout(pad=5.0)

    # Set global x and y labels with half the size
    fig.text(0.5, 0.02, 'Time (hours)', ha='center', fontsize=20)
    fig.text(0.02, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize=20)

    plt.savefig(plot_fn)
    plt.show()


# figure size and font size for standard curves
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (20, 12),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

# file names
current_date = datetime.now().strftime("%Y%m%d")

#fit griess assay model using standard curves
std_meta_fn = '/Users/ik/Pycharm/cascade/data_20240914/standard_metadata.csv'
std_no2_540_fn = "data_20240914/20240914_Ik_NO2_standard_540.CSV"
std_no2_900_fn = "data_20240914/20240914_Ik_NO2_standard_900.CSV"
std_no2no3_540_fn = "data_20240914/20240914_Ik_NO2NO3_standard_540.CSV"
std_no2no3_900_fn = "data_20240914/20240914_Ik_NO2NO3_standard_900.CSV"

# # plot standard curve
# gr.plot_griess_fit(meta_fn = std_meta_fn, no2_540_fn=std_no2_540_fn, no2_900_fn=std_no2_900_fn, no2no3_540_fn = std_no2no3_540_fn, no2no3_900_fn = std_no2no3_900_fn)
# #plt.savefig(f'/Users/ik/Pycharm/cascade/plots/{current_date}_standard_no2.png')
# plt.cla()
# gr.plot_no3_fit(meta_fn = std_meta_fn, no2_540_fn=std_no2_540_fn, no2_900_fn=std_no2_900_fn, no2no3_540_fn = std_no2no3_540_fn, no2no3_900_fn = std_no2no3_900_fn)
# #plt.savefig(f'/Users/ik/Pycharm/cascade/plots/{current_date}_standard_no2no3.png')

# fitted parameters
[[no2_blank,no2no3_blank], g_fit, v_fit, no3_fit] = gr.fit_griess(meta_fn = std_meta_fn, no2_540_fn=std_no2_540_fn, no2_900_fn=std_no2_900_fn, no2no3_540_fn = std_no2no3_540_fn, no2no3_900_fn = std_no2no3_900_fn)

filepath = 'data_20240914/'
meta_fn = f'{filepath}sample_metadata.csv'
no2_540_fns = sorted(glob.glob(f'{filepath}*_Ik_NO2_time*_540*'))
no2_900_fns = sorted(glob.glob(f'{filepath}*_Ik_NO2_time*_900*'))
no2no3_540_fns = sorted(glob.glob(f'{filepath}*_Ik_NO2NO3_time*_540*'))
no2no3_900_fns = sorted(glob.glob(f'{filepath}*_Ik_NO2NO3_time*_900*'))

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
meta_df = pd.read_csv(meta_fn).dropna(how='all')
time = np.array([0, 22*60 + 35-21*60-50, 34*60 + 40-21*60-50, 39*60 + 40-21*60-50, 43*60+55-21*60-50, 58*60+30-21*60-50, 63*60+20-21*60-50, 68*60+19-21*60-50])/60
six_col_plot(no2_time_series, no3_time_series, meta_df)