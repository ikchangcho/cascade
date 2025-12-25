import sys
sys.path.append('_functions')
print("Python path: ", sys.executable)
import os
#print("Current working directory:", os.getcwd())
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
import ammonia as am
import os
import re

def no2_no3_abs_to_conc(date, key):
    std_meta_fn = glob.glob(f'absorbances/{date}_standards_metadata.csv')[0]
    std_no2_540_fn = glob.glob(f"absorbances/{date}_Ik_STD_NO2_*540.CSV")[0]
    std_no2_900_fn = glob.glob(f"absorbances/{date}_Ik_STD_NO2_*900.CSV")[0]
    std_no2no3_540_fn = glob.glob(f"absorbances/{date}_Ik_STD_NO2NO3_*540.CSV")[0]
    std_no2no3_900_fn = glob.glob(f"absorbances/{date}_Ik_STD_NO2NO3_*900.CSV")[0]

    meta_fn = glob.glob(f'absorbances/{date}_samples_metadata.csv')[0]
    no2_540_fns = sorted(glob.glob(f'absorbances/{date}_Ik_NO2_{key}*540*'))
    no2_900_fns = sorted(glob.glob(f'absorbances/{date}_Ik_NO2_{key}*900*'))
    no2no3_540_fns = sorted(glob.glob(f'absorbances/{date}_Ik_NO2NO3_{key}*540*'))
    no2no3_900_fns = sorted(glob.glob(f'absorbances/{date}_Ik_NO2NO3_{key}*900*'))

    # fitted parameters from standards
    [[no2_blank, no2no3_blank], g_fit, v_fit, no3_fit] = gr.fit_griess(meta_fn = std_meta_fn, no2_540_fn=std_no2_540_fn, no2_900_fn=std_no2_900_fn, no2no3_540_fn = std_no2no3_540_fn, no2no3_900_fn = std_no2no3_900_fn)

    # create times series dataframe
    no2_conc_dic = {}
    no3_conc_dic = {}
    col_num = 0
    for no2_540_fn, no2_900_fn, no2no3_540_fn, no2no3_900_fn in zip(no2_540_fns, no2_900_fns, no2no3_540_fns, no2no3_900_fns):
        df = gr.get_concentration(no2_blank=no2_blank, no2no3_blank=no2no3_blank, g_fit=g_fit, v_fit=no3_fit,
                            meta_fn=meta_fn, no2_fn=None, no2_540_fn=no2_540_fn, no2_900_fn=no2_900_fn,
                            no2no3_fn=None, no2no3_540_fn=no2no3_540_fn, no2no3_900_fn=no2no3_900_fn)
        no2_conc_dic[col_num] = df['NO2_mM'].copy()
        no3_conc_dic[col_num] = df['NO3_mM'].copy()
        col_num += 1
    
    no2_conc = pd.DataFrame(no2_conc_dic)
    no3_conc = pd.DataFrame(no3_conc_dic)

    return no2_conc, no3_conc

def nh4_abs_to_conc(date, chl):
    std_am_meta_fn = f"data/{date}_standards_metadata.csv"
    std_am_absorb_fn = glob.glob(f"data/{date}_NH4_STD_{chl}_650.CSV")[0]
    std_am_900_fn = glob.glob(f"data/{date}_NH4_STD_{chl}_900.CSV")[0]

    meta_fn = f"data/{date}_samples_metadata.csv"
    nh4_650_fns = sorted(glob.glob(f"data/{date}_NH4_{chl}*650*"))
    nh4_900_fns = sorted(glob.glob(f"data/{date}_NH4_{chl}*900*"))

    fit_list = am.fit_ammonia(meta_fn = std_am_meta_fn, wavelength="650", data_absorb_fn = std_am_absorb_fn, data_900_fn = std_am_900_fn)
    [[ammonia_blank], fit, b_fit] = fit_list 

    # create times series dataframe
    nh4_dic = {}
    col_num = 0
    for nh4_650_fn, nh4_900_fn in zip(nh4_650_fns, nh4_900_fns):
        df = am.get_concentration(ammonia_blank = ammonia_blank, fit = b_fit, meta_fn = meta_fn, wavelength = "650", data_absorb_fn=nh4_650_fn, data_900_fn=nh4_900_fn)
        nh4_dic[col_num] = df['Ammonia_mM'].copy()
        col_num += 1

    nh4_conc = pd.DataFrame(nh4_dic)

    return nh4_conc

def create_figure(times, no2_data, no3_data, rows_chunk, filename, figure_index=None, num_of_replicates=3, nh4_data=None):
    # Determine the global y-axis limits
    all_values = pd.concat([no2_data, no3_data])
    if nh4_data is not None:
        all_values = pd.concat([all_values, nh4_data])
    y_min = all_values.min().min()
    y_max = all_values.max().max()

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

    for i in range(0, len(rows_chunk), num_of_replicates):
        if i // num_of_replicates < len(axes):  # Ensure we don't exceed the number of subplots
            ax = axes[i // num_of_replicates]
            for j in range(num_of_replicates):
                if i + j < len(rows_chunk):
                    row = rows_chunk[i + j]
                    ax.plot(times, no2_data.loc[row], 'r.-', label='NO2')
                    ax.plot(times, no3_data.loc[row], 'b.-', label='NO3')
                    if nh4_data is not None:
                        ax.plot(times, nh4_data.loc[row], 'g.-', label='NH4')
            ax.set_title(f'{rows_chunk[i:i+num_of_replicates]}')
            ax.set_ylim(y_min, y_max)
            ax.legend().set_visible(False)  # Hide individual legends

    # Add a single legend for the entire figure
    handles = [plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label='NO2'),
               plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label='NO3')]
    if nh4_data is not None:
        handles.append(plt.Line2D([0], [0], color='g', marker='.', linestyle='-', label='NH4'))
    fig.legend(handles=handles, loc='upper right', fontsize=20)

    # Add a single set of x and y labels for the entire figure
    fig.text(0.5, 0.04, 'Time (hours)', ha='center', fontsize=30)
    fig.text(0.04, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize=30)

    # Adjust layout to prevent overlap and set custom spacing
    plt.subplots_adjust(hspace=0.4, bottom=0.1, left=0.1)

    # Save the figure
    if figure_index is None:
        plt.savefig(f'{filename}.png')
        print(f'Saved figure: {filename}.png')
    else:
        plt.savefig(f'{filename}_{figure_index}.png')
        print(f'Saved figure: {filename}_{figure_index}.png')
    plt.close()
        

# Convert absorbances to concentration and save as dataframes
date = "20251224"
key = "batch4"
exp_num = "4.2"
[no2_conc1, no3_conc1] = no2_no3_abs_to_conc(date, key)
no2_conc = pd.concat([no2_conc1], axis=1)
no3_conc = pd.concat([no3_conc1], axis=1)

# # Multiply column 4 (index 4) by 24/15 for rows E01-E12
# no2_conc.loc['E01':'E12', 4] *= 24/15
# no3_conc.loc['E01':'E12', 4] *= 24/15

# Time arrays
datetime_array = [
    datetime(2025, 1, 1, 8, 17),       # T0
    datetime(2025, 1, 1, 12, 32),       # T1
    datetime(2025, 1, 1, 15, 59),       # T2
    datetime(2025, 1, 1, 20, 0),       # T3
    datetime(2025, 1, 1, 23, 29),       # T4
    datetime(2025, 1, 2, 6, 31),       # T5
    datetime(2025, 1, 2, 15, 12),       # T6
    datetime(2025, 1, 2, 23, 0),       # T7
    datetime(2025, 1, 3, 7, 58),       # T8
    datetime(2025, 1, 3, 21, 13),        # T9
    datetime(2025, 1, 4, 13, 1)]       # T10

# Concentration
times = [0]
for i in range(1, len(datetime_array)):
    time_diff = datetime_array[i] - datetime_array[0]
    times.append(time_diff.total_seconds() / 3600)

no2_conc.columns = times
no3_conc.columns = times
no3_cons = no3_conc.iloc[:, 0].values.reshape(-1, 1) - no3_conc
no2_cons = no2_conc.iloc[:, 0].values.reshape(-1, 1) - no2_conc + no3_cons

# no2_conc.to_csv(f"concentrations/{date}_{key}_no2_conc.csv")
# no3_conc.to_csv(f"concentrations/{date}_{key}_no3_conc.csv")
# no2_cons.to_csv(f"concentrations/{date}_{key}_no2_cons.csv")
# no3_cons.to_csv(f"concentrations/{date}_{key}_no3_cons.csv")

# Evaporation Correction
evap_rows = ['A01', 'A02', 'A03']
norm_no2 = no2_conc.loc[evap_rows].mean(axis=0) / no2_conc.loc[evap_rows].mean(axis=0).iloc[0]
norm_no3 = no3_conc.loc[evap_rows].mean(axis=0) / no3_conc.loc[evap_rows].mean(axis=0).iloc[0]
# norm_nh4_chl0 = (norm_no2_chl0 + norm_no3_chl0) / 2
# norm_nh4_chl1 = (norm_no2_chl1 + norm_no3_chl1) / 2

no2_conc_evap = no2_conc.div(norm_no2.values, axis=1)
no3_conc_evap = no3_conc.div(norm_no3.values, axis=1)
no3_cons_evap = no3_conc_evap.iloc[:, 0].values.reshape(-1, 1) - no3_conc_evap
no2_cons_evap = no2_conc_evap.iloc[:, 0].values.reshape(-1, 1) - no2_conc_evap + no3_cons_evap

no2_conc_evap.to_csv(f"concentrations/{exp_num}.{key}_no2_conc.csv")
no3_conc_evap.to_csv(f"concentrations/{exp_num}.{key}_no3_conc.csv")
no2_cons_evap.to_csv(f"concentrations/{exp_num}.{key}_no2_cons.csv")
no3_cons_evap.to_csv(f"concentrations/{exp_num}.{key}_no3_cons.csv")

# Create figures
# List of row names in the order you want to display them
rows_to_plot = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
                'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12',
                'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12',
                'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12',
                'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12',
                'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12',
                'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12',
                'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12']

# Number of rows and columns for the subplots grid
num_rpl = 3
ncols = 4
nrows = int(np.ceil(len(rows_to_plot) / ncols / num_rpl))
plots_per_figure = nrows * ncols

for [times, no2_data, no3_data, filename] in zip([times, times], [no2_conc, no2_cons], [no3_conc, no3_cons], [f'plots/{exp_num}.{key}_no3_no2_conc_all', f'plots/{exp_num}.{key}_no3_no2_cons_all']):
    for i in range(0, len(rows_to_plot), plots_per_figure * num_rpl):
        rows_chunk = rows_to_plot[i:i + plots_per_figure * num_rpl]
        create_figure(times=times, no2_data=no2_data, no3_data=no3_data, rows_chunk=rows_chunk, filename=filename, num_of_replicates=num_rpl) #figure_index=i // (plots_per_figure * num_rpl)
