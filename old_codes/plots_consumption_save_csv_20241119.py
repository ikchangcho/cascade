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
filepath = '20241103'

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

# Create the output folder if it doesn't exist
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

# Determine the global y-axis limits for consumption values
no3_consumption = no3_time_series.iloc[:, 0].values.reshape(-1, 1) - no3_time_series
no2_consumption = no2_time_series.iloc[:, 0].values.reshape(-1, 1) - no2_time_series + no3_consumption

# Save consumption DataFrames to CSV files
no3_consumption.to_csv(f'{filepath}/no3_consumption.csv')
no2_consumption.to_csv(f'{filepath}/no2_consumption.csv')
print('Consumption DataFrames saved as CSV files in the folder:', filepath)