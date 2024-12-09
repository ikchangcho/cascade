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

# Create a new folder to save the PNG files
output_folder = f'{filepath}/plots'
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(10, 7))
plt.plot(times, no2_time_series.iloc[12], 'ro', label='I')
plt.plot(times, no3_time_series.iloc[12], 'bo', label='A')

plt.xlabel('Time (hours)', fontsize=25)
plt.ylabel('Concentration (mM)', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)

#plt.title('Time Series of NO2 and NO3 Concentrations', fontsize=15)
#plt.grid(True)
plt.savefig(f'{output_folder}/B01.png')
plt.show()