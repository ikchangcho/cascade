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

filepath = '20250114'

# Create a new folder to save the PNG files
output_folder = f'{filepath}/plots'
os.makedirs(output_folder, exist_ok=True)

# Set figure size and font size for standard curves
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (20, 12),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

# Load file names
std_meta_fn = f'{filepath}/standard_metadata.csv'
std_no2_540_fn = glob.glob(f"{filepath}/*_Ik_STD_NO2_540*")[0]
std_no2_900_fn = glob.glob(f"{filepath}/*_Ik_STD_NO2_900*")[0]
std_no2no3_540_fn = glob.glob(f"{filepath}/*_Ik_STD_NO2NO3_540*")[0]
std_no2no3_900_fn = glob.glob(f"{filepath}/*_Ik_STD_NO2NO3_900*")[0]

# Plot standard curves and save .png files
gr.plot_griess_fit(meta_fn = std_meta_fn, no2_540_fn=std_no2_540_fn, no2_900_fn=std_no2_900_fn, no2no3_540_fn = std_no2no3_540_fn, no2no3_900_fn = std_no2no3_900_fn)
plt.savefig(f'{filepath}/plots/standard_no2.png')
plt.cla()
gr.plot_no3_fit(meta_fn = std_meta_fn, no2_540_fn=std_no2_540_fn, no2_900_fn=std_no2_900_fn, no2no3_540_fn = std_no2no3_540_fn, no2no3_900_fn = std_no2no3_900_fn)
plt.savefig(f'{filepath}/plots/standard_no3.png')