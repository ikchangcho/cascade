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

# Determine the global y-axis limits
all_values = pd.concat([no2_data, no3_data])
y_min = all_values.min().min()
y_max = all_values.max().max()
times = no2_data.columns.astype(float).tolist()

# List of row names in the order you want to display them
num_of_replicates = 3
rows_to_plot = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03']
data = no3_data.loc[['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03']]
title = '2025-01-14 CHL- \n NO3 Concentration'
filename = 'test'
colors = ["#b30000", "#4421af", "#0d88e6", "#5ad45a", "#ebdc78"]
labels = ['I(0)=2.0', 'I(0)=1.5', 'I(0)=1.0', 'I(0)=0.5', 'I(0)=0.0']

fig, ax = plt.subplots(figsize=(10, 8))
for ii, row in enumerate(rows_to_plot):
    ax.plot(times, no3_data.loc[row], '.-', label=labels[ii//num_of_replicates], color=colors[ii//num_of_replicates])
handles = [plt.Line2D([0], [0], color=colors[0], marker='.', linestyle='-', label=labels[0]),
           plt.Line2D([0], [0], color=colors[1], marker='.', linestyle='-', label=labels[1]),
           plt.Line2D([0], [0], color=colors[2], marker='.', linestyle='-', label=labels[2]),
           plt.Line2D([0], [0], color=colors[3], marker='.', linestyle='-', label=labels[3]),
           plt.Line2D([0], [0], color=colors[4], marker='.', linestyle='-', label=labels[4])]
ax.legend(handles=handles)
ax.set_xlabel('Time (hours)', fontsize=15)
ax.set_ylabel('Concentration (mM)', fontsize=15)
ax.grid(True)
ax.set_title(title, fontsize=20)

plt.show()