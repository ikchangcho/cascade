import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5', '4.2.batch6']
data = {}
for id in ids:
    data[id] = {}
    data[id]['no3'] = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0)
    data[id]['no2'] = pd.read_csv(f'concentrations/{id}_no2_conc.csv', index_col=0)

no3_or_no2 = 'no3'
chl = 1
dfs_to_plot = [data[id][no3_or_no2] for id in ids]
dfs_to_plot = [df[(df['Chloramphenicol'] == chl) & (df['Sample_type'] != 'Blank')] for df in dfs_to_plot]
print(dfs_to_plot[0].head())
