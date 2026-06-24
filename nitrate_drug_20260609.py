from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import fsolve

ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5', '4.2.batch6']
batch_colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
batch_labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
add_conc = [0.0, 0.7, 1.4, 2.0]
markers = ['o', 's', '^', 'D']
colors = ['blue', 'green', 'red', 'cyan']

data_dict = {}
for id in ids:
    data_dict[id] = {}
    data_dict[id]['no3_conc'] = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0).iloc[:, :-4]
    data_dict[id]['no2_conc'] = pd.read_csv(f'concentrations/{id}_no2_conc.csv', index_col=0).iloc[:, :-4]
    data_dict[id]['no3_cons'] = pd.read_csv(f'concentrations/{id}_no3_cons.csv', index_col=0).iloc[:, :-4]
    data_dict[id]['no2_cons'] = pd.read_csv(f'concentrations/{id}_no2_cons.csv', index_col=0).iloc[:, :-4]
    data_dict[id]['metadata'] = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0).iloc[:, -4:]

mask_chl_blank_dict = {}
for id in ids:
    mask_chl_blank_dict[id] = (data_dict[id]['metadata']['Chloramphenicol'] == 1.0) & (data_dict[id]['metadata']['Sample_type'] != 'Blank')


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i in range(6):
    id = ids[i]
    ax = axes[i // 3, i % 3]
    no3_conc_df = data_dict[id]['no3_conc'].loc[mask_chl_blank_dict[id]]
    meta_df = data_dict[id]['metadata'].loc[mask_chl_blank_dict[id]]
    
    no3_conc_norm_df = no3_conc_df
    for no3_add in add_conc:
        mask = (meta_df['Nitrate_input'] == no3_add)
        mean_no3_value = no3_conc_df.loc[mask].iloc[:, 0].mean()
        no3_conc_norm_df.loc[mask] = no3_conc_df.loc[mask] - mean_no3_value
    
    times = no3_conc_norm_df.columns.values
    for index in no3_conc_norm_df.index:
        no3_conc = no3_conc_norm_df.loc[index].values
        no3_add = meta_df.loc[index, 'Nitrate_input']
        ax.plot(times, no3_conc, linestyle='-', color=colors[add_conc.index(no3_add)], alpha = 0.3)

plt.show()