import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5', '4.2.batch6']
data = {}
for id in ids:
    data[id] = {}
    data[id]['no3'] = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0)
    data[id]['no2'] = pd.read_csv(f'concentrations/{id}_no2_conc.csv', index_col=0)

# Figure 1
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(r'Raw $NO_3^-$ concentration $A(t)$ - drug condition' + '\n' + r'one curve per replicate | color = batch')
subplot_titles = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
for i in range(6):
    ax = axes[i // 3, i % 3]
    ax.set_title(subplot_titles[i])
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel(r'$NO_3^-$ (mM)')
    df = data[ids[i]]['no3']        # no3 data
    df = df.iloc[:, :-4][(df['Chloramphenicol'] == 1) & (df['Sample_type'] != 'Blank')]        # drug data
    x = df.columns.astype(float)        # time points
    for well in df.index:
        y = df.loc[well].values.astype(float)        # concentrations
        ax.plot(x, y, color=colors[i], linestyle='-', marker='.', alpha=0.5)
plt.tight_layout()
plt.show()
    
    

