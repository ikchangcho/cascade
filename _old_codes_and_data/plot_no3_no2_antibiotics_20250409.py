import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

no2_data = pd.read_csv('concentrations/no2_antibiotics.csv', index_col=0)
no3_data = pd.read_csv('concentrations/no3_antibiotics.csv', index_col=0)
all_values = pd.concat([no2_data, no3_data])
y_min = all_values.min().min()
y_max = all_values.max().max()

meta_data = pd.read_csv('concentrations/samples_metadata_antibiotics.csv', index_col=0).dropna(how='all')

times = no2_data.columns.astype(float).tolist()
initial_chl = meta_data.loc[:, 'Chloramphenicol']
initial_tetra = meta_data.loc[:, 'Tetracycline']
initial_no3 = meta_data.loc[:, 'Nitrate_input']
initial_no2 = meta_data.loc[:, 'Nitrite_input']

num_col = 4
num_row = 4
num_rpl = 2

fig, axes = plt.subplots(num_row, num_col, figsize=(4*num_row, 3*num_col))
axes = axes.flatten()

for i, row in enumerate(['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08']):
    ax = axes[i//num_rpl]
    #ax.plot(times, no2_data.loc[row], 'r.-')
    ax.plot(times, no3_data.loc[row], 'b.-')
    ax.set_xticks([0, 20, 40, 60])
    ax.tick_params(axis='x', labelsize=12)
    #ax.set_ylim(y_min, y_max)
    #ax.set_yticks([0, 1, 2, 3])
    ax.tick_params(axis='y', labelsize=12)

filename = 'antibiotics_no3_closeup.png'    
title = f'$NO_3$ Dynamics in the Presence of Antibiotics'
chl1 = initial_chl['A01']
chl2 = initial_chl['A03']
chl3 = initial_chl['A05']
chl4 = initial_chl['A07']
chl5 = initial_chl['A09']
chl6 = initial_chl['A11']
tetra1 = initial_tetra['A01']
tetra2 = initial_tetra['A03']
tetra3 = initial_tetra['A05']
tetra4 = initial_tetra['A07']
tetra5 = initial_tetra['A09']
tetra6 = initial_tetra['A11']
fig.text(0.48, 0.03, 'Time (hours)', fontsize=15)
fig.text(0.17, 0.9, f'CHL {chl1} mM \nTetra {tetra1} mM', fontsize=15)
fig.text(0.37, 0.9, f'CHL {chl2} mM \nTetra {tetra2} mM', fontsize=15)
fig.text(0.58, 0.9, f'CHL {chl3} mM \nTetra {tetra3} mM', fontsize=15)
fig.text(0.78, 0.9, f'CHL {chl4} mM \nTetra {tetra4} mM', fontsize=15)
#fig.text(0.67, 0.9, f'CHL {chl5} mM \nTetra {tetra5} mM', fontsize=15)
#fig.text(0.80, 0.9, f'CHL {chl6} mM \nTetra {tetra6} mM', fontsize=15)
fig.text(0.07, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize=15)
fig.text(0.91, 0.78, f'$A_0$ 2.0 mM \n$I_0$ 2.0 mM', fontsize=15)
fig.text(0.91, 0.58, f'$A_0$ 2.0 mM \n$I_0$ 0.0 mM', fontsize=15)
fig.text(0.91, 0.38, f'$A_0$ 1.0 mM \n$I_0$ 1.0 mM', fontsize=15)
fig.text(0.91, 0.17, f'$A_0$ 1.0 mM \n$I_0$ 0.0 mM', fontsize=15)
fig.suptitle(title, fontsize=20, y=1)
handles = [plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'NO_2'),
            plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'NO_3')]
#fig.legend(handles=handles, bbox_to_anchor=(1, 0.96), fontsize=12)
plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
print(f'Figure saved as plots/{filename}')
#plt.show()
