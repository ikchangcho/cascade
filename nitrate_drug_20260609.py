from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import fsolve

ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5', '4.2.batch6']
batch_colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
batch_labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
A_add_list = [0.0, 0.7, 1.4, 2.0]
A_add_markers = ['o', 's', '^', 'D']
A_add_colors = ['blue', 'green', 'red', 'cyan']

data = {}
for id in ids:
    data[id] = {}
    data[id]['no3_conc'] = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0).iloc[:, :-4]
    data[id]['no2_conc'] = pd.read_csv(f'concentrations/{id}_no2_conc.csv', index_col=0).iloc[:, :-4]
    data[id]['no3_cons'] = pd.read_csv(f'concentrations/{id}_no3_cons.csv', index_col=0).iloc[:, :-4]
    data[id]['no2_cons'] = pd.read_csv(f'concentrations/{id}_no2_cons.csv', index_col=0).iloc[:, :-4]
    data[id]['metadata'] = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0).iloc[:, -4:]

# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Raw $NO_3^-$ consumption $\Delta A(t)$ - drug condition' 
#                 + '\n' + r'$A_{add}$ > 1 mM only | one curve per replicate | colour = batch')
# subplot_titles = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
# I_add_colors = ['blue', 'green', 'red', 'cyan']
# I_add_list = [0.0, 0.7, 1.4, 2.0]
# for i in range(6):
#     ax = axes[i // 3, i % 3]
#     ax.set_title(subplot_titles[i])
#     if i // 3 == 1:
#         ax.set_xlabel('Time (hr)')
#     if i % 3 == 0:
#         ax.set_ylabel(r'$\Delta NO_3^-$ (mM)')

#     df = data_to_plot[ids[i]]
#     df = df[df['Nitrate_input'] > 1.0]

#     for I_add, color in zip(I_add_list, I_add_colors):
#         df_I = df[df['Nitrite_input'] == I_add].iloc[:, :-4]
#         for well in df_I.index:
#             time = df_I.columns.astype(float)
#             y = df_I.loc[well].values.astype(float)
#             ax.plot(time, y, color=color, linestyle='-', marker='.', alpha=0.2, label=f'$I_{{add}}$={I_add}' if well == df_I.index[0] else None)
#     if i == 0:
#         ax.legend(loc='upper right')
# plt.savefig(f'plots/no3_cons_A_add_bigger_than_1mM_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# plt.show()

key = 'no3_cons'
chl = 1.0
data_to_plot = []
meta_to_plot = []
for id in ids:
    meta = data[id]['metadata']
    wells_to_plot = meta[(meta['Chloramphenicol'] == chl) & (meta['Nitrite_input'].isin(A_add_list))].index
    data_to_plot.append(data[id][key].loc[wells_to_plot])
    meta_to_plot.append(meta.loc[wells_to_plot])
        
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(r'Median s values for the first N time points - drug condition'
                + '\n' + r'diamond = across all $A_{add}$ | shaded colour = $A_{add}$ ')
for i in range(6):
    ax = axes[i // 3, i % 3]
    ax.set_title(batch_labels[i])
    if i // 3 == 1:
        ax.set_xlabel('Number of time points')
    if i % 3 == 0:
        ax.set_ylabel('s (mM/hr)')

    df = data_to_plot[i]
    time = df.columns.astype(float).values
    meta = meta_to_plot[i]
    slopes_df = pd.DataFrame(index=df.index, columns=range(2, len(time) + 1))
    for well in df.index:
        y = df.loc[well].values.astype(float)
        for n in range(2, len(time) + 1):
            slope, intercept = np.polyfit(time[:n], y[:n], 1)
            slopes_df.loc[well, n] = slope
    
    median_slopes = slopes_df.median()
    std_slopes = slopes_df.std()
    ax.errorbar(range(2, len(time) + 1), median_slopes, yerr=std_slopes, marker='D'
                , color='black', label='Median', alpha = 0.8)
    
    for A_add, marker, color in zip(A_add_list, A_add_markers, A_add_colors):
        wells_A = meta[meta['Nitrate_input'] == A_add].index
        mean_slopes = slopes_df.loc[wells_A].median()
        std_slopes = slopes_df.loc[wells_A].std()
        x = np.array(range(2, len(time) + 1)) + 0.1 * (A_add - 0.1)
        ax.errorbar(x, mean_slopes, yerr=std_slopes, marker=marker, alpha = 0.3, color=color
                    , label=f'$A_{{add}}$={A_add}')
    ax.legend(loc='upper right') if i == 0 else None
plt.savefig(f'plots/s_vs_N_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
plt.show()
            