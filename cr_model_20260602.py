import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5', '4.2.batch6']
data = {}
for id in ids:
    data[id] = {}
    data[id]['no3'] = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0)
    data[id]['no2'] = pd.read_csv(f'concentrations/{id}_no2_conc.csv', index_col=0)

def load_data_to_plot(ids, data, no3_or_no2, chl):
    data_to_plot = {}
    for id in ids:
        df = data[id][no3_or_no2]
        df = df[(df['Chloramphenicol'] == chl) & (df['Sample_type'] != 'Blank')]
        data_to_plot[id] = df
    return data_to_plot

# # Figure 1
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Raw $NO_3^-$ concentration $A(t)$ - drug condition' + '\n' + r'one curve per replicate | colour = batch')
# subplot_titles = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
# colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
# for i in range(6):
#     ax = axes[i // 3, i % 3]
#     ax.set_title(subplot_titles[i])
#     ax.set_xlabel('Time (hr)')
#     ax.set_ylabel(r'$NO_3^-$ (mM)')
#     df = data[ids[i]]['no3']        # no3 data
#     df = df.iloc[:, :-4][(df['Chloramphenicol'] == 1) & (df['Sample_type'] != 'Blank')]        # drug data
#     x = df.columns.astype(float)        # time points
#     for well in df.index:
#         y = df.loc[well].values.astype(float)        # concentrations
#         ax.plot(x, y, color=colors[i], linestyle='-', marker='.', alpha=0.5)
# plt.close()


# # Figure 2
# no3_or_no2 = 'no3'
# chl = 1
# data_to_plot = {}
# for id in ids:
#     df = data[id][no3_or_no2]
#     df = df[(df['Chloramphenicol'] == chl) & (df['Sample_type'] != 'Blank')]
#     data_to_plot[id] = df
# y_min = min(df.iloc[:, :-4].min().min() for df in data_to_plot.values())
# y_max = max(df.iloc[:, :-4].max().max() for df in data_to_plot.values())

# fig, axes = plt.subplots(4, 4, figsize=(20, 20))
# fig.suptitle(r'Raw $NO_3^-$ concentration by condition ($A_{add}$, $I_{add}$) - colour = batch'
#                 + '\n' + r'rows = nominal $I_{add}$ | columns = nominal $A_{add}$'
#                 + '\n' + r'shared y-axis | thick = condition mean | thin = replicates')
# colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
# legends = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
# no3_add_values = [0.0, 0.7, 1.4, 2.0]
# no2_add_values = [0.0, 0.7, 1.4, 2.0]

# for i in range(4):
#     for j in range(4):
#         ax = axes[i, j]
#         ax.set_xlabel('Time (hr)') if i == 3 else None
#         ax.set_ylabel(r'$NO_3^-$ (mM)') if j == 0 else None
#         ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
#         no3_add = no3_add_values[j]
#         no2_add = no2_add_values[i]
#         if i == 0:
#             ax.set_title(f'$A_{{add}}$ = {no3_add}', fontsize=10)
#         if j == 0:
#             ax.text(-0.35, 0.5, f'$I_{{add}}$ = {no2_add}', 
#                    transform=ax.transAxes, fontsize=10, 
#                    rotation=90, va='center', ha='center')

#         for id, color, legend in zip(ids, colors, legends):
#             df = data_to_plot[id]
#             df = df.iloc[:, :-4][(df['Sample_type'] != 'Blank') & (df['Nitrate_input'] == no3_add) & (df['Nitrite_input'] == no2_add)]
#             x = df.columns.astype(float)
#             y_mean = df.mean(axis=0).values.astype(float)
#             ax.plot(x, y_mean, color=color, linestyle='-', linewidth=1.5, alpha=1, label=legend)
#             for well in df.index:
#                 y = df.loc[well].values.astype(float)
#                 ax.plot(x, y, color=color, linestyle='-', linewidth=0.5, alpha=0.5)
# handles, labels = axes[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10)
# plt.show()


# # Figure 3
# no3_or_no2 = 'no3'
# chl = 1
# data_to_plot = {}
# for id in ids:
#     df = data[id][no3_or_no2]
#     df = df[(df['Chloramphenicol'] == chl) & (df['Sample_type'] != 'Blank')]
#     data_to_plot[id] = df.iloc[:, :-4]

# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle(r'Normalized cumulative consumption $\tilde{C}(t)$ - individual replicates'
#                  + '\n' + r'$t_{last}$ = first zero time point (or last measured if never zero) | diagonal = linear model')
# batch_labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
# colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
# for i in range(6):
#     ax = axes[i // 3, i % 3]
#     df = data_to_plot[ids[i]]
#     time = df.columns.astype(float)
#     count = 0
#     for well in df.index:
#         y = df.loc[well].values.astype(float)
#         thrsh_init = 0.3
#         if y[0] < thrsh_init:
#             continue
        
#         thrsh_end = 0.05
#         zero_indices = np.where(y < thrsh_end)[0]
#         if len(zero_indices) > 0:
#             last_index = zero_indices[0]
#         else:
#             last_index = len(y) - 1

#         y = y[:last_index + 1]
#         y = (y[0] - y) / (y[0] - y[-1]) 
#         x = time[:last_index + 1] / time[last_index]
        
#         ax.plot(x, y, color=colors[i], linestyle='-', alpha=0.5)
#         count += 1
#     ax.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=1.0, label='Linear')
#     ax.set_title(batch_labels[i] + f' (n={count})', color=colors[i])
#     if i == 0:
#         ax.legend(loc='upper left')
#     if i  % 3 == 0:
#         ax.set_ylabel(r'$\tilde{C}(t) = \frac{A(0) - A(t)}{A(0) - A(t_{last})}$')
#     if i // 3 == 1:
#         ax.set_xlabel(r'$\frac{t}{t_{last}}$')
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
# plt.show()


# Figure 4
data_to_plot = load_data_to_plot(ids, data, 'no3', chl=1)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(r'$I_0$ independance: all $I_0$ pairs, all relicate combinations - one panel per batch' 
                 + '\n' + r'colour = nominal $A_{add}$ | diagonal = perfect agreement | all 3 x3 replicate pairs per ($A_{add}, I_{add}$ pair, batch)')
colors = ['black', 'blue', 'orange', 'red']
batch_labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
for i in range(6):
    ax = axes[i // 3, i % 3]
    df = data_to_plot[ids[i]]
    N = 0
    se = 0
    for A_add, color in zip([0.0, 0.7, 1.4, 2.0], colors):
        df1 = df[df['Nitrate_input'] == A_add]
        if A_add == 2.0:
            I_add_pairs = [(0.0, 0.7), (0.0, 1.4), (0.7, 1.4)]
        else:
            I_add_pairs = [(0.0, 0.7), (0.0, 1.4), (0.0, 2.0), (0.7, 1.4), (0.7, 2.0), (1.4, 2.0)]
        for I_add_pair in I_add_pairs:
            df_x = df1[df1['Nitrite_input'] == I_add_pair[0]].iloc[:, :-4]
            df_y = df1[df1['Nitrite_input'] == I_add_pair[1]].iloc[:, :-4]
            if len(df_x.index) != 3 or len(df_y.index) != 3:
                print(f"Warning: Batch {ids[i]} - A_add={A_add} - I_add_pair={I_add_pair} have {len(df_x.index)} and {len(df_y.index)} replicates respectively, expected 3 each.")
            for idx_x in df_x.index:
                for idx_y in df_y.index:
                    x = df_x.loc[idx_x]
                    y = df_y.loc[idx_y]
                    ax.scatter(x, y, color=color, s=0.5, marker='o', alpha=0.3, label=f'$A_{{add}}$={A_add}' if (I_add_pair == (0.0, 0.7)) & (idx_x == df_x.index[0]) & (idx_y == df_y.index[0]) else None)
                    se += np.sum((x - y) ** 2 / 2)
                    N += len(x)
    ax.legend(loc='upper left', ncols=2, fontsize=8) if i == 0 else None
    ax.set_xlabel(r'A(t), $I_{add}$=a (mM)') if i // 3 == 1 else None
    ax.set_ylabel(r'A(t), $I_{add}$=b (mM)') if i % 3 == 0 else None
    ax.set_title(batch_labels[i], fontsize=10)
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(0.95, 0.05, f'RMSE = {np.sqrt(se/N):.3f} mM', 
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.show()
                
                    
                    
        





    

