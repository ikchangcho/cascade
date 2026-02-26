import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_no3_no2(x, no3_df, no2_df, row_labels, title, y_label, num_rpl=3, num_col=4, fontsize=25, output_fn='', show_plot=False):
    all_values = pd.concat([no3_df.loc[row_labels], no2_df.loc[row_labels]])
    y_min = all_values.min().min()
    y_max = all_values.max().max()

    num_row = int(np.ceil(len(row_labels) / num_col / num_rpl))
    fig, axes = plt.subplots(num_row, num_col, figsize=(5*num_row, 4*num_col))
    axes = axes.flatten()
    markers = ['o', 's', '^']
    for i, row in enumerate(row_labels):
        marker = markers[i % num_rpl]
        ax = axes[i // num_rpl + 1]
        ax.plot(x, no3_df.loc[row], marker, color='b', linestyle='-')
        ax.plot(x, no2_df.loc[row], marker, color='r', linestyle='-')
        ax.tick_params(axis='x', labelsize=fontsize - 5)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='y', labelsize=fontsize - 5)
    fig.text(0.55, 0.05, 'Time (hours)', ha='center', fontsize=fontsize)
    fig.text(0.145, 0.9, f'A_add = 2.0 mM', fontsize=fontsize)
    fig.text(0.35, 0.9, f'A_add = 1.4 mM', fontsize=fontsize)
    fig.text(0.55, 0.9, f'A_add = 0.7 mM', fontsize=fontsize)
    fig.text(0.75, 0.9, f'A_add = 0.0 mM', fontsize=fontsize)
    fig.text(0.08, 0.5, y_label, va='center', rotation='vertical', fontsize=fontsize)
    fig.text(0.91, 0.77, f'I_add =\n2.0 mM', fontsize=fontsize)
    fig.text(0.91, 0.575, f'I_add =\n1.4 mM', fontsize=fontsize)
    fig.text(0.91, 0.37, f'I_add =\n0.7 mM', fontsize=fontsize)
    fig.text(0.91, 0.165, f'I_add =\n0.0 mM', fontsize=fontsize)
    handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'$NO_3$ (A)'),
                plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'$NO_2$ (I)')]
    fig.legend(handles=handles, loc='upper right', fontsize=fontsize)
    fig.suptitle(title, fontsize=fontsize+5, fontweight='bold')
    if output_fn != '':
        plt.savefig(f'plots/{output_fn}', dpi=300, bbox_inches='tight')
        print(f'Saved plots/{output_fn}')
    if show_plot:
        plt.show()

def load_csv(id, meta_col_num=4):
    no3_conc_df = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0).iloc[:, :-meta_col_num]
    no2_conc_df = pd.read_csv(f'concentrations/{id}_no2_conc.csv', index_col=0).iloc[:, :-meta_col_num]
    no3_cons_df = pd.read_csv(f'concentrations/{id}_no3_cons.csv', index_col=0).iloc[:, :-meta_col_num]
    no2_cons_df = pd.read_csv(f'concentrations/{id}_no2_cons.csv', index_col=0).iloc[:, :-meta_col_num]
    meta_df = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0).iloc[:, -meta_col_num:]
    return no3_conc_df, no2_conc_df, no3_cons_df, no2_cons_df, meta_df

ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5']
for id in ids[0:]:
    # no3_conc_df, no2_conc_df, no3_cons_df, no2_cons_df, meta_df = load_csv(id)
    # x = no3_conc_df.columns.astype(float).tolist()
    # row_labels_chl1 = meta_df[(meta_df['Chloramphenicol'] == 1) & (meta_df['Sample_type'] != 'Blank')].index.tolist()
    # row_labels_chl0 = meta_df[(meta_df['Chloramphenicol'] == 0) & (meta_df['Sample_type'] != 'Blank')].index.tolist()
    
    # for no3_df, no2_df, row_labels, title, y_label, output_fn in [
    #     (no3_conc_df, no2_conc_df, row_labels_chl1, f'{id} CHL+ Concentration', 'Concentration (mM)', f'{id}.chl1_no3_no2_conc.png'),
    #     (no3_cons_df, no2_cons_df, row_labels_chl1, f'{id} CHL+ Consumption', 'Consumption (mM)', f'{id}.chl1_no3_no2_cons.png'),
    #     (no3_conc_df, no2_conc_df, row_labels_chl0, f'{id} CHL- Concentration', 'Concentration (mM)', f'{id}.chl0_no3_no2_conc.png'),
    #     (no3_cons_df, no2_cons_df, row_labels_chl0, f'{id} CHL- Consumption', 'Consumption (mM)', f'{id}.chl0_no3_no2_cons.png')
    # ]:
    #     plot_fifteen_conditions(x, no3_df, no2_df, row_labels, title, y_label, 
    #                             output_fn=output_fn, show_plot=False)

    no3_conc_dfs, no2_conc_dfs, no3_cons_dfs, no2_cons_dfs = [], [], [], []
    for id in ids:
        no3_conc_df, no2_conc_df, no3_cons_df, no2_cons_df, meta_df = load_csv(id)
        no3_conc_dfs.append(no3_conc_df)
        no2_conc_dfs.append(no2_conc_df)
        no3_cons_dfs.append(no3_cons_df)
        no2_cons_dfs.append(no2_cons_df)
    
row_labels = meta_df[(meta_df['Chloramphenicol'] == 1) & (meta_df['Sample_type'] != 'Blank')].index.tolist()
dfs_to_plot = no3_conc_dfs

fig, axes = plt.subplots(4, 5, figsize=(20, 15))
axes = axes.flatten()
markers = ['o', 's', '^']
num_rpl, num_col = 3, 4
for i, row in enumerate(row_labels):
    marker = markers[i % num_rpl]
    q, r = divmod(i // 3 + 1, num_col)
    axes[5*q + r + 2].plot(dfs_to_plot[0].columns.astype(float).tolist(), dfs_to_plot[0].loc[row], marker, color='b', linestyle='-')
plt.show()




