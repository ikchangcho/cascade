import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_csv(id, meta_col_num=4):
    no3_conc_df = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0).iloc[:, :-meta_col_num]
    no2_conc_df = pd.read_csv(f'concentrations/{id}_no2_conc.csv', index_col=0).iloc[:, :-meta_col_num]
    no3_cons_df = pd.read_csv(f'concentrations/{id}_no3_cons.csv', index_col=0).iloc[:, :-meta_col_num]
    no2_cons_df = pd.read_csv(f'concentrations/{id}_no2_cons.csv', index_col=0).iloc[:, :-meta_col_num]
    meta_df = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0).iloc[:, -meta_col_num:]
    return no3_conc_df, no2_conc_df, no3_cons_df, no2_cons_df, meta_df

def plot_sample_type(x, no3_df, no2_df, meta_df, sample_type, title, y_label, output_fn, fontsize=18):
    sub_meta = meta_df[meta_df['Sample_type'] == sample_type]
    chl_levels = sorted(sub_meta['Chloramphenicol'].unique(), reverse=True)

    condition_order = [(1.25, 1.25), (0.0, 1.25), (1.25, 0.0), (0.0, 0.0)]
    present = sub_meta[['Nitrite_input', 'Nitrate_input']].drop_duplicates().apply(tuple, axis=1).tolist()
    conditions = [c for c in condition_order if c in present]

    all_values = pd.concat([no3_df.loc[sub_meta.index], no2_df.loc[sub_meta.index]])
    y_min, y_max = all_values.min().min(), all_values.max().max()

    n_rows, n_cols = len(chl_levels), len(conditions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(4.5 * n_cols, 6), max(4 * n_rows, 4.5)), squeeze=False)

    markers = ['o', 's', '^']
    for r, chl in enumerate(chl_levels):
        for c, (nitrite, nitrate) in enumerate(conditions):
            ax = axes[r][c]
            rows = sub_meta[(sub_meta['Chloramphenicol'] == chl) &
                             (sub_meta['Nitrite_input'] == nitrite) &
                             (sub_meta['Nitrate_input'] == nitrate)].index.tolist()
            for i, row in enumerate(rows):
                marker = markers[i % len(markers)]
                ax.plot(x, no3_df.loc[row], marker=marker, color='b', linestyle='-', markersize=4)
                ax.plot(x, no2_df.loc[row], marker=marker, color='r', linestyle='-', markersize=4)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(labelsize=fontsize - 6)
            if r == 0:
                ax.set_title(f'NO2_in={nitrite}, NO3_in={nitrate}', fontsize=fontsize - 4)
            if c == 0:
                ax.set_ylabel(f'CHL={chl}', fontsize=fontsize - 4)

    fig.text(0.5, 0.02, 'Time (hours)', ha='center', fontsize=fontsize)
    fig.text(0.02, 0.5, y_label, va='center', rotation='vertical', fontsize=fontsize)
    handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label='$NO_3$'),
               plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label='$NO_2$')]
    axes[0][-1].legend(handles=handles, loc='best', fontsize=fontsize - 6)
    fig.suptitle(title, fontsize=fontsize + 4, fontweight='bold')
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.93])
    plt.savefig(f'plots/{output_fn}', dpi=300, bbox_inches='tight')
    print(f'Saved plots/{output_fn}')
    plt.close(fig)

id = '4.3.isolate'
no3_conc_df, no2_conc_df, no3_cons_df, no2_cons_df, meta_df = load_csv(id)
x = no3_conc_df.columns.astype(float).tolist()

for sample_type in ['Blank', 'Strain1', 'Strain2', 'Strain3']:
    if sample_type not in meta_df['Sample_type'].values:
        continue
    plot_sample_type(x, no3_conc_df, no2_conc_df, meta_df, sample_type,
                      title=f'{id} {sample_type} Concentration', y_label='Concentration (mM)',
                      output_fn=f'{id}.{sample_type}_no3_no2_conc.png')
    plot_sample_type(x, no3_cons_df, no2_cons_df, meta_df, sample_type,
                      title=f'{id} {sample_type} Consumption', y_label='Consumption (mM)',
                      output_fn=f'{id}.{sample_type}_no3_no2_cons.png')
