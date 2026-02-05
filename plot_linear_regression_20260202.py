import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

exp_nums = ['4.2', '4.2', '4.2', '4.2', '4.2']
dates = ['20251129', '20251206', '20251213', '20251224', '20260124']
ids = ['batch1', 'batch2', 'batch3', 'batch4', 'batch5']
wcs = [98.9, 62.5, 34.36, 7.10, 5.16]

for i in [0, 1, 2, 3, 4]:       # batch number - 1
    exp_num = exp_nums[i]
    date = dates[i]
    id = ids[i]
    wc = wcs[i]

    meta = pd.read_csv(f'absorbances/{date}_samples_metadata.csv', index_col=0).dropna(how='all')
    no2_conc = pd.read_csv(f'concentrations/{exp_num}.{id}_no2_conc.csv', index_col=0)
    no3_conc = pd.read_csv(f'concentrations/{exp_num}.{id}_no3_conc.csv', index_col=0)
    linear_regression_results = pd.read_csv(f'fitting_results/{exp_num}.{id}_linear_regression_results.csv', index_col=0)

    times = no2_conc.columns.astype(float).tolist()

    # CHL+
    rows_chl1 = ['A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12']
    all_values = pd.concat([no3_conc.loc[rows_chl1], no2_conc.loc[rows_chl1]])
    y_min = all_values.min().min()
    y_max = all_values.max().max()

    num_rpl = 3
    num_col = 4
    num_row = int(np.ceil(len(rows_chl1) / num_col / num_rpl))
    fig, axes = plt.subplots(num_row, num_col, figsize=(5*num_row, 4*num_col))
    axes = axes.flatten()
    for i, row in enumerate(rows_chl1):
        ax = axes[i // num_rpl + 1]
        marker_styles = ['o', 's', '^']
        marker = marker_styles[i % num_rpl]
        ax.scatter(times, no2_conc.loc[row], color='r', marker=marker)
        ax.scatter(times, no3_conc.loc[row], color='b', marker=marker)
        no2_slope, no2_intercept, no3_slope, no3_intercept = linear_regression_results.loc[row]
        times_array = np.array(times)
        ax.plot(times_array, no2_slope * times_array + no2_intercept, 'r-')
        ax.plot(times_array, no3_slope * times_array + no3_intercept, 'b-')
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.tick_params(axis='x', labelsize=25)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([0, 1, 2, 3])
        ax.tick_params(axis='y', labelsize=25)
    fig.text(0.55, 0.05, 'Time (hours)', ha='center', fontsize=30)
    fig.text(0.145, 0.9, f'A(0) = 2.0 mM', fontsize=25)
    fig.text(0.35, 0.9, f'A(0) = 1.4 mM', fontsize=25)
    fig.text(0.55, 0.9, f'A(0) = 0.7 mM', fontsize=25)
    fig.text(0.75, 0.9, f'A(0) = 0.0 mM', fontsize=25)
    fig.text(0.08, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize=30)
    fig.text(0.91, 0.77, f'I(0) =\n2.0 mM', fontsize=25)
    fig.text(0.91, 0.575, f'I(0) =\n1.4 mM', fontsize=25)
    fig.text(0.91, 0.37, f'I(0) =\n0.7 mM', fontsize=25)
    fig.text(0.91, 0.165, f'I(0) =\n0.0 mM', fontsize=25)
    handles = [plt.Line2D([0], [0], color='b', marker='o', label=f'$NO_3$ (A)'),
                plt.Line2D([0], [0], color='r', marker='o', label=f'$NO_2$ (I)')]
    fig.legend(handles=handles, loc='upper right', fontsize=20)
    fig.suptitle(f'{id} ({wc} %whc), CHL+\nLinear Regression on Early Time Points', fontsize=30, fontweight='bold')
    filename = f'{exp_num}.{id}_linear_regression_no3_no2_conc_chl1.png'
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    print(f'Saved plots/{filename}')
    plt.close()

    # CHL-
    rows_chl0 = ['E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12']
    all_values = pd.concat([no3_conc.loc[rows_chl0], no2_conc.loc[rows_chl0]])
    y_min = all_values.min().min()
    y_max = all_values.max().max()
    num_rpl = 3
    num_col = 4
    num_row = int(np.ceil(len(rows_chl0) / num_col / num_rpl))
    fig, axes = plt.subplots(num_row, num_col, figsize=(5*num_row, 4*num_col))
    axes = axes.flatten()
    for i, row in enumerate(rows_chl0):
        ax = axes[i // num_rpl + 1]
        marker = marker_styles[i % num_rpl]
        ax.scatter(times, no2_conc.loc[row], color='r', marker=marker)
        ax.scatter(times, no3_conc.loc[row], color='b', marker=marker)
        no2_slope, no2_intercept, no3_slope, no3_intercept = linear_regression_results.loc[row]
        times_array = np.array(times)
        ax.plot(times_array, no2_slope * times_array + no2_intercept, 'r-')
        ax.plot(times_array, no3_slope * times_array + no3_intercept, 'b-')
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.tick_params(axis='x', labelsize=25)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([0, 1, 2, 3])
        ax.tick_params(axis='y', labelsize=25)
    fig.text(0.55, 0.05, 'Time (hours)', ha='center', fontsize=30)
    fig.text(0.145, 0.9, f'A(0) = 2.0 mM', fontsize=25)
    fig.text(0.35, 0.9, f'A(0) = 1.4 mM', fontsize=25)
    fig.text(0.55, 0.9, f'A(0) = 0.7 mM', fontsize=25)
    fig.text(0.75, 0.9, f'A(0) = 0.0 mM', fontsize=25)
    fig.text(0.08, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize=30)
    fig.text(0.91, 0.77, f'I(0) =\n2.0 mM', fontsize=25)
    fig.text(0.91, 0.575, f'I(0) =\n1.4 mM', fontsize=25)
    fig.text(0.91, 0.37, f'I(0) =\n0.7 mM', fontsize=25)
    fig.text(0.91, 0.165, f'I(0) =\n0.0 mM', fontsize=25)
    handles = [plt.Line2D([0], [0], color='b', marker='o', label=f'$NO_3$ (A)'),
                plt.Line2D([0], [0], color='r', marker='o', label=f'$NO_2$ (I)')]
    fig.legend(handles=handles, loc='upper right', fontsize=20)
    fig.suptitle(f'{id} ({wc} %whc), CHL-\nLinear Regression on Early Time Points', fontsize=30, fontweight='bold')
    filename = f'{exp_num}.{id}_linear_regression_no3_no2_conc_chl0.png'
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    print(f'Saved plots/{filename}')
    plt.close()



