import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D

# Create an array of the intervals between Nov 24 12:56, Dec 1 13:00, Dec 8 9:35, Dec 15 12:58, Dec 19 11:00, Jan 14 11:15, in days
datetime_array = [
    datetime.datetime(2024, 11, 24, 12, 56),
    datetime.datetime(2024, 12, 1, 13, 0),
    datetime.datetime(2024, 12, 8, 9, 35),
    datetime.datetime(2024, 12, 19, 11, 0),
    datetime.datetime(2025, 1, 14, 11, 15),
    datetime.datetime(2025, 2, 23, 11, 37)]
days_of_drought = [0]
for i in range(1, len(datetime_array)):
    time_diff = datetime_array[i] - datetime_array[0]
    days_of_drought.append(time_diff.total_seconds() / 3600 / 24)

water_contents =  [98.9, 62.5, 34.4, 7.44, 7.10, 5.16, 4.74]


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

mask_chl1 = (data_dict['4.2.batch1']['metadata']['Chloramphenicol'] == 1.0) & (data_dict['4.2.batch1']['metadata']['Sample_type'] != 'Blank')
mask_chl0 = (data_dict['4.2.batch1']['metadata']['Chloramphenicol'] == 0.0) & (data_dict['4.2.batch1']['metadata']['Sample_type'] != 'Blank')


def end_time_point(upper_bound, cons, tol=0.02):
    "Index of time point where the consumption reaches the upper bound."
    for i in range(len(cons)):
        if cons[i] > upper_bound - tol:
            return i
    return len(cons)


analyte_labels = {'no3_cons': 'Nitrate', 'no2_cons': 'Nitrite'}


def plot_cons_chl_compare_ax(ax, id, chl1_row, chl0_row, analyte='no3_cons', n_lin=6, legend=True, fix_s=True, upper_bound_tol=0.2):
    "Draw the CHL+ vs CHL- consumption comparison (with fits) for a matched pair of wells onto ax."
    cons_df = data_dict[id][analyte]
    no3_conc_df = data_dict[id]['no3_conc']
    no2_conc_df = data_dict[id]['no2_conc']
    time = cons_df.columns.values.astype(float)

    rows = {chl1_row: ('CHL+', 'tab:purple'), chl0_row: ('CHL-', 'tab:orange')}

    cons_chl1 = cons_df.loc[chl1_row].values.astype(float)
    cons_chl0 = cons_df.loc[chl0_row].values.astype(float)
    no3_conc0 = no3_conc_df.loc[chl0_row].values.astype(float)[0]
    no2_conc0 = no2_conc_df.loc[chl0_row].values.astype(float)[0]
    upper_bound = no3_conc0 if analyte == 'no3_cons' else no3_conc0 + no2_conc0

    exp_formula_label = 'y = $\\frac{s}{\\gamma}(e^{\\gamma t} - 1)$'

    if upper_bound < upper_bound_tol:
        # Upper bound too small (e.g. A_add=0): skip fitting entirely.
        slope, s_fit, gamma = np.nan, np.nan, np.nan
        t_exp = time
        n_used = {chl1_row: len(time), chl0_row: len(time)}
    else:
        # Linear fit through zero intercept, on the first n_lin points, for the CHL+ condition
        t_lin, y_lin = time[:n_lin], cons_chl1[:n_lin]
        slope = np.dot(t_lin, y_lin) / np.dot(t_lin, t_lin)

        # Exponential fit (s/gamma) * (exp(gamma * t) - 1) for the CHL- condition, either with s fixed
        # to the linear-fit slope, or with s free (fit jointly with gamma)
        etp = end_time_point(upper_bound, cons_chl0)
        k = max(etp + 1, 2)
        t_exp, y_exp = time[:k], cons_chl0[:k]

        def exp_growth(t, s, gamma):
            "s/gamma * (exp(gamma*t) - 1), safe for gamma near 0 (limit -> s*t)."
            return np.where(np.abs(gamma) < 1e-8, s * t, s / np.where(gamma == 0, 1, gamma) * (np.exp(gamma * t) - 1))

        gamma_bounds = (-5.0, 5.0)

        if fix_s:
            def exp_func(t, gamma):
                return exp_growth(t, slope, gamma)

            try:
                (gamma,), _ = curve_fit(exp_func, t_exp, y_exp, p0=[0.1], maxfev=5000, bounds=gamma_bounds)
            except RuntimeError:
                gamma = np.nan
            s_fit = slope
        else:
            def exp_func_free(t, s, gamma):
                return exp_growth(t, s, gamma)

            try:
                (s_fit, gamma), _ = curve_fit(exp_func_free, t_exp, y_exp, p0=[slope, 0.1], maxfev=5000,
                                               bounds=([-10.0, gamma_bounds[0]], [10.0, gamma_bounds[1]]))
            except RuntimeError:
                s_fit, gamma = slope, np.nan

            def exp_func(t, gamma):
                return exp_func_free(t, s_fit, gamma)

        n_used = {chl1_row: n_lin, chl0_row: k}

    for row_id, (label, color) in rows.items():
        cons = cons_df.loc[row_id].values.astype(float)
        n = n_used[row_id]
        ax.scatter(time[:n], cons[:n], color=color, s=50, label=label)
        ax.scatter(time[n:], cons[n:], color=color, s=50, alpha=0.25)

    if not np.isnan(slope):
        t_line = np.array([0, time[-1]])
        ax.plot(t_line, slope * t_line, '--', color='tab:purple', label='y = st')
    else:
        ax.plot([], [], '--', color='tab:purple', label='y = st')

    if not np.isnan(gamma):
        t_curve = np.linspace(0, t_exp[-1], 100)
        ax.plot(t_curve, exp_func(t_curve, gamma), '--', color='tab:orange', label=exp_formula_label)
    else:
        ax.plot([], [], '--', color='tab:orange', label=exp_formula_label)

    ax.grid(True, alpha=0.3)
    main_legend = ax.legend(fontsize=13) if legend else None

    s_str = f'{s_fit:.4f}' if not np.isnan(s_fit) else 'NaN'
    gamma_str = f'{gamma:.4f}' if not np.isnan(gamma) else 'NaN'
    value_handles = [Line2D([], [], linestyle='none', label=f's = {s_str}'),
                     Line2D([], [], linestyle='none', label=f'$\\gamma$ = {gamma_str}')]
    ax.legend(handles=value_handles, fontsize=11, loc='lower right',
              frameon=False, handlelength=0, handletextpad=0)
    if main_legend is not None:
        ax.add_artist(main_legend)

    return slope, s_fit, gamma


def plot_cons_chl_compare(id, chl1_row, chl0_row, analyte='no3_cons', n_lin=6, fix_s=True):
    "Compare consumption with vs without chloramphenicol for a matched pair of wells."
    metadata = data_dict[id]['metadata']
    analyte_label = analyte_labels[analyte]

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_cons_chl_compare_ax(ax, id, chl1_row, chl0_row, analyte=analyte, n_lin=n_lin, fix_s=fix_s)

    a_add = metadata.loc[chl1_row, 'Nitrate_input']
    i_add = metadata.loc[chl1_row, 'Nitrite_input']
    batch_label = batch_labels[ids.index(id)]

    ax.set_xlabel('Time (h)', fontsize=18)
    ax.set_ylabel(f'{analyte_label} consumption (mM)', fontsize=18)
    ax.set_title(f'{analyte_label} consumption: CHL+ vs CHL-\n{batch_label}, $A_{{add}}$={a_add}, $I_{{add}}$={i_add}', fontsize=20)
    ax.tick_params(axis='both', labelsize=16)

    fig.tight_layout()
    s_tag = 'fixed_s' if fix_s else 'free_s'
    fn = f'plots/{id}_{analyte}_chl_compare_{chl1_row}_{chl0_row}_{s_tag}_{datetime.datetime.now().strftime("%Y%m%d")}.png'
    fig.savefig(fn, dpi=200, bbox_inches='tight')
    print(f'Figure saved to {fn}')
    plt.close(fig)
    return fn


def plot_cons_chl_compare_grid(pairs, analyte='no3_cons', n_lin=6, fix_s=True):
    "Grid of CHL+ vs CHL- consumption comparisons: rows = well pairs, columns = batches."
    analyte_label = analyte_labels[analyte]
    n_rows, n_cols = len(pairs), len(ids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), sharex=True, sharey=True)

    for r_i, (chl1_row, chl0_row) in enumerate(pairs):
        for c_i, id in enumerate(ids):
            ax = axes[r_i, c_i]
            plot_cons_chl_compare_ax(ax, id, chl1_row, chl0_row, analyte=analyte, n_lin=n_lin, legend=False, fix_s=fix_s)

            ax.tick_params(axis='both', labelsize=13)
            if r_i == n_rows - 1:
                ax.set_xlabel('Time (h)', fontsize=15)
            if c_i == 0:
                metadata = data_dict[ids[0]]['metadata']
                a_add = metadata.loc[chl1_row, 'Nitrate_input']
                i_add = metadata.loc[chl1_row, 'Nitrite_input']
                ax.set_ylabel(f'{analyte_label} consumption (mM)', fontsize=15)
                ax.annotate(f'$A_{{add}}$={a_add}, $I_{{add}}$={i_add}',
                            xy=(-0.45, 0.5), xycoords='axes fraction', fontsize=17, fontweight='bold',
                            ha='center', va='center', rotation=90, rotation_mode='anchor')
            if r_i == 0:
                ax.set_title(batch_labels[c_i], fontsize=18)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=22, bbox_to_anchor=(0.5, 1.02))
    s_desc = 's fixed to CHL+ linear-fit slope' if fix_s else 's free'
    fig.suptitle(f'{analyte_label} consumption: CHL+ vs CHL- ({s_desc})', fontsize=24, y=1.05)
    fig.tight_layout()
    s_tag = 'fixed_s' if fix_s else 'free_s'
    fn = f'plots/4.2.{analyte}_chl_compare_grid_{s_tag}_{datetime.datetime.now().strftime("%Y%m%d")}.png'
    fig.savefig(fn, dpi=200, bbox_inches='tight')
    print(f'Figure saved to {fn}')
    plt.close(fig)
    return fn


well_pairs = [('A04', 'E04'), ('B04', 'F04'), ('C04', 'G04'), ('D04', 'H04'), ('B01', 'F01'), ('B10', 'F10')]
analyte_n_lin = {'no3_cons': 6, 'no2_cons': 8}
for analyte, n_lin in analyte_n_lin.items():
    plot_cons_chl_compare_grid(well_pairs, analyte=analyte, n_lin=n_lin, fix_s=False)
    plot_cons_chl_compare_grid(well_pairs, analyte=analyte, n_lin=n_lin, fix_s=True)

