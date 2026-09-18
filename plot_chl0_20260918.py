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


def exp_growth(t, s, gamma):
    "s/gamma * (exp(gamma*t) - 1), safe for gamma near 0 (limit -> s*t)."
    return np.where(np.abs(gamma) < 1e-8, s * t, s / np.where(gamma == 0, 1, gamma) * (np.exp(gamma * t) - 1))


def fit_exp_growth(time, cons, upper_bound, s=None, gamma_bounds=(-5.0, 5.0), s_min=1e-3, exclude_last=False):
    """Fit s/gamma * (exp(gamma*t) - 1) to cons vs time, using points up to end_time_point(upper_bound, cons).
    If s is given it is held fixed and only gamma is fit; otherwise s and gamma are fit jointly.
    If s is given and is negative or below s_min, skip fitting entirely (the model has no traction
    when the fixed rate is ~0, and the optimizer just runs gamma to the bound trying to compensate).
    If exclude_last, the last point of that fit window is dropped before fitting (still returned in
    t_fit/y_fit so callers can still show it, e.g. as an excluded point).
    Returns (s_fit, gamma, t_fit, y_fit, rmse)."""

    etp = end_time_point(upper_bound, cons)
    k = max(etp + 1, 2)
    if exclude_last and k > 2:
        k -= 1
    t_fit, y_fit = time[:k], cons[:k]

    if s is not None and s < s_min:
        return np.nan, np.nan, t_fit, y_fit, np.nan

    if s is not None:
        def exp_func(t, gamma_fit):
            return exp_growth(t, s, gamma_fit)

        try:
            (gamma_fit,), _ = curve_fit(exp_func, t_fit, y_fit, p0=[0.1], maxfev=5000, bounds=gamma_bounds)
        except (RuntimeError, ValueError):
            gamma_fit = np.nan
        s_fit = s
    else:
        try:
            (s_fit, gamma_fit), _ = curve_fit(exp_growth, t_fit, y_fit, p0=[0.1, 0.1], maxfev=5000,
                                           bounds=([-10.0, gamma_bounds[0]], [10.0, gamma_bounds[1]]))
        except (RuntimeError, ValueError):
            s_fit, gamma_fit = np.nan, np.nan

    if not np.isnan(s_fit) and not np.isnan(gamma_fit):
        rmse = np.sqrt(np.mean((y_fit - exp_growth(t_fit, s_fit, gamma_fit)) ** 2))
    else:
        rmse = np.nan

    return s_fit, gamma_fit, t_fit, y_fit, rmse


def fit_linear_chl0(time, cons, upper_bound):
    """Through-origin OLS on CHL- data, excluding end_time_point.
    Same window as fit_exp_growth(..., exclude_last=True). Returns (slope, k, rmse)."""
    etp = end_time_point(upper_bound, cons)
    k = max(etp, 2)
    t, y = time[:k], cons[:k]
    slope = np.dot(t, y) / np.dot(t, t)
    return slope, k, np.sqrt(np.mean((y - slope * t) ** 2))


for id in ids:
    time = data_dict[id]['no3_cons'].columns.astype(float)
    data_dict[id]['rates_chl01'] = pd.DataFrame(index=data_dict[id]['no3_cons'][mask_chl1].index, 
                                                columns=['no3_chl1_slope_six_points', 'no3_chl0_exponent', 'no3_chl1_rmse', 'no3_chl0_rmse',
                                                'no2_chl1_slope_eight_points', 'no2_chl0_exponent', 'no2_chl1_rmse', 'no2_chl0_rmse'])
    for i in range(len(data_dict[id]['no3_cons'][mask_chl1])):
        index_chl1 = data_dict[id]['no3_cons'][mask_chl1].index[i]
        index_chl0 = data_dict[id]['no3_cons'][mask_chl0].index[i]

        init_no3_chl1 = data_dict[id]['no3_conc'].loc[index_chl1].values[0]
        init_no3_chl0 = data_dict[id]['no3_conc'].loc[index_chl0].values[0]
        init_no2_chl1 = data_dict[id]['no2_conc'].loc[index_chl1].values[0]
        init_no2_chl0 = data_dict[id]['no2_conc'].loc[index_chl0].values[0]

        no3_chl1 = data_dict[id]['no3_cons'].loc[index_chl1].values.astype(float)
        no3_chl0 = data_dict[id]['no3_cons'].loc[index_chl0].values.astype(float)
        no2_chl1 = data_dict[id]['no2_cons'].loc[index_chl1].values.astype(float)
        no2_chl0 = data_dict[id]['no2_cons'].loc[index_chl0].values.astype(float)

        init_cut_off = 0.2
        if init_no3_chl1 + init_no2_chl1 < init_cut_off or init_no3_chl0 + init_no2_chl0 < init_cut_off:
            no3_chl1_slope, no3_chl0_exponent, no3_chl1_rmse, no3_chl0_rmse = np.nan, np.nan, np.nan, np.nan
            no2_chl1_slope, no2_chl0_exponent, no2_chl1_rmse, no2_chl0_rmse = np.nan, np.nan, np.nan, np.nan

        elif init_no3_chl1 < init_cut_off or init_no3_chl0 < init_cut_off:
            no3_chl1_slope, no3_chl0_exponent, no3_chl1_rmse, no3_chl0_rmse = np.nan, np.nan, np.nan, np.nan

            no2_chl1_slope = np.dot(time[:8], no2_chl1[:8]) / np.dot(time[:8], time[:8])
            no2_chl1_rmse = np.sqrt(np.mean((no2_chl1[:8] - no2_chl1_slope * time[:8]) ** 2))
            _, no2_chl0_exponent, _, _, no2_chl0_rmse = fit_exp_growth(time, no2_chl0, init_no2_chl0 + init_no3_chl0, s=no2_chl1_slope)

        else:
            no3_chl1_slope = np.dot(time[:6], no3_chl1[:6]) / np.dot(time[:6], time[:6])
            no3_chl1_rmse = np.sqrt(np.mean((no3_chl1[:6] - no3_chl1_slope * time[:6]) ** 2))
            _, no3_chl0_exponent, _, _, no3_chl0_rmse = fit_exp_growth(time, no3_chl0, init_no3_chl0, s=no3_chl1_slope)

            no2_chl1_slope = np.dot(time[:8], no2_chl1[:8]) / np.dot(time[:8], time[:8])
            no2_chl1_rmse = np.sqrt(np.mean((no2_chl1[:8] - no2_chl1_slope * time[:8]) ** 2))
            _, no2_chl0_exponent, _, _, no2_chl0_rmse = fit_exp_growth(time, no2_chl0, init_no2_chl0 + init_no3_chl0, s=no2_chl1_slope)
        
        data_dict[id]['rates_chl01'].loc[index_chl1] = [no3_chl1_slope, no3_chl0_exponent, no3_chl1_rmse, no3_chl0_rmse,
                                                        no2_chl1_slope, no2_chl0_exponent, no2_chl1_rmse, no2_chl0_rmse]


quantity_info = {
    'growth_rate': {
        'columns': {'no3': 'no3_chl0_exponent', 'no2': 'no2_chl0_exponent'},
        'name': 'Growth rate', 'symbol': r'\gamma', 'unit': '1/h', 'condition': 'CHL-', 'file_tag': 'growth_rate',
    },
    'slope': {
        'columns': {'no3': 'no3_chl1_slope_six_points', 'no2': 'no2_chl1_slope_eight_points'},
        'name': 'Slope', 'symbol': 's', 'unit': 'mM/h', 'condition': 'CHL+', 'file_tag': 'slope',
    },
}
analyte_labels = {'no3': 'Nitrate (A)', 'no2': 'Nitrite (I)'}
add_var_info = {'A_add': ('Nitrate_input', 'A'), 'I_add': ('Nitrite_input', 'I')}


def plot_quantity_vs_drought(quantity):
    "Quantity (growth rate or slope) vs days of drought for no3 and no2, one point per batch, faint replicates behind."
    info = quantity_info[quantity]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, analyte in zip(axes, analyte_labels):
        col = info['columns'][analyte]
        for i, id in enumerate(ids):
            x_batch = days_of_drought[i]
            values = data_dict[id]['rates_chl01'][col].astype(float).dropna()

            ax.scatter([x_batch] * len(values), values, color=batch_colors[i], alpha=0.2, s=50)

            median = values.median()
            ax.scatter(x_batch, median, marker='D', color=batch_colors[i], s=100, zorder=3,
                       edgecolor='black', linewidth=0.8, label=f'{batch_labels[i]} (N={len(values)})')

        ax.set_xlabel('Days of drought', fontsize=15)
        ax.set_ylabel(rf"{info['name']} ${info['symbol']}$ ({info['unit']})", fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.set_title(analyte_labels[analyte], fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle(rf"{info['name']} ${info['symbol']}$ vs days of drought ({info['condition']}) | diamond = median", fontsize=18)
    fig.tight_layout()
    fn = f"plots/4.2.chl0_{info['file_tag']}_vs_drought_{datetime.datetime.now().strftime('%Y%m%d')}.png"
    fig.savefig(fn, dpi=300, bbox_inches='tight')
    print(f'Figure saved to {fn}')
    plt.close(fig)
    return fn


def plot_quantity_vs_add(quantity, analyte, add_var):
    "Quantity (growth rate or slope) vs the given add concentration, one subplot per batch."
    info = quantity_info[quantity]
    add_col, symbol_add = add_var_info[add_var]
    col = info['columns'][analyte]
    analyte_label = analyte_labels[analyte]

    all_values = pd.concat([data_dict[id]['rates_chl01'][col].astype(float) for id in ids]).dropna()
    y_span = all_values.max() - all_values.min()
    y_pad = 0.05 * y_span if y_span > 0 else 1.0
    y_lim = (all_values.min() - y_pad, all_values.max() + y_pad)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(rf"{info['name']} ${info['symbol']}$ vs ${symbol_add}_{{add}}$ ({analyte_label} {info['condition']}) | error bar = mean $\pm$ SEM", fontsize=22)

    for i, id in enumerate(ids):
        ax = axes[i // 3, i % 3]
        ax.set_xlabel(rf'${symbol_add}_{{add}}$ (mM)' if i >= 3 else '', fontsize=15)
        ax.set_xticks(add_conc)
        ax.set_xlim(-0.1, 2.1)
        ax.set_ylim(y_lim)
        ax.set_ylabel(rf"{info['name']} ${info['symbol']}$ ({info['unit']})" if i % 3 == 0 else '', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)

        values = data_dict[id]['rates_chl01'][col].astype(float)
        metadata = data_dict[id]['metadata'].loc[values.index]

        mean_vals, sem_vals = [], []
        for add_val in add_conc:
            group = values.loc[metadata[add_col] == add_val].dropna()
            mean_vals.append(group.mean())
            sem_vals.append(group.sem())
        ax.errorbar(add_conc, mean_vals, yerr=sem_vals, fmt='o', color=batch_colors[i], alpha=0.7,
                    capsize=5, markersize=8, elinewidth=1.5, zorder=3)

        for add_val in add_conc:
            group = values.loc[metadata[add_col] == add_val].dropna()
            ax.scatter([add_val] * len(group), group, color=batch_colors[i], alpha=0.2, s=50)

        ax.set_title(batch_labels[i], fontsize=15)

    fig.tight_layout()
    fn = f"plots/4.2.chl0_{info['file_tag']}_{analyte}_vs_{symbol_add}_add_{datetime.datetime.now().strftime('%Y%m%d')}.png"
    fig.savefig(fn, dpi=300, bbox_inches='tight')
    print(f'Figure saved to {fn}')
    plt.close(fig)
    return fn


for quantity in quantity_info:
    plot_quantity_vs_drought(quantity)
    for analyte in analyte_labels:
        for add_var in add_var_info:
            plot_quantity_vs_add(quantity, analyte, add_var)


def plot_rmse_heatmap(rmse_col, analyte_label, fit_label, file_tag):
    "Mean RMSE of the given fit by (A_add, I_add) condition, one heatmap per batch: rows = I_add, columns = A_add."
    matrices = {}
    counts = {}
    for id in ids:
        values = data_dict[id]['rates_chl01'][rmse_col].astype(float)
        metadata = data_dict[id]['metadata'].loc[values.index]
        mat = np.full((len(add_conc), len(add_conc)), np.nan)
        n = np.zeros((len(add_conc), len(add_conc)), dtype=int)
        for r, i_val in enumerate(add_conc):
            for c, a_val in enumerate(add_conc):
                group = values.loc[(metadata['Nitrite_input'] == i_val) & (metadata['Nitrate_input'] == a_val)].dropna()
                n[r, c] = len(group)
                if len(group) > 0:
                    mat[r, c] = group.mean()
        matrices[id] = mat
        counts[id] = n

    all_vals = np.concatenate([m[~np.isnan(m)] for m in matrices.values()])
    vmin, vmax = (all_vals.min(), all_vals.max()) if len(all_vals) else (0, 1)

    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad('#eeeeee')

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    im = None
    for i, id in enumerate(ids):
        ax = axes[i // 3, i % 3]
        mat, n = matrices[id], counts[id]
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        for r in range(len(add_conc)):
            for c in range(len(add_conc)):
                val = mat[r, c]
                text = f'{val:.2f}\n(n={n[r, c]})' if not np.isnan(val) else '–'
                color = 'white' if (not np.isnan(val) and val > vmin + 0.6 * (vmax - vmin)) else 'black'
                ax.text(c, r, text, ha='center', va='center', fontsize=10, color=color)

        ax.set_xticks(range(len(add_conc)))
        ax.set_xticklabels(add_conc)
        ax.set_yticks(range(len(add_conc)))
        ax.set_yticklabels(add_conc)
        ax.set_xlabel('$A_{add}$ (mM)' if i >= 3 else '', fontsize=13)
        ax.set_ylabel('$I_{add}$ (mM)' if i % 3 == 0 else '', fontsize=13)
        ax.set_title(batch_labels[i], fontsize=15)

    fig.suptitle(f'RMSE of {analyte_label} {fit_label} by ($A_{{add}}$, $I_{{add}}$)', fontsize=20)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='RMSE (mM)')

    fn = f'plots/4.2.chl0_rmse_{file_tag}_heatmap_{datetime.datetime.now().strftime("%Y%m%d")}.png'
    fig.savefig(fn, dpi=300, bbox_inches='tight')
    print(f'Figure saved to {fn}')
    plt.close(fig)
    return fn


rmse_configs = [
    ('no3_chl0_rmse', 'Nitrate (A)', 'CHL- exponential fit', 'no3_chl0'),
    ('no2_chl0_rmse', 'Nitrite (I)', 'CHL- exponential fit', 'no2_chl0'),
    ('no3_chl1_rmse', 'Nitrate (A)', 'CHL+ linear fit', 'no3_chl1'),
    ('no2_chl1_rmse', 'Nitrite (I)', 'CHL+ linear fit', 'no2_chl1'),
]
for rmse_col, analyte_label, fit_label, file_tag in rmse_configs:
    plot_rmse_heatmap(rmse_col, analyte_label, fit_label, file_tag)


chl1_to_chl0 = {id: dict(zip(data_dict[id]['no3_cons'][mask_chl1].index, data_dict[id]['no3_cons'][mask_chl0].index))
                for id in ids}


def plot_cons_chl_compare_ax(ax, id, chl1_row, chl0_row, analyte='no3', legend=True, exclude_last=False):
    "Draw the CHL+ vs CHL- consumption comparison (s fixed to the CHL+ slope) for a matched well pair onto ax."
    cons_df = data_dict[id][f'{analyte}_cons']
    time = cons_df.columns.values.astype(float)
    n_lin = 6 if analyte == 'no3' else 8

    cons_chl1 = cons_df.loc[chl1_row].values.astype(float)
    cons_chl0 = cons_df.loc[chl0_row].values.astype(float)

    no3_conc0 = data_dict[id]['no3_conc'].loc[chl0_row].values.astype(float)[0]
    no2_conc0 = data_dict[id]['no2_conc'].loc[chl0_row].values.astype(float)[0]
    upper_bound = no3_conc0 if analyte == 'no3' else no3_conc0 + no2_conc0

    t_lin, y_lin = time[:n_lin], cons_chl1[:n_lin]
    slope = np.dot(t_lin, y_lin) / np.dot(t_lin, t_lin)
    s_fit, gamma, t_fit, y_fit, rmse = fit_exp_growth(time, cons_chl0, upper_bound, s=slope, exclude_last=exclude_last)
    n_chl0_used = len(t_fit)

    exp_formula_label = r'y = $\frac{s}{\gamma}(e^{\gamma t} - 1)$'

    ax.scatter(time[:n_lin], cons_chl1[:n_lin], color='tab:purple', s=50, label='CHL+')
    ax.scatter(time[n_lin:], cons_chl1[n_lin:], color='tab:purple', s=50, alpha=0.25)
    ax.scatter(time[:n_chl0_used], cons_chl0[:n_chl0_used], color='tab:orange', s=50, label='CHL-')
    ax.scatter(time[n_chl0_used:], cons_chl0[n_chl0_used:], color='tab:orange', s=50, alpha=0.25)

    t_line = np.array([0, time[-1]])
    ax.plot(t_line, slope * t_line, '--', color='tab:purple', label='y = st')

    if not np.isnan(gamma):
        t_curve = np.linspace(0, t_fit[-1], 100)
        ax.plot(t_curve, exp_growth(t_curve, s_fit, gamma), '--', color='tab:orange', label=exp_formula_label)
    else:
        ax.plot([], [], '--', color='tab:orange', label=exp_formula_label)

    ax.grid(True, alpha=0.3)
    main_legend = ax.legend(fontsize=13) if legend else None

    s_str = f'{s_fit:.4f}' if not np.isnan(s_fit) else 'NaN'
    gamma_str = f'{gamma:.4f}' if not np.isnan(gamma) else 'NaN'
    rmse_str = f'{rmse:.3f}' if not np.isnan(rmse) else 'NaN'
    value_handles = [Line2D([], [], linestyle='none', label=f's = {s_str}'),
                     Line2D([], [], linestyle='none', label=f'$\\gamma$ = {gamma_str}'),
                     Line2D([], [], linestyle='none', label=f'rmse = {rmse_str}')]
    ax.legend(handles=value_handles, fontsize=11, loc='lower right',
              frameon=False, handlelength=0, handletextpad=0)
    if main_legend is not None:
        ax.add_artist(main_legend)

    return slope, s_fit, gamma, rmse


def chl0_rmse_by_well(analyte, exclude_last=False):
    "Per-well chl0 exponential-fit rmse (s fixed to the already-computed CHL+ slope), keyed by (id, chl1_row)."
    slope_col = 'no3_chl1_slope_six_points' if analyte == 'no3' else 'no2_chl1_slope_eight_points'
    result = {}
    for id in ids:
        time = data_dict[id]['no3_cons'].columns.astype(float)
        slopes = data_dict[id]['rates_chl01'][slope_col].astype(float).dropna()
        for chl1_row, slope in slopes.items():
            chl0_row = chl1_to_chl0[id][chl1_row]
            cons = data_dict[id][f'{analyte}_cons'].loc[chl0_row].values.astype(float)
            no3_conc0 = data_dict[id]['no3_conc'].loc[chl0_row].values.astype(float)[0]
            no2_conc0 = data_dict[id]['no2_conc'].loc[chl0_row].values.astype(float)[0]
            upper_bound = no3_conc0 if analyte == 'no3' else no3_conc0 + no2_conc0
            _, _, _, _, rmse = fit_exp_growth(time, cons, upper_bound, s=slope, exclude_last=exclude_last)
            if not np.isnan(rmse):
                result[(id, chl1_row)] = rmse
    return result


def plot_worst_fits_grid(analyte, n_top=24, n_cols=4, exclude_last=False):
    "Grid of CHL+ vs CHL- fits for the n_top individual wells with the highest rmse (no grouping by condition)."
    rmse_by_well = chl0_rmse_by_well(analyte, exclude_last=exclude_last)
    cases = sorted(((rmse, id, chl1_row) for (id, chl1_row), rmse in rmse_by_well.items()),
                    key=lambda c: c[0], reverse=True)[:n_top]

    n_rows = int(np.ceil(len(cases) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.4 * n_rows), squeeze=False)

    legend_ax = None
    for k, (rmse_val, id, chl1_row) in enumerate(cases):
        ax = axes[k // n_cols, k % n_cols]
        chl0_row = chl1_to_chl0[id][chl1_row]
        plot_cons_chl_compare_ax(ax, id, chl1_row, chl0_row, analyte=analyte, legend=False, exclude_last=exclude_last)
        legend_ax = ax

        metadata = data_dict[id]['metadata']
        a_val = metadata.loc[chl1_row, 'Nitrate_input']
        i_val = metadata.loc[chl1_row, 'Nitrite_input']
        batch_label = batch_labels[ids.index(id)]
        ax.set_title(f'{batch_label}, {chl1_row}\n$A_{{add}}$={a_val}, $I_{{add}}$={i_val}', fontsize=11)
        ax.tick_params(axis='both', labelsize=9)

    for k in range(len(cases), n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis('off')

    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=18, bbox_to_anchor=(0.5, 1.01))

    excl_tag = ' (last CHL- fit point excluded)' if exclude_last else ''
    fig.text(0.06, 0.5, f'{analyte_labels[analyte]} consumption (mM)', va='center', rotation='vertical', fontsize=15)
    fig.suptitle(f'{analyte_labels[analyte]} consumption: CHL+ vs CHL- (worst {len(cases)} individual fits by rmse, '
                 f's fixed to CHL+ slope{excl_tag})', fontsize=20, y=1.02)
    fig.tight_layout(rect=[0.03, 0, 1, 1])
    excl_suffix = '_excl_last' if exclude_last else ''
    fn = f'plots/4.2.chl0_{analyte}_worst_fits_grid{excl_suffix}_{datetime.datetime.now().strftime("%Y%m%d")}.png'
    fig.savefig(fn, dpi=200, bbox_inches='tight')
    print(f'Figure saved to {fn}')
    plt.close(fig)
    return fn


for analyte in analyte_labels:
    plot_worst_fits_grid(analyte)
    plot_worst_fits_grid(analyte, exclude_last=True)


def plot_a_add_all_wells(analyte='no3', no3_add=2.0):
    "9×6 grid: all A_add=no3_add wells, rows=wells grouped by I_add, columns=batches."
    i_add_vals = [v for v in add_conc if v != no3_add]
    n_rows = len(i_add_vals) * 3  # 3 replicates per I_add level
    n_cols = len(ids)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False)

    for col_idx, id in enumerate(ids):
        metadata = data_dict[id]['metadata']
        rates = data_dict[id]['rates_chl01']
        a_add_rows = rates.index[metadata.loc[rates.index, 'Nitrate_input'] == no3_add]

        row_idx = 0
        for i_val in i_add_vals:
            wells = [r for r in a_add_rows if metadata.loc[r, 'Nitrite_input'] == i_val]
            for chl1_row in wells:
                ax = axes[row_idx, col_idx]
                chl0_row = chl1_to_chl0[id][chl1_row]
                plot_cons_chl_compare_ax(ax, id, chl1_row, chl0_row, analyte=analyte,
                                         legend=(row_idx == 0 and col_idx == 0))
                ax.tick_params(axis='both', labelsize=8)
                if row_idx == 0:
                    ax.set_title(batch_labels[col_idx], fontsize=13, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f'$I_{{add}}$={i_val}\nConsumption (mM)', fontsize=9)
                ax.set_xlabel('Time (h)' if row_idx == n_rows - 1 else '', fontsize=9)
                row_idx += 1

    for i_idx in range(1, len(i_add_vals)):
        y = 1.0 - i_idx / len(i_add_vals)
        fig.add_artist(plt.Line2D([0, 1], [y, y], transform=fig.transFigure,
                                  color='gray', linewidth=0.8, linestyle='--'))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=14, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f'{analyte_labels[analyte]} consumption: all $A_{{add}}$={no3_add} wells | CHL+ vs CHL-',
                 fontsize=20, y=1.015)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fn = f'plots/4.2.chl0_{analyte}_Aadd{no3_add}_all_wells_{datetime.datetime.now().strftime("%Y%m%d")}.png'
    fig.savefig(fn, dpi=150, bbox_inches='tight')
    print(f'Figure saved to {fn}')
    plt.close(fig)
    return fn


plot_a_add_all_wells('no3', no3_add=2.0)


def _lin_vs_exp_config(analyte):
    "Return analyte-specific config dict for the lin-vs-exp comparison plots."
    if analyte == 'no3':
        return dict(
            slope_col='no3_chl1_slope_six_points',
            cons_key='no3_cons', conc_key='no3_conc',
            ub_fn=lambda no3, no2: no3,
            filter_fn=lambda meta, row, fval: abs(meta.loc[row, 'Nitrate_input'] - fval) < 0.01,
            group_fn=lambda meta, rows, fval: [
                (i_val, [r for r in rows if abs(meta.loc[r, 'Nitrite_input'] - i_val) < 0.01])
                for i_val in add_conc if abs(i_val - fval) > 0.01
            ],
            ylabel_fn=lambda a_val, i_val: f'$I_{{add}}$={i_val} mM\nConsumption (mM)',
            filter_tag_fn=lambda fval: f'Aadd{fval}',
            filter_title_fn=lambda fval: f'$A_{{add}}$ = {fval} mM conditions',
            x_col_fn=lambda meta, row: meta.loc[row, 'Nitrate_input'],
            x_label='$A_{add}$ (mM)', x_tag='Aadd',
        )
    else:
        return dict(
            slope_col='no2_chl1_slope_eight_points',
            cons_key='no2_cons', conc_key='no2_conc',
            ub_fn=lambda no3, no2: no3 + no2,
            filter_fn=lambda meta, row, fval: abs(meta.loc[row, 'Nitrate_input'] + meta.loc[row, 'Nitrite_input'] - fval) < 0.01,
            group_fn=lambda meta, rows, fval: [
                ((a, i), [r for r in rows if abs(meta.loc[r, 'Nitrate_input'] - a) < 0.01 and abs(meta.loc[r, 'Nitrite_input'] - i) < 0.01])
                for a in add_conc for i in add_conc if abs(a + i - fval) < 0.01
            ],
            ylabel_fn=lambda a_val, i_val: f'$A_{{add}}$={a_val}, $I_{{add}}$={i_val} mM\nConsumption (mM)',
            filter_tag_fn=lambda fval: f'Atot{fval}',
            filter_title_fn=lambda fval: f'$A_{{add}}+I_{{add}}$ = {fval} mM conditions',
            x_col_fn=lambda meta, row: meta.loc[row, 'Nitrate_input'] + meta.loc[row, 'Nitrite_input'],
            x_label='$A_{add}+I_{add}$ (mM)', x_tag='Atot',
        )


def plot_lin_vs_exp_grid(analyte, filter_val):
    "Grid of CHL- fits (linear vs constrained exponential) for a fixed concentration condition."
    cfg = _lin_vs_exp_config(analyte)
    analyte_label = analyte_labels[analyte]

    # collect groups across all batches to know grid size
    sample_meta = data_dict[ids[0]]['metadata']
    sample_rates = data_dict[ids[0]]['rates_chl01']
    sample_rows = [r for r in sample_rates.index if cfg['filter_fn'](sample_meta, r, filter_val)]
    groups = cfg['group_fn'](sample_meta, sample_rows, filter_val)
    n_rows = len(groups) * 3
    n_cols = len(ids)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows), squeeze=False)

    for col_idx, id in enumerate(ids):
        meta  = data_dict[id]['metadata']
        rates = data_dict[id]['rates_chl01']
        time  = data_dict[id][cfg['cons_key']].columns.astype(float)
        id_rows = [r for r in rates.index if cfg['filter_fn'](meta, r, filter_val)]
        id_groups = cfg['group_fn'](meta, id_rows, filter_val)

        row_idx = 0
        for grp_key, wells in id_groups:
            a_val = grp_key[0] if isinstance(grp_key, tuple) else filter_val
            i_val = grp_key[1] if isinstance(grp_key, tuple) else grp_key
            for chl1_row in wells:
                ax = axes[row_idx, col_idx]
                chl0_row  = chl1_to_chl0[id][chl1_row]
                cons_chl0 = data_dict[id][cfg['cons_key']].loc[chl0_row].values.astype(float)
                no3_0 = data_dict[id]['no3_conc'].loc[chl0_row].values.astype(float)[0]
                no2_0 = data_dict[id]['no2_conc'].loc[chl0_row].values.astype(float)[0]
                ub     = cfg['ub_fn'](no3_0, no2_0)
                slope_chl1 = float(rates.loc[chl1_row, cfg['slope_col']])

                slope_lin, k_lin, rmse_lin = fit_linear_chl0(time, cons_chl0, ub)
                _, _, _, _, rmse_exp = fit_exp_growth(time, cons_chl0, ub, s=slope_chl1, exclude_last=True)
                s_exp, gamma, t_fit, _, _ = fit_exp_growth(time, cons_chl0, ub, s=slope_chl1, exclude_last=True)
                k_exp = len(t_fit)
                k = max(k_lin, k_exp)

                ax.scatter(time[:k], cons_chl0[:k], color='tab:orange', s=35, label='CHL-', zorder=3)
                ax.scatter(time[k:], cons_chl0[k:], color='tab:orange', s=35, alpha=0.2, zorder=2)

                t_lin = np.linspace(0, time[k_lin - 1], 100)
                ax.plot(t_lin, slope_lin * t_lin, '--', color='steelblue', lw=1.8, label=r'$y = s_- t$')
                if not np.isnan(gamma):
                    t_exp = np.linspace(0, t_fit[-1], 100)
                    ax.plot(t_exp, exp_growth(t_exp, s_exp, gamma), '--', color='firebrick', lw=1.8,
                            label=r'$y = \frac{s_+}{\gamma}(e^{\gamma t}-1)$')
                else:
                    ax.plot([], [], '--', color='firebrick', lw=1.8,
                            label=r'$y = \frac{s_+}{\gamma}(e^{\gamma t}-1)$')

                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=10)
                if row_idx == 0:
                    ax.set_title(batch_labels[col_idx], fontsize=13, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(cfg['ylabel_fn'](a_val, i_val), fontsize=11)
                if row_idx == n_rows - 1:
                    ax.set_xlabel('Time (h)', fontsize=11)

                rmse_lin_s = f'{rmse_lin:.3f}' if not np.isnan(rmse_lin) else 'NaN'
                rmse_exp_s = f'{rmse_exp:.3f}' if not np.isnan(rmse_exp) else 'NaN'
                ax.text(0.97, 0.03, f'RMSE (Linear) = {rmse_lin_s}\nRMSE (Exp.) = {rmse_exp_s}',
                        transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                        bbox=dict(fc='white', alpha=0.7, pad=2))
                row_idx += 1

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='upper center', ncol=3, fontsize=14, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f'{analyte_label} Consumption CHL- Fits: Linear vs Constrained Exponential\n'
                 f'{cfg["filter_title_fn"](filter_val)} | faded points are not used for fitting',
                 fontsize=18, y=1.022)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fn = f'plots/4.2.chl0_{analyte}_lin_vs_exp_{cfg["filter_tag_fn"](filter_val)}_{datetime.datetime.now().strftime("%Y%m%d")}.png'
    fig.savefig(fn, dpi=150, bbox_inches='tight')
    print(f'Figure saved to {fn}')
    plt.close(fig)
    return fn


def plot_rmse_lin_vs_exp(analyte):
    "2×3 grid: mean ± SEM RMSE vs concentration for linear and constrained-exp fits, one panel per batch."
    cfg = _lin_vs_exp_config(analyte)
    analyte_label = analyte_labels[analyte]

    records = []
    for id in ids:
        meta  = data_dict[id]['metadata']
        rates = data_dict[id]['rates_chl01']
        time  = data_dict[id][cfg['cons_key']].columns.astype(float)
        for chl1_row in rates.index:
            chl0_row = chl1_to_chl0[id][chl1_row]
            cons   = data_dict[id][cfg['cons_key']].loc[chl0_row].values.astype(float)
            no3_0  = data_dict[id]['no3_conc'].loc[chl0_row].values.astype(float)[0]
            no2_0  = data_dict[id]['no2_conc'].loc[chl0_row].values.astype(float)[0]
            ub     = cfg['ub_fn'](no3_0, no2_0)
            slope_chl1 = float(rates.loc[chl1_row, cfg['slope_col']])
            _, _, rmse_lin = fit_linear_chl0(time, cons, ub)
            _, _, _, _, rmse_exp = fit_exp_growth(time, cons, ub, s=slope_chl1, exclude_last=True)
            records.append({'id': id, 'x': cfg['x_col_fn'](meta, chl1_row),
                            'rmse_lin': rmse_lin, 'rmse_exp': rmse_exp})
    df = pd.DataFrame(records)
    x_vals = sorted(df['x'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=False)
    fig.suptitle(f'{analyte_label} CHL−: RMSE vs {cfg["x_label"]} — Linear vs Constrained Exponential'
                 r' | error bar = mean $\pm$ SEM', fontsize=15)

    for i, id in enumerate(ids):
        ax = axes[i // 3, i % 3]
        sub = df[df['id'] == id].dropna(subset=['rmse_lin', 'rmse_exp'])
        for x_val in x_vals:
            grp = sub[abs(sub['x'] - x_val) < 0.01]
            ax.scatter([x_val - 0.04] * len(grp), grp['rmse_lin'], color='steelblue', alpha=0.3, s=35, zorder=2)
            ax.scatter([x_val + 0.04] * len(grp), grp['rmse_exp'], color='firebrick', alpha=0.3, s=35, zorder=2)
        means_lin = [sub[abs(sub['x'] - x) < 0.01]['rmse_lin'].mean() for x in x_vals]
        sems_lin  = [sub[abs(sub['x'] - x) < 0.01]['rmse_lin'].sem()  for x in x_vals]
        means_exp = [sub[abs(sub['x'] - x) < 0.01]['rmse_exp'].mean() for x in x_vals]
        sems_exp  = [sub[abs(sub['x'] - x) < 0.01]['rmse_exp'].sem()  for x in x_vals]
        xarr = np.array(x_vals)
        ax.errorbar(xarr - 0.04, means_lin, yerr=sems_lin, fmt='o', color='steelblue',
                    capsize=5, markersize=8, lw=0, elinewidth=2, label='Linear', zorder=4)
        ax.errorbar(xarr + 0.04, means_exp, yerr=sems_exp, fmt='s', color='firebrick',
                    capsize=5, markersize=8, lw=0, elinewidth=2, label='Exp.', zorder=4)
        ax.set_title(batch_labels[i], fontsize=14)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f'{v:.1f}' for v in x_vals], rotation=45, fontsize=9)
        ax.set_xlim(min(x_vals) - 0.2, max(x_vals) + 0.2)
        ax.set_xlabel(cfg['x_label'] if i >= 3 else '', fontsize=13)
        ax.set_ylabel('RMSE (mM)' if i % 3 == 0 else '', fontsize=13)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

    fig.tight_layout()
    fn = f'plots/4.2.chl0_{analyte}_rmse_lin_vs_exp_vs_{cfg["x_tag"]}_{datetime.datetime.now().strftime("%Y%m%d")}.png'
    fig.savefig(fn, dpi=150, bbox_inches='tight')
    print(f'Figure saved to {fn}')
    plt.close(fig)
    return fn


for analyte, fval in [('no3', 2.0), ('no2', 3.4)]:
    plot_lin_vs_exp_grid(analyte, fval)
    plot_rmse_lin_vs_exp(analyte)
