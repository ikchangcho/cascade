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


def fit_exp_growth(time, cons, upper_bound, s=None, gamma_bounds=(-5.0, 5.0), s_min=1e-3):
    """Fit s/gamma * (exp(gamma*t) - 1) to cons vs time, using points up to end_time_point(upper_bound, cons).
    If s is given it is held fixed and only gamma is fit; otherwise s and gamma are fit jointly.
    If s is given and is negative or below s_min, skip fitting entirely (the model has no traction
    when the fixed rate is ~0, and the optimizer just runs gamma to the bound trying to compensate).
    Returns (s_fit, gamma, t_fit, y_fit, rmse)."""

    etp = end_time_point(upper_bound, cons)
    k = max(etp + 1, 2)
    t_fit, y_fit = time[:k], cons[:k]

    if s is not None and s < s_min:
        return np.nan, np.nan, t_fit, y_fit, np.nan

    if s is not None:
        def exp_func(t, gamma_fit):
            return exp_growth(t, s, gamma_fit)

        try:
            (gamma_fit,), _ = curve_fit(exp_func, t_fit, y_fit, p0=[0.1], maxfev=5000, bounds=gamma_bounds)
        except RuntimeError:
            gamma_fit = np.nan
        s_fit = s
    else:
        try:
            (s_fit, gamma_fit), _ = curve_fit(exp_growth, t_fit, y_fit, p0=[0.1, 0.1], maxfev=5000,
                                           bounds=([-10.0, gamma_bounds[0]], [10.0, gamma_bounds[1]]))
        except RuntimeError:
            s_fit, gamma_fit = np.nan, np.nan

    if not np.isnan(s_fit) and not np.isnan(gamma_fit):
        rmse = np.sqrt(np.mean((y_fit - exp_growth(t_fit, s_fit, gamma_fit)) ** 2))
    else:
        rmse = np.nan

    return s_fit, gamma_fit, t_fit, y_fit, rmse


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
