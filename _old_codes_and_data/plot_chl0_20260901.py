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
    for i in range(len(cons)):
        if cons[i] > upper_bound - tol:
            return i
    return len(cons)


def pre_plateau_rates(time, upper_bound, cons):
    etp = end_time_point(upper_bound, cons)
    if etp == 0:
        return np.nan, np.nan
    k = max(etp + 1, 2)

    t = time[:k]
    y = cons[:k]
    rate_zero = np.dot(t, y) / np.dot(t, t)
    rate_free, intercept_free = np.polyfit(t, y, 1)

    return rate_zero, rate_free

def estimated_consumption_time(time, upper_bound, cons, r):
    if upper_bound == 0:
        return np.nan

    target = r * upper_bound

    k = None
    for i in range(1, len(cons)):
        if cons[i - 1] < target <= cons[i]:
            k = i
            break

    if k is None:
        t0, t1 = time[-2], time[-1]
        c0, c1 = cons[-2], cons[-1]
        slope = (c1 - c0) / (t1 - t0)
        return t1 + (target - c1) / slope

    t0, t1 = time[k - 1], time[k]
    c0, c1 = cons[k - 1], cons[k]
    frac = (target - c0) / (c1 - c0)
    return t0 + frac * (t1 - t0)


def plot_illustration(id, analyte, rows):
    no3_conc_df = data_dict[id]['no3_conc']
    no2_conc_df = data_dict[id]['no2_conc']
    conc_df = data_dict[id][f'{analyte}_conc']
    cons_df = data_dict[id][f'{analyte}_cons']
    time = conc_df.columns.values.astype(float)

    quantities = ['rate_zero_intercept', 'rate_free_intercept', 'auc', 'half_cons_time', 'full_cons_time']
    titles = {
        'rate_zero_intercept': 'Rate (zero intercept)',
        'rate_free_intercept': 'Rate (free intercept)',
        'auc': 'AUC',
        'half_cons_time': 'Half consumption time',
        'full_cons_time': 'Full consumption time',
    }

    fig, axes = plt.subplots(len(rows), len(quantities), figsize=(4 * len(quantities), 3.2 * len(rows)))

    for r_i, row_id in enumerate(rows):
        cons = cons_df.loc[row_id].values.astype(float)
        no3_conc0 = no3_conc_df.loc[row_id].values.astype(float)[0]
        no2_conc0 = no2_conc_df.loc[row_id].values.astype(float)[0]
        upper_bound = no3_conc0 if analyte == 'no3' else no3_conc0 + no2_conc0

        k = max(end_time_point(upper_bound, cons) + 1, 2)
        t_pre, y_pre = time[:k], cons[:k]

        for c_i, qty in enumerate(quantities):
            ax = axes[r_i, c_i]

            if qty == 'auc':
                ax.plot(time, cons, 'o', color='gray', alpha=0.4, markersize=4)
                ax.plot(t_pre, y_pre, 'o-', color='tab:blue', markersize=4, zorder=3)
                ax.fill_between(t_pre, y_pre, 0, color='tab:blue', alpha=0.25)
                auc = np.trapezoid(y_pre, t_pre)
                ax.set_ylabel(f'{analyte} cons (mM)')
                ax.text(0.97, 0.95, f'AUC = {auc:.3f} mM\N{MIDDLE DOT}h', transform=ax.transAxes,
                        ha='right', va='top', fontsize=9, color='tab:blue')

            elif qty in ('rate_zero_intercept', 'rate_free_intercept'):
                ax.plot(time, cons, 'o', color='gray', alpha=0.4, markersize=4)
                ax.plot(t_pre, y_pre, 'o', color='tab:red', markersize=6, zorder=3)
                t_line = np.array([0, t_pre[-1]])
                if qty == 'rate_zero_intercept':
                    rate = np.dot(t_pre, y_pre) / np.dot(t_pre, t_pre)
                    ax.plot(t_line, rate * t_line, '--', color='tab:red')
                    label = f'rate = {rate:.4f} mM/h\n(forced through 0)'
                else:
                    rate, intercept = np.polyfit(t_pre, y_pre, 1)
                    ax.plot(t_line, rate * t_line + intercept, '--', color='tab:red')
                    label = f'rate = {rate:.4f} mM/h\nintercept = {intercept:.3f}'
                ax.text(0.97, 0.05, label, transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=9, color='tab:red')
                ax.set_ylabel(f'{analyte} cons (mM)')

            elif qty in ('half_cons_time', 'full_cons_time'):
                r_target = 0.5 if qty == 'half_cons_time' else 0.98
                t_target = estimated_consumption_time(time, upper_bound, cons, r_target)
                y_target = r_target * upper_bound
                ax.plot(time, cons, 'o-', color='tab:green', markersize=4)
                ax.axhline(y_target, color='tab:orange', linestyle=':', linewidth=1)
                ax.axvline(t_target, color='tab:orange', linestyle=':', linewidth=1)
                ax.plot(t_target, y_target, 'D', color='tab:orange', markersize=8, zorder=3)
                ax.text(0.97, 0.05, f't = {t_target:.2f} h', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=9, color='tab:orange')
                ax.set_ylabel(f'{analyte} cons (mM)')

            ax.set_xlabel('time (h)')
            if r_i == 0:
                ax.set_title(titles[qty], fontsize=12)
            if c_i == 0:
                ax.annotate(row_id, xy=(-0.35, 0.5), xycoords='axes fraction', fontsize=13,
                            fontweight='bold', ha='center', va='center', rotation=90)

    fig.suptitle(f'{id} / {analyte} consumption : how each quantity is computed', fontsize=20, y=1.02)
    fig.tight_layout()
    fn = f'plots/{id}_{analyte}_chl0_quantitiy_illustration_{datetime.datetime.now().strftime("%Y%m%d")}.png'
    fig.savefig(fn, dpi=200, bbox_inches='tight')
    print(f'Figure saved to {fn}')
    plt.close(fig)
    return fn


# rows = ['E04', 'F04', 'G04', 'H04']
# for id in ids:
#     plot_illustration(id, 'no3', rows)
#     plot_illustration(id, 'no2', rows)

SUBSTRATE_EPS = 0.05  # below this, [analyte]_conc[0] is treated as "not present" (assay noise floor)

for id in ids:
    time = data_dict[id]['no3_conc'].columns.values.astype(float)
    for key in ['rate_zero_intercept', 'rate_free_intercept', 'auc', 'half_cons_time', 'full_cons_time']:
        data_dict[id][key] = pd.DataFrame(index=data_dict[id]['no3_conc'].index, columns=['no3', 'no2'])

    no3_conc_chl0 = data_dict[id]['no3_conc'][mask_chl0]
    no2_conc_chl0 = data_dict[id]['no2_conc'][mask_chl0]
    no3_cons_chl0 = data_dict[id]['no3_cons'][mask_chl0]
    no2_cons_chl0 = data_dict[id]['no2_cons'][mask_chl0]
    for index in no3_conc_chl0.index:
        no3_conc = no3_conc_chl0.loc[index].values.astype(float)
        no2_conc = no2_conc_chl0.loc[index].values.astype(float)
        no3_cons = no3_cons_chl0.loc[index].values.astype(float)
        no2_cons = no2_cons_chl0.loc[index].values.astype(float)

        no3_present = no3_conc[0] >= SUBSTRATE_EPS
        no2_present = no3_present or no2_conc[0] >= SUBSTRATE_EPS

        if no3_present:
            rate_zero_no3, rate_free_no3 = pre_plateau_rates(time, no3_conc[0], no3_cons)
            no3_k = end_time_point(no3_conc[0], no3_cons) + 1
            auc_no3 = np.trapezoid(no3_cons[:no3_k], time[:no3_k])
            half_no3 = estimated_consumption_time(time, no3_conc[0], no3_cons, 0.5)
            full_no3 = estimated_consumption_time(time, no3_conc[0], no3_cons, 0.98)
        else:
            rate_zero_no3 = rate_free_no3 = auc_no3 = half_no3 = full_no3 = np.nan

        if no2_present:
            no2_upper_bound = no2_conc[0] + no3_conc[0]
            rate_zero_no2, rate_free_no2 = pre_plateau_rates(time, no2_upper_bound, no2_cons)
            no2_k = end_time_point(no2_upper_bound, no2_cons) + 1
            auc_no2 = np.trapezoid(no2_cons[:no2_k], time[:no2_k])
            half_no2 = estimated_consumption_time(time, no2_upper_bound, no2_cons, 0.5)
            full_no2 = estimated_consumption_time(time, no2_upper_bound, no2_cons, 0.98)
        else:
            rate_zero_no2 = rate_free_no2 = auc_no2 = half_no2 = full_no2 = np.nan

        data_dict[id]['rate_zero_intercept'].loc[index, 'no3'] = rate_zero_no3
        data_dict[id]['rate_free_intercept'].loc[index, 'no3'] = rate_free_no3
        data_dict[id]['rate_zero_intercept'].loc[index, 'no2'] = rate_zero_no2
        data_dict[id]['rate_free_intercept'].loc[index, 'no2'] = rate_free_no2

        data_dict[id]['auc'].loc[index, 'no3'] = auc_no3
        data_dict[id]['auc'].loc[index, 'no2'] = auc_no2

        data_dict[id]['half_cons_time'].loc[index, 'no3'] = half_no3
        data_dict[id]['half_cons_time'].loc[index, 'no2'] = half_no2
        data_dict[id]['full_cons_time'].loc[index, 'no3'] = full_no3
        data_dict[id]['full_cons_time'].loc[index, 'no2'] = full_no2


def plot_chl0_mean_sem(quantity_key, analyte, quantity_label, ylabel):
    add_col = 'Nitrate_input' if analyte == 'no3' else 'Nitrite_input'
    add_symbol = 'A' if analyte == 'no3' else 'I'
    analyte_label = 'Nitrate' if analyte == 'no3' else 'Nitrite'

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    min_gap = np.min(np.diff(sorted(days_of_drought)))
    offset_unit = 0.15 * min_gap

    for i, id in enumerate(ids):
        x_batch = days_of_drought[i]
        values_all = data_dict[id][quantity_key].loc[mask_chl0, analyte].astype(float)
        metadata = data_dict[id]['metadata']
        for add_i, add_val in enumerate(add_conc):
            group_index = metadata.index[metadata[add_col] == add_val].intersection(values_all.index)
            values = values_all.loc[group_index].dropna()
            if len(values) == 0:
                continue
            mean = values.mean()
            sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            x = x_batch # + offset_unit * (add_i - 1.5)
            ax.errorbar(x, mean, yerr=sem, fmt=markers[add_i], color=batch_colors[i],
                        markersize=6, capsize=4, elinewidth=1.5, markeredgecolor='black', markeredgewidth=0.5, alpha=0.5)

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title(f'{analyte_label} {quantity_label} (CHL-)' + '\n' +
                 rf'mean $\pm$ SEM | colour = batch | marker = ${add_symbol}_{{add}}$', fontsize=14)
    ax.set_xlabel('Days of drought')
    ax.set_ylabel(ylabel)
    custom_lines = [Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=8, label=f'${add_symbol}_{{add}}$ = {conc} mM') for marker, conc in zip(markers, add_conc)]
    ax.legend(handles=custom_lines)
    ax.grid(True, alpha=0.5)
    fn = f'plots/4.2.chl0_{analyte}_{quantity_key}_mean_sem_{datetime.datetime.now().strftime("%Y%m%d")}.png'
    fig.savefig(fn, dpi=300, bbox_inches='tight')
    print(f'Figure saved to {fn}')
    plt.close(fig)
    return fn


mean_sem_plot_specs = [
    ('rate_zero_intercept', 'consumption rate (zero intercept)', 'Rate (mM/h)'),
    ('rate_free_intercept', 'consumption rate (free intercept)', 'Rate (mM/h)'),
    ('auc', 'AUC', 'AUC (mM\N{MIDDLE DOT}h)'),
    ('half_cons_time', 'half consumption time', 'Time (h)'),
    ('full_cons_time', 'full consumption time', 'Time (h)'),
]

for quantity_key, quantity_label, ylabel in mean_sem_plot_specs:
    for analyte in ['no3', 'no2']:
        plot_chl0_mean_sem(quantity_key, analyte, quantity_label, ylabel)
