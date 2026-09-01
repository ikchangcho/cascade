import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D

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
    if etp > 1:
        k = etp
    elif etp == 1:
        k = 2
    elif etp == 0:
        return np.nan, np.nan

    t = time[:k]
    y = cons[:k]
    rate_zero = np.dot(t, y) / np.dot(t, t)
    rate_free, intercept_free = np.polyfit(t, y, 1)

    return rate_zero, rate_free

def estimated_consumption_time(time, upper_bound, cons, r):
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
        conc = conc_df.loc[row_id].values.astype(float)
        cons = cons_df.loc[row_id].values.astype(float)
        no3_conc0 = no3_conc_df.loc[row_id].values.astype(float)[0]
        no2_conc0 = no2_conc_df.loc[row_id].values.astype(float)[0]
        upper_bound = no3_conc0 if analyte == 'no3' else no3_conc0 + no2_conc0

        k = max(end_time_point(upper_bound, cons), 2)
        t_pre, y_pre = time[:k], cons[:k]

        for c_i, qty in enumerate(quantities):
            ax = axes[r_i, c_i]

            if qty == 'auc':
                ax.plot(time, conc, 'o-', color='tab:blue', markersize=4)
                if analyte == 'no2':
                    no3_conc_curve = no3_conc_df.loc[row_id].values.astype(float)
                    ax.plot(time, no3_conc_curve, 'o-', color='tab:blue', alpha=0.4, markersize=3)
                    ax.fill_between(time, conc, no3_conc_curve, color='tab:blue', alpha=0.25)
                    auc = np.trapezoid(conc, time) - np.trapezoid(no3_conc_curve, time)
                else:
                    ax.fill_between(time, conc, 0, color='tab:blue', alpha=0.25)
                    auc = np.trapezoid(conc, time)
                ax.set_ylabel(f'{analyte} conc (mM)')
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
                r_target = 0.5 if qty == 'half_cons_time' else 0.95
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

    fig.suptitle(f'{id} / {analyte}: how each quantity is computed', fontsize=15, y=1.02)
    fig.tight_layout()
    fn = f'plots/{id}_{analyte}_no_drug_examples_{datetime.datetime.now().strftime("%Y%m%d")}.png'
    fig.savefig(fn, dpi=200, bbox_inches='tight')
    print(f'Figure saved to {fn}')
    plt.close(fig)
    return fn


rows = ['E04', 'F04', 'G04', 'H04']
for id in ids:
    plot_illustration(id, 'no3', rows)
    plot_illustration(id, 'no2', rows)

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

        rate_zero_no3, rate_free_no3 = pre_plateau_rates(time, no3_conc[0], no3_cons)
        rate_zero_no2, rate_free_no2 = pre_plateau_rates(time, no2_conc[0] + no3_conc[0], no2_cons)
        data_dict[id]['rate_zero_intercept'].loc[index, 'no3'] = rate_zero_no3
        data_dict[id]['rate_free_intercept'].loc[index, 'no3'] = rate_free_no3
        data_dict[id]['rate_zero_intercept'].loc[index, 'no2'] = rate_zero_no2
        data_dict[id]['rate_free_intercept'].loc[index, 'no2'] = rate_free_no2

        data_dict[id]['auc'].loc[index, 'no3'] = np.trapezoid(no3_conc, time)
        data_dict[id]['auc'].loc[index, 'no2'] = np.trapezoid(no2_conc, time) - np.trapezoid(no3_conc, time)

        data_dict[id]['half_cons_time'].loc[index, 'no3'] = estimated_consumption_time(time, no3_conc[0], no3_cons, 0.5)
        data_dict[id]['half_cons_time'].loc[index, 'no2'] = estimated_consumption_time(time, no2_conc[0] + no3_conc[0], no2_cons, 0.5)
        data_dict[id]['full_cons_time'].loc[index, 'no3'] = estimated_consumption_time(time, no3_conc[0], no3_cons, 0.95)
        data_dict[id]['full_cons_time'].loc[index, 'no2'] = estimated_consumption_time(time, no2_conc[0] + no3_conc[0], no2_cons, 0.95)
