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


def pre_plateau_rates(time, upper_bound, cons, tol=0.2):
    "Slopes of the consumption curve before the plateau. [mM/h]"
    if upper_bound < tol:
        return np.nan, np.nan
    etp = end_time_point(upper_bound, cons)
    k = max(etp + 1, 2)

    t = time[:k]
    y = cons[:k]
    rate_zero = np.dot(t, y) / np.dot(t, t)
    rate_free, intercept_free = np.polyfit(t, y, 1)

    return rate_zero, rate_free

def consumption_time(time, upper_bound, cons, target, tol=0.2):
    "Estimated time when the consumption reaches target. [hours]"
    if upper_bound < tol:
        return np.nan
    if upper_bound < target:
        return np.nan
    
    for k in range(1, len(cons)):
        if cons[k - 1] < target < cons[k]:
            slope = (cons[k] - cons[k - 1]) / (time[k] - time[k - 1])
            return time[k - 1] + (target - cons[k - 1]) / slope
    slope = (cons[-1] - cons[-2]) / (time[-1] - time[-2])
    return time[-1] + (target - cons[-1]) / slope

def norm_auc(time, upper_bound, cons, tol=0.2):
    "Area under the consumption curve before the plateau divided by the upper bound. [hours]"
    if upper_bound < tol:
        return np.nan
    etp = end_time_point(upper_bound, cons)
    auc = np.trapz(cons[:etp], time[:etp])
    return auc / upper_bound


def plot_illustration(id, analyte, rows, target):
    "Illustration figure of rate_zero, rate_free, consumption_time, norm_auc for a few wells."
    no3_conc_df = data_dict[id]['no3_conc']
    no2_conc_df = data_dict[id]['no2_conc']
    cons_df = data_dict[id][analyte]
    metadata = data_dict[id]['metadata']
    time = cons_df.columns.values.astype(float)

    quantities = ['rate_zero', 'rate_free', 'consumption_time', 'norm_auc']
    titles = {
        'rate_zero': 'rate_zero',
        'rate_free': 'rate_free',
        'consumption_time': 'consumption_time',
        'norm_auc': 'norm_auc',
    }

    fig, axes = plt.subplots(len(rows), len(quantities), figsize=(4 * len(quantities), 3.2 * len(rows)))

    for r_i, row_id in enumerate(rows):
        cons = cons_df.loc[row_id].values.astype(float)
        no3_conc0 = no3_conc_df.loc[row_id].values.astype(float)[0]
        no2_conc0 = no2_conc_df.loc[row_id].values.astype(float)[0]
        upper_bound = no3_conc0 if analyte == 'no3_cons' else no3_conc0 + no2_conc0

        etp = end_time_point(upper_bound, cons)
        k = max(etp + 1, 2)
        t_pre, y_pre = time[:k], cons[:k]

        a_add = metadata.loc[row_id, 'Nitrate_input']
        i_add = metadata.loc[row_id, 'Nitrite_input']
        row_label = f'$A_{{add}}$={a_add}\n$I_{{add}}$={i_add}'

        for c_i, qty in enumerate(quantities):
            ax = axes[r_i, c_i]

            if qty in ('rate_zero', 'rate_free'):
                rate_zero, rate_free = pre_plateau_rates(time, upper_bound, cons)
                ax.plot(time, cons, 'o', color='gray', alpha=0.4, markersize=4)
                ax.plot(t_pre, y_pre, 'o', color='tab:red', markersize=6, zorder=3)
                t_line = np.array([0, t_pre[-1]])
                if qty == 'rate_zero':
                    if not np.isnan(rate_zero):
                        ax.plot(t_line, rate_zero * t_line, '--', color='tab:red')
                        label = f'rate_zero = {rate_zero:.4f} mM/h\n(forced through 0)'
                    else:
                        label = 'undefined'
                else:
                    if not np.isnan(rate_free):
                        slope, intercept = np.polyfit(t_pre, y_pre, 1)
                        ax.plot(t_line, slope * t_line + intercept, '--', color='tab:red')
                        label = f'rate_free = {rate_free:.4f} mM/h\nintercept = {intercept:.3f}'
                    else:
                        label = 'undefined'
                ax.text(0.97, 0.05, label, transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=11, color='tab:red')

            elif qty == 'consumption_time':
                t_target = consumption_time(time, upper_bound, cons, target)
                ax.plot(time, cons, 'o-', color='tab:green', markersize=4)
                ax.axhline(target, color='tab:orange', linestyle=':', linewidth=1)
                if not np.isnan(t_target):
                    ax.axvline(t_target, color='tab:orange', linestyle=':', linewidth=1)
                    ax.plot(t_target, target, 'D', color='tab:orange', markersize=8, zorder=3)
                    label = f't = {t_target:.2f} h\n(target = {target} mM)'
                else:
                    label = f'undefined\n(target = {target} mM)'
                ax.text(0.97, 0.05, label, transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=11, color='tab:orange')

            elif qty == 'norm_auc':
                ax.plot(time, cons, 'o', color='gray', alpha=0.4, markersize=4)
                ax.plot(t_pre, y_pre, 'o-', color='tab:blue', markersize=4, zorder=3)
                ax.fill_between(t_pre, y_pre, 0, color='tab:blue', alpha=0.25)
                n_auc = norm_auc(time, upper_bound, cons)
                ax.text(0.97, 0.95, f'norm_auc = {n_auc:.3f} h', transform=ax.transAxes,
                        ha='right', va='top', fontsize=11, color='tab:blue')

            ax.tick_params(axis='both', labelsize=12)
            ax.set_xlabel('time (h)', fontsize=14)
            if c_i == 0:
                ax.annotate(row_label, xy=(-0.4, 0.5), xycoords='axes fraction', fontsize=16,
                            fontweight='bold', ha='center', va='center', rotation=90)
            if r_i == len(rows) - 1:
                ax.set_ylabel(f'{analyte} (mM)', fontsize=14)
            if r_i == 0:
                ax.set_title(titles[qty], fontsize=16)

    analyte_label, symbol = ('Nitrate', 'A') if analyte == 'no3_cons' else ('Nitrite', 'I')
    batch_label = batch_labels[ids.index(id)]
    fig.suptitle(f'{analyte_label} ({symbol}) CHL- analysis, {batch_label}', fontsize=20, y=1.02)
    fig.tight_layout()
    fn = f'plots/{id}_{analyte}_quantity_illustration_{datetime.datetime.now().strftime("%Y%m%d")}.png'
    fig.savefig(fn, dpi=200, bbox_inches='tight')
    print(f'Figure saved to {fn}')
    plt.close(fig)
    return fn


illustration_rows = ['E04', 'F10', 'G04', 'H04']
illustration_targets = {'no3_cons': 0.7, 'no2_cons': 1.0}
for id in ids:
    for analyte, target in illustration_targets.items():
        plot_illustration(id, analyte, illustration_rows, target)
