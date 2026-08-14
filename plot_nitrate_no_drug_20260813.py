from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D

def exceeds_time(time, cons, threshold):
    """Time at which `cons` first exceeds `threshold`, linearly interpolated
    between the bracketing timepoints. Returns NaN if `cons` never exceeds
    `threshold`."""
    for i in range(len(cons) - 1):
        y0, y1 = cons[i], cons[i + 1]
        if y0 < threshold and y1 >= threshold:
            t0, t1 = time[i], time[i + 1]
            return t0 + (threshold - y0) / (y1 - y0) * (t1 - t0)
    return 0.0


def plateau_start(cons, tol):
    """First index i such that every later value in `cons` stays within
    `tol` of `cons[i]` -- i.e. the earliest point after which nothing but
    noise happens. Falls back to the last index if `cons` never settles."""
    n = len(cons)
    for i in range(n):
        if np.max(np.abs(cons[i:] - cons[i])) <= tol:
            return i
    return n - 1


PLATEAU_TOL = 0.02


def full_consumption_time(time, cons, tol=PLATEAU_TOL):
    """Time at which `cons` reaches its plateau (i.e. the analyte is fully
    consumed), estimated by linearly extrapolating the line through the two
    timepoints immediately before the plateau starts, out to the plateau
    value. `tol` is the absolute (mM) tolerance used to decide a point is
    "part of the plateau" -- see `plateau_start`. Returns NaN if there aren't
    two timepoints before the plateau starts."""
    k = plateau_start(cons, tol)
    if k < 2:
        return time[k]
    t0, y0 = time[k - 2], cons[k - 2]
    t1, y1 = time[k - 1], cons[k - 1]
    target = cons[k]
    return t0 + (target - y0) / (y1 - y0) * (t1 - t0)


def rate_zero_intercept(t, y):
    """Least-squares slope of y vs t with the line forced through the origin.
    A negative slope isn't physically meaningful for a consumption rate --
    it means the fit window was noise, not real consumption -- so NaN."""
    rate = np.sum(t * y) / np.sum(t ** 2)
    return rate if rate >= 0 else np.nan


def rate_free_intercept(t, y):
    """Least-squares slope of y vs t with the intercept also inferred. A
    negative slope isn't physically meaningful for a consumption rate, so
    NaN (see `rate_zero_intercept`)."""
    slope, intercept, *_ = stats.linregress(t, y)
    return slope if slope >= 0 else np.nan


def pre_plateau_rates(time, cons, tol=PLATEAU_TOL):
    """(rate_zero_intercept, rate_free_intercept) of `cons` vs `time`, fit
    using only the timepoints before the plateau starts. NaN pair if the
    plateau starts at the very first point (k=0), since there's no
    pre-plateau data at all. If it starts at the second point (k=1), there's
    only one pre-plateau point, so the first two timepoints are used
    instead."""
    k = plateau_start(cons, tol)
    if k == 0:
        return np.nan, np.nan
    t, y = time[:max(k, 2)], cons[:max(k, 2)]
    return rate_zero_intercept(t, y), rate_free_intercept(t, y)


def auc_upto(df, threshold):
    """Trapezoidal AUC per row, integrated only up to `threshold` (a time value).
    If `threshold` isn't an existing column, the value there is linearly
    interpolated from the surrounding timepoints before integrating."""
    col_times = df.columns.astype(float)
    before = col_times <= threshold
    after = col_times > threshold
    cols_before = df.columns[before]
    times_before = col_times[before]
    if after.any():
        col_after = df.columns[after][0]
        time_after = col_times[after][0]

    auc = pd.Series(index=df.index, dtype=float)
    for idx in df.index:
        x = list(times_before)
        y = list(df.loc[idx, cols_before].astype(float))
        if after.any() and threshold not in times_before:
            t_before, y_before = x[-1], y[-1]
            y_after = float(df.loc[idx, col_after])
            y_thresh = y_before + (y_after - y_before) * (threshold - t_before) / (time_after - t_before)
            x.append(threshold)
            y.append(y_thresh)
        auc[idx] = np.trapezoid(y, x)
    return auc


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

for id in ids:
    time = data_dict[id]['no3_conc'].columns.values.astype(float)
    data_dict[id]['auc'] = pd.DataFrame(index=data_dict[id]['no3_cons'].index, columns=['no3', 'no2'])
    data_dict[id]['half_life'] = pd.DataFrame(index=data_dict[id]['no3_cons'].index, columns=['no3', 'no2'])
    data_dict[id]['full_consumption'] = pd.DataFrame(index=data_dict[id]['no3_cons'].index, columns=['no3', 'no2'])
    data_dict[id]['rate_zero_intercept'] = pd.DataFrame(index=data_dict[id]['no3_cons'].index, columns=['no3', 'no2'])
    data_dict[id]['rate_free_intercept'] = pd.DataFrame(index=data_dict[id]['no3_cons'].index, columns=['no3', 'no2'])

    no3_conc_chl0 = data_dict[id]['no3_conc'][mask_chl0]
    no2_conc_chl0 = data_dict[id]['no2_conc'][mask_chl0]
    no3_cons_chl0 = data_dict[id]['no3_cons'][mask_chl0]
    no2_cons_chl0 = data_dict[id]['no2_cons'][mask_chl0]
    for index in no3_conc_chl0.index:
        no3_conc = no3_conc_chl0.loc[index].values.astype(float)
        no2_conc = no2_conc_chl0.loc[index].values.astype(float)
        no3_cons = no3_cons_chl0.loc[index].values.astype(float)
        no2_cons = no2_cons_chl0.loc[index].values.astype(float)
        data_dict[id]['auc'].loc[index, 'no3'] = np.trapezoid(no3_conc, time)
        data_dict[id]['auc'].loc[index, 'no2'] = np.trapezoid(no2_conc, time)

        no3_half = no3_conc[0] / 2
        no2_half = (no3_conc[0] + no2_conc[0]) / 2
        data_dict[id]['half_life'].loc[index, 'no3'] = exceeds_time(time, no3_cons, no3_half)
        data_dict[id]['half_life'].loc[index, 'no2'] = exceeds_time(time, no2_cons, no2_half)

        data_dict[id]['full_consumption'].loc[index, 'no3'] = full_consumption_time(time, no3_cons)
        data_dict[id]['full_consumption'].loc[index, 'no2'] = full_consumption_time(time, no2_cons)

        rate0, rate_free = pre_plateau_rates(time, no3_cons)
        data_dict[id]['rate_zero_intercept'].loc[index, 'no3'] = rate0
        data_dict[id]['rate_free_intercept'].loc[index, 'no3'] = rate_free
        rate0, rate_free = pre_plateau_rates(time, no2_cons)
        data_dict[id]['rate_zero_intercept'].loc[index, 'no2'] = rate0
        data_dict[id]['rate_free_intercept'].loc[index, 'no2'] = rate_free

    end_time = data_dict['4.2.batch3']['no3_conc'].columns.values.astype(float)[-1]
    no3_conc_chl1 = data_dict[id]['no3_conc'][mask_chl1]
    no2_conc_chl1 = data_dict[id]['no2_conc'][mask_chl1]
    no3_cons_chl1 = data_dict[id]['no3_cons'][mask_chl1]
    no2_cons_chl1 = data_dict[id]['no2_cons'][mask_chl1]
    data_dict[id]['auc'].loc[no3_conc_chl1.index, 'no3'] = auc_upto(no3_conc_chl1, end_time)
    data_dict[id]['auc'].loc[no2_conc_chl1.index, 'no2'] = auc_upto(no2_conc_chl1, end_time)

    for index in no3_conc_chl1.index:
        no3_cons = no3_cons_chl1.loc[index].values.astype(float)
        no2_cons = no2_cons_chl1.loc[index].values.astype(float)
        rate0, rate_free = pre_plateau_rates(time, no3_cons)
        data_dict[id]['rate_zero_intercept'].loc[index, 'no3'] = rate0
        data_dict[id]['rate_free_intercept'].loc[index, 'no3'] = rate_free
        rate0, rate_free = pre_plateau_rates(time, no2_cons)
        data_dict[id]['rate_zero_intercept'].loc[index, 'no2'] = rate0
        data_dict[id]['rate_free_intercept'].loc[index, 'no2'] = rate_free


def plot_chl0_strip(quantity_key, analyte, quantity_label, ylabel):
    add_col = 'Nitrate_input' if analyte == 'no3' else 'Nitrite_input'
    add_symbol = 'A' if analyte == 'no3' else 'I'
    analyte_label = 'Nitrate' if analyte == 'no3' else 'Nitrite'

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for i, id in enumerate(ids):
        df = data_dict[id][quantity_key].loc[mask_chl0, analyte].astype(float)
        for index in df.index:
            y = df.loc[index]
            add_val = data_dict[id]['metadata'].loc[index, add_col]
            marker = markers[add_conc.index(add_val)]
            ax.scatter(i + 0.1 * (add_conc.index(add_val) - 1.0), y, color=batch_colors[i], marker=marker, alpha=0.5)
        y_med = df.median()
        ax.scatter(i, y_med, color='white', edgecolors='black', marker='D', s=100, alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title(f'{analyte_label} {quantity_label} (CHL-)' + '\n' + rf'colour = batch | marker = ${add_symbol}_{{add}}$ | diamond = batch median', fontsize=14)
    ax.set_xticks(range(6))
    ax.set_xticklabels(batch_labels)
    ax.set_ylabel(ylabel)
    custom_lines = [Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=8, label=f'${add_symbol}_{{add}}$ = {conc} mM') for marker, conc in zip(markers, add_conc)]
    legend = ax.legend(handles=custom_lines)
    ax.add_artist(legend)
    fn = f'4.2.chl0_{analyte}_{quantity_key}'
    plt.savefig(f'plots/{fn}_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


plot_specs = [
    ('auc', 'AUC', 'AUC (mM\N{MIDDLE DOT}h)'),
    ('half_life', 'half-life', 'Time (h)'),
    ('full_consumption', 'full consumption time', 'Time (h)'),
    ('rate_zero_intercept', 'consumption rate (zero intercept)', 'Rate (mM/h)'),
    ('rate_free_intercept', 'consumption rate (free intercept)', 'Rate (mM/h)'),
]

for quantity_key, quantity_label, ylabel in plot_specs:
    for analyte in ['no3', 'no2']:
        plot_chl0_strip(quantity_key, analyte, quantity_label, ylabel)


