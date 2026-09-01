from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D

def depletion_time_point(conc, tol=0.02):
    n = len(conc)
    for i in range(n):
        if conc[i] < tol:
            return i
    return n

def full_consumption_time(time, cons, k, total):
    """Time at which `cons` reaches its plateau (i.e. the analyte is fully
    consumed), estimated by linearly extrapolating the line through the two
    timepoints immediately before index `k` (the depletion time point), out
    to `cons[k]`."""
    if k < 2:
        return time[k]
    t0, y0 = time[k - 2], cons[k - 2]
    t1, y1 = time[k - 1], cons[k - 1]
    target = cons[k] if k < len(cons) else total
    return t0 + (target - y0) / (y1 - y0) * (t1 - t0)

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


def pre_plateau_rates(time, cons, k):
    """(rate_zero_intercept, rate_free_intercept) of `cons` vs `time`, fit
    using only the timepoints before index `k` (the depletion time point).
    NaN pair if `k` is the very first point (k=0), since there's no
    pre-plateau data at all. If it's the second point (k=1), there's only
    one pre-plateau point, so the first two timepoints are used instead."""
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
    for key in ['depletion_time_point', 'auc', 'half_life', 'full_consumption', 'rate_zero_intercept', 'rate_free_intercept']:
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

        no3_k = depletion_time_point(no3_conc)
        no2_k = depletion_time_point(no2_conc)
        data_dict[id]['depletion_time_point'].loc[index, 'no3'] = no3_k
        data_dict[id]['depletion_time_point'].loc[index, 'no2'] = no2_k

        data_dict[id]['auc'].loc[index, 'no3'] = np.trapezoid(no3_conc, time)
        data_dict[id]['auc'].loc[index, 'no2'] = np.trapezoid(no2_conc, time)

        no3_half = no3_conc[0] / 2
        no2_half = (no3_conc[0] + no2_conc[0]) / 2
        data_dict[id]['half_life'].loc[index, 'no3'] = exceeds_time(time, no3_cons, no3_half)
        data_dict[id]['half_life'].loc[index, 'no2'] = exceeds_time(time, no2_cons, no2_half)

        data_dict[id]['full_consumption'].loc[index, 'no3'] = full_consumption_time(time, no3_cons, no3_k, no3_conc[0])
        data_dict[id]['full_consumption'].loc[index, 'no2'] = full_consumption_time(time, no2_cons, no2_k, no3_conc[0] + no2_conc[0])

        rate0, rate_free = pre_plateau_rates(time, no3_cons, no3_k)
        data_dict[id]['rate_zero_intercept'].loc[index, 'no3'] = rate0
        data_dict[id]['rate_free_intercept'].loc[index, 'no3'] = rate_free
        rate0, rate_free = pre_plateau_rates(time, no2_cons, no2_k)
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
        no3_conc = no3_conc_chl1.loc[index].values.astype(float)
        no2_conc = no2_conc_chl1.loc[index].values.astype(float)
        no3_cons = no3_cons_chl1.loc[index].values.astype(float)
        no2_cons = no2_cons_chl1.loc[index].values.astype(float)
        
        no3_k = depletion_time_point(no3_conc)
        no2_k = depletion_time_point(no2_conc)
        data_dict[id]['depletion_time_point'].loc[index, 'no3'] = no3_k
        data_dict[id]['depletion_time_point'].loc[index, 'no2'] = no2_k

        rate0, rate_free = pre_plateau_rates(time, no3_cons, no3_k)
        data_dict[id]['rate_zero_intercept'].loc[index, 'no3'] = rate0
        data_dict[id]['rate_free_intercept'].loc[index, 'no3'] = rate_free

        rate0, rate_free = pre_plateau_rates(time, no2_cons, no2_k)
        data_dict[id]['rate_zero_intercept'].loc[index, 'no2'] = rate0
        data_dict[id]['rate_free_intercept'].loc[index, 'no2'] = rate_free


OUTLIER_IQR_FACTOR = 2.0
OUTLIER_COMPRESSION_THRESHOLD = 1.8


def split_outliers(ys):
    """Boolean mask flagging points far enough from the bulk of `ys` (by a
    generous IQR rule) that showing them on the main axes would compress
    everything else into an unreadable clump. Returns an all-False mask if
    there's too little data, no spread, or the flagged points don't actually
    compress the range by more than `OUTLIER_COMPRESSION_THRESHOLD`x."""
    ys = np.asarray(ys, dtype=float)
    finite = np.isfinite(ys)
    outlier = np.zeros(len(ys), dtype=bool)
    if finite.sum() < 4:
        return outlier
    q1, q3 = np.percentile(ys[finite], [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return outlier
    lo_bound, hi_bound = q1 - OUTLIER_IQR_FACTOR * iqr, q3 + OUTLIER_IQR_FACTOR * iqr
    outlier = finite & ((ys < lo_bound) | (ys > hi_bound))
    if not outlier.any():
        return outlier
    inlier_range = ys[finite & ~outlier].max() - ys[finite & ~outlier].min()
    full_range = ys[finite].max() - ys[finite].min()
    if inlier_range <= 0 or full_range / inlier_range <= OUTLIER_COMPRESSION_THRESHOLD:
        return np.zeros(len(ys), dtype=bool)
    return outlier


def plot_chl0_strip(quantity_key, analyte, quantity_label, ylabel):
    add_col = 'Nitrate_input' if analyte == 'no3' else 'Nitrite_input'
    add_symbol = 'A' if analyte == 'no3' else 'I'
    analyte_label = 'Nitrate' if analyte == 'no3' else 'Nitrite'

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    points = []
    medians = []
    for i, id in enumerate(ids):
        df = data_dict[id][quantity_key].loc[mask_chl0, analyte].astype(float)
        for index in df.index:
            y = df.loc[index]
            add_val = data_dict[id]['metadata'].loc[index, add_col]
            marker = markers[add_conc.index(add_val)]
            x = i + 0.1 * (add_conc.index(add_val) - 1.0)
            points.append((x, y, batch_colors[i], marker))
        medians.append((i, df.median()))

    ys = np.array([p[1] for p in points], dtype=float)
    outlier = split_outliers(ys)

    for x, y, color, marker in points:
        ax.scatter(x, y, color=color, marker=marker, alpha=0.5)
    for i, y_med in medians:
        ax.scatter(i, y_med, color='white', edgecolors='black', marker='D', s=100, alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)

    if outlier.any():
        # zoom the main axes to the reliable data and annotate off-scale
        # points in place (their value + a small edge marker) rather than
        # drawing them in a separate inset
        inlier_ys = ys[np.isfinite(ys) & ~outlier]
        pad = 0.15 * (inlier_ys.max() - inlier_ys.min())
        y0, y1 = inlier_ys.min() - pad, inlier_ys.max() + pad
        ax.set_ylim(y0, y1)

        edge = 0.03 * (y1 - y0)
        for (x, y, color, marker), is_out in zip(points, outlier):
            if not is_out:
                continue
            above = y > y1
            yy = y1 - edge if above else y0 + edge
            ax.scatter(x, yy, color=color, marker=marker, s=70, zorder=5, linewidths=0.8)
            ax.annotate(f'{y:.3g}', (x, yy), xytext=(0, -8 if above else 8), textcoords='offset points',
                        ha='center', va='top' if above else 'bottom', fontsize=7.5, color=color, fontweight='bold')

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


