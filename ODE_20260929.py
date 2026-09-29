import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
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

water_contents = [98.9, 62.5, 34.4, 7.44, 7.10, 5.16, 4.74]


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


# ---------------------------------------------------------------------------
# ODE model
#   dA/dt = -r_A * X_A * A / (A + K_A)
#   dI/dt = -dA/dt - r_I * X_I * I / (I + K_I)
#   dX_A/dt = -delta * X_A
#   dX_I/dt = -delta * X_I
#   X_A(0) = X_I(0) = 1
# ---------------------------------------------------------------------------

def odes(t, y, r_A, r_I, K_A, K_I, delta):
    A, I, X_A, X_I = y
    dAdt = -r_A * X_A * A / (A + K_A)
    dIdt = -dAdt - r_I * X_I * I / (I + K_I)
    dX_Adt = -delta * X_A
    dX_Idt = -delta * X_I
    return [dAdt, dIdt, dX_Adt, dX_Idt]


def solve_model(t_eval, A0, I0, r_A, r_I, K_A, K_I, delta):
    t_span = (t_eval[0], t_eval[-1])
    sol = solve_ivp(odes, t_span, [A0, I0, 1.0, 1.0], t_eval=t_eval, args=(r_A, r_I, K_A, K_I, delta),
                     method='LSODA', rtol=1e-8, atol=1e-10)
    return sol.y  # shape (4, len(t_eval)): rows are A, I, X_A, X_I


def get_condition_data(id, A_add, I_add, chl=1):
    """Return (time, A_reps (n_rep, n_t), I_reps (n_rep, n_t)) for non-blank wells
    matching Chloramphenicol == chl, Nitrate_input == A_add and Nitrite_input == I_add."""
    meta = data_dict[id]['metadata']
    mask = ((meta['Chloramphenicol'] == chl) & (meta['Sample_type'] != 'Blank') &
            (meta['Nitrate_input'] == A_add) & (meta['Nitrite_input'] == I_add))
    wells = meta.index[mask]
    if len(wells) == 0:
        return None, None, None
    time = data_dict[id]['no3_conc'].columns.values.astype(float)
    A_reps = data_dict[id]['no3_conc'].loc[wells].values
    I_reps = data_dict[id]['no2_conc'].loc[wells].values
    return time, A_reps, I_reps


def fit_condition(time, A_reps, I_reps, free_K=False):
    A0 = A_reps[:, 0].mean()
    I0 = I_reps[:, 0].mean()

    K_fixed = 1e-3

    def unpack(x):
        r_A, r_I, delta = x[0], x[1], x[2]
        K_A, K_I = (x[3], x[4]) if free_K else (K_fixed, K_fixed)
        return r_A, r_I, K_A, K_I, delta

    def residuals(x):
        r_A, r_I, K_A, K_I, delta = unpack(x)
        y = solve_model(time, A0, I0, r_A, r_I, K_A, K_I, delta)
        if y.shape[1] != len(time):
            return np.full(A_reps.size + I_reps.size, 1e3)
        res_A = (y[0][None, :] - A_reps).ravel()
        res_I = (y[1][None, :] - I_reps).ravel()
        return np.concatenate([res_A, res_I])

    x0 = [0.05, 0.03, 1 / 30] + ([K_fixed, K_fixed] if free_K else [])
    lb = [1e-6, 1e-6, 1e-4] + ([1e-6, 1e-6] if free_K else [])
    ub = [10.0, 10.0, 100.0] + ([10.0, 10.0] if free_K else [])

    result = least_squares(residuals, x0, bounds=(lb, ub))
    r_A, r_I, K_A, K_I, delta = unpack(result.x)
    rmse = np.sqrt(np.mean(result.fun ** 2))
    return {'r_A': r_A, 'r_I': r_I, 'K_A': K_A, 'K_I': K_I, 'delta': delta, 'A0': A0, 'I0': I0, 'rmse': rmse}


def fit_batch_global(id, free_K=False):
    """Fit one shared set of parameters (r_A, r_I, delta[, K_A, K_I]) to all 45 chl1
    replicates in a batch at once; only A0/I0 vary per condition."""
    conditions = []
    for I_add in add_conc:
        for A_add in add_conc:
            time, A_reps, I_reps = get_condition_data(id, A_add, I_add)
            if time is None:
                continue
            conditions.append({
                'A_add': A_add, 'I_add': I_add, 'time': time,
                'A_reps': A_reps, 'I_reps': I_reps,
                'A0': A_reps[:, 0].mean(), 'I0': I_reps[:, 0].mean(),
            })

    K_fixed = 1e-3

    def unpack(x):
        r_A, r_I, delta = x[0], x[1], x[2]
        K_A, K_I = (x[3], x[4]) if free_K else (K_fixed, K_fixed)
        return r_A, r_I, K_A, K_I, delta

    def residuals(x):
        r_A, r_I, K_A, K_I, delta = unpack(x)
        res = []
        for c in conditions:
            y = solve_model(c['time'], c['A0'], c['I0'], r_A, r_I, K_A, K_I, delta)
            if y.shape[1] != len(c['time']):
                res.append(np.full(c['A_reps'].size + c['I_reps'].size, 1e3))
                continue
            res.append((y[0][None, :] - c['A_reps']).ravel())
            res.append((y[1][None, :] - c['I_reps']).ravel())
        return np.concatenate(res)

    x0 = [0.05, 0.03, 1 / 30] + ([K_fixed, K_fixed] if free_K else [])
    lb = [1e-6, 1e-6, 1e-4] + ([1e-6, 1e-6] if free_K else [])
    ub = [10.0, 10.0, 100.0] + ([10.0, 10.0] if free_K else [])

    result = least_squares(residuals, x0, bounds=(lb, ub))
    r_A, r_I, K_A, K_I, delta = unpack(result.x)
    rmse = np.sqrt(np.mean(result.fun ** 2))
    return {'r_A': r_A, 'r_I': r_I, 'K_A': K_A, 'K_I': K_I, 'delta': delta, 'rmse': rmse, 'conditions': conditions}


def fit_and_plot_batch_global(id, free_K=False, plot_fn=None):
    fit = fit_batch_global(id, free_K=free_K)
    cond_lookup = {(c['A_add'], c['I_add']): c for c in fit['conditions']}

    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)

    for row, I_add in enumerate(add_conc):
        for col, A_add in enumerate(add_conc):
            ax = axes[row, col]
            c = cond_lookup.get((A_add, I_add))

            if c is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                continue

            y_at_data = solve_model(c['time'], c['A0'], c['I0'], fit['r_A'], fit['r_I'], fit['K_A'], fit['K_I'], fit['delta'])
            n_rep = c['A_reps'].shape[0]
            rep_rmse = [np.sqrt(np.mean(np.concatenate([
                y_at_data[0] - c['A_reps'][i], y_at_data[1] - c['I_reps'][i]]) ** 2)) for i in range(n_rep)]
            worst = int(np.argmax(rep_rmse))

            ax.plot(c['time'], c['A_reps'][worst], 'o', color='blue', ms=4)
            ax.plot(c['time'], c['I_reps'][worst], 'o', color='red', ms=4)

            t_plot = np.linspace(c['time'][0], c['time'][-1], 200)
            y_plot = solve_model(t_plot, c['A0'], c['I0'], fit['r_A'], fit['r_I'], fit['K_A'], fit['K_I'], fit['delta'])
            ax.plot(t_plot, y_plot[0], color='blue')
            ax.plot(t_plot, y_plot[1], color='red')

    for col, A_add in enumerate(add_conc):
        axes[0, col].annotate(f'A_add = {A_add}', xy=(0.5, 1.28), xycoords='axes fraction',
                               ha='center', fontsize=11, fontweight='bold')
    for row, I_add in enumerate(add_conc):
        axes[row, 0].annotate(f'I_add = {I_add}', xy=(-0.35, 0.5), xycoords='axes fraction',
                               ha='center', va='center', rotation=90, fontsize=11, fontweight='bold')

    for row in range(4):
        axes[row, 0].set_ylabel('Concentration (mM)')
    for col in range(4):
        axes[3, col].set_xlabel('Time (days)')

    handles = [Line2D([0], [0], color='blue', marker='o', linestyle='-', label='$NO_3^-$ (A)'),
               Line2D([0], [0], color='red', marker='o', linestyle='-', label='$NO_2^-$ (I)')]
    fig.legend(handles=handles, loc='upper right')

    batch_num = id.split('batch')[-1]
    if free_K:
        fixed_line = 'Fixed: X_A(0) = X_I(0) = 1.0'
        inferred_line = (f'Inferred: r_A={fit["r_A"]:.4f}, r_I={fit["r_I"]:.4f}, delta={fit["delta"]:.4f}, '
                          f'K_A={fit["K_A"]:.3g}, K_I={fit["K_I"]:.3g}, RMSE={fit["rmse"]:.4f}')
    else:
        fixed_line = 'Fixed: K_A = K_I = 0.001, X_A(0) = X_I(0) = 1.0'
        inferred_line = f'Inferred: r_A={fit["r_A"]:.4f}, r_I={fit["r_I"]:.4f}, delta={fit["delta"]:.4f}, RMSE={fit["rmse"]:.4f}'
    eq_line = (r"$\dot{A}=-r_A X_A \frac{A}{A+K_A}$,  $\dot{I}=-\dot{A}-r_I X_I \frac{I}{I+K_I}$,  "
               r"$\dot{X}_A=-\delta X_A$,  $\dot{X}_I=-\delta X_I$")
    fig.suptitle(f'Batch {batch_num} CHL+ Global Fit (worst replicate for each condition)\n{fixed_line}\n{inferred_line}\n{eq_line}', fontsize=14)
    fig.tight_layout(rect=[0.02, 0.02, 0.95, 0.90])

    if plot_fn:
        fig.savefig(plot_fn, dpi=200)
        print(f'Saved {plot_fn}  (r_A={fit["r_A"]:.4f}, r_I={fit["r_I"]:.4f}, delta={fit["delta"]:.4f}, RMSE={fit["rmse"]:.4f})')
    plt.close(fig)
    return fit


# ---------------------------------------------------------------------------
# CHL- (chl0) ODE model: X grows on consumption and decays with the same delta
#   dA/dt = -r_A * X_A * A / (A + K_A)
#   dI/dt = -dA/dt - r_I * X_I * I / (I + K_I)
#   dX_A/dt = Gamma_A * X_A * A / (A + K_A) - delta * X_A
#   dX_I/dt = Gamma_I * X_I * I / (I + K_I) - delta * X_I
#   X_A(0) = X_I(0) = 1
# ---------------------------------------------------------------------------

def odes_chl0(t, y, r_A, r_I, K_A, K_I, delta, Gamma_A, Gamma_I):
    A, I, X_A, X_I = y
    monod_A = A / (A + K_A)
    monod_I = I / (I + K_I)
    dAdt = -r_A * X_A * monod_A
    dIdt = -dAdt - r_I * X_I * monod_I
    dX_Adt = Gamma_A * X_A * monod_A - delta * X_A
    dX_Idt = Gamma_I * X_I * monod_I - delta * X_I
    return [dAdt, dIdt, dX_Adt, dX_Idt]


def solve_model_chl0(t_eval, A0, I0, r_A, r_I, K_A, K_I, delta, Gamma_A, Gamma_I):
    t_span = (t_eval[0], t_eval[-1])
    sol = solve_ivp(odes_chl0, t_span, [A0, I0, 1.0, 1.0], t_eval=t_eval,
                     args=(r_A, r_I, K_A, K_I, delta, Gamma_A, Gamma_I),
                     method='LSODA', rtol=1e-8, atol=1e-10)
    return sol.y  # shape (4, len(t_eval)): rows are A, I, X_A, X_I


def fit_batch_global_chl0(id, free_K=False):
    """Fit one shared set of parameters (r_A, r_I, delta, Gamma_A, Gamma_I[, K_A, K_I])
    to all 45 chl0 replicates in a batch at once; only A0/I0 vary per condition."""
    conditions = []
    for I_add in add_conc:
        for A_add in add_conc:
            time, A_reps, I_reps = get_condition_data(id, A_add, I_add, chl=0)
            if time is None:
                continue
            conditions.append({
                'A_add': A_add, 'I_add': I_add, 'time': time,
                'A_reps': A_reps, 'I_reps': I_reps,
                'A0': A_reps[:, 0].mean(), 'I0': I_reps[:, 0].mean(),
            })

    K_fixed = 1e-3

    def unpack(x):
        r_A, r_I, delta, Gamma_A, Gamma_I = x[0], x[1], x[2], x[3], x[4]
        K_A, K_I = (x[5], x[6]) if free_K else (K_fixed, K_fixed)
        return r_A, r_I, K_A, K_I, delta, Gamma_A, Gamma_I

    def residuals(x):
        r_A, r_I, K_A, K_I, delta, Gamma_A, Gamma_I = unpack(x)
        res = []
        for c in conditions:
            y = solve_model_chl0(c['time'], c['A0'], c['I0'], r_A, r_I, K_A, K_I, delta, Gamma_A, Gamma_I)
            if y.shape[1] != len(c['time']):
                res.append(np.full(c['A_reps'].size + c['I_reps'].size, 1e3))
                continue
            res.append((y[0][None, :] - c['A_reps']).ravel())
            res.append((y[1][None, :] - c['I_reps']).ravel())
        return np.concatenate(res)

    x0 = [0.05, 0.03, 1 / 30, 0.05, 0.05] + ([K_fixed, K_fixed] if free_K else [])
    lb = [1e-6, 1e-6, 1e-4, 1e-6, 1e-6] + ([1e-6, 1e-6] if free_K else [])
    ub = [10.0, 10.0, 100.0, 10.0, 10.0] + ([10.0, 10.0] if free_K else [])

    result = least_squares(residuals, x0, bounds=(lb, ub))
    r_A, r_I, K_A, K_I, delta, Gamma_A, Gamma_I = unpack(result.x)
    rmse = np.sqrt(np.mean(result.fun ** 2))
    return {'r_A': r_A, 'r_I': r_I, 'K_A': K_A, 'K_I': K_I, 'delta': delta,
            'Gamma_A': Gamma_A, 'Gamma_I': Gamma_I, 'rmse': rmse, 'conditions': conditions}


def fit_and_plot_batch_global_chl0(id, free_K=False, plot_fn=None):
    fit = fit_batch_global_chl0(id, free_K=free_K)
    cond_lookup = {(c['A_add'], c['I_add']): c for c in fit['conditions']}

    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)

    for row, I_add in enumerate(add_conc):
        for col, A_add in enumerate(add_conc):
            ax = axes[row, col]
            c = cond_lookup.get((A_add, I_add))

            if c is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                continue

            y_at_data = solve_model_chl0(c['time'], c['A0'], c['I0'], fit['r_A'], fit['r_I'],
                                          fit['K_A'], fit['K_I'], fit['delta'], fit['Gamma_A'], fit['Gamma_I'])
            n_rep = c['A_reps'].shape[0]
            rep_rmse = [np.sqrt(np.mean(np.concatenate([
                y_at_data[0] - c['A_reps'][i], y_at_data[1] - c['I_reps'][i]]) ** 2)) for i in range(n_rep)]
            worst = int(np.argmax(rep_rmse))

            ax.plot(c['time'], c['A_reps'][worst], 'o', color='blue', ms=4)
            ax.plot(c['time'], c['I_reps'][worst], 'o', color='red', ms=4)

            t_plot = np.linspace(c['time'][0], c['time'][-1], 200)
            y_plot = solve_model_chl0(t_plot, c['A0'], c['I0'], fit['r_A'], fit['r_I'],
                                       fit['K_A'], fit['K_I'], fit['delta'], fit['Gamma_A'], fit['Gamma_I'])
            ax.plot(t_plot, y_plot[0], color='blue')
            ax.plot(t_plot, y_plot[1], color='red')

    for col, A_add in enumerate(add_conc):
        axes[0, col].annotate(f'A_add = {A_add}', xy=(0.5, 1.28), xycoords='axes fraction',
                               ha='center', fontsize=11, fontweight='bold')
    for row, I_add in enumerate(add_conc):
        axes[row, 0].annotate(f'I_add = {I_add}', xy=(-0.35, 0.5), xycoords='axes fraction',
                               ha='center', va='center', rotation=90, fontsize=11, fontweight='bold')

    for row in range(4):
        axes[row, 0].set_ylabel('Concentration (mM)')
    for col in range(4):
        axes[3, col].set_xlabel('Time (days)')

    handles = [Line2D([0], [0], color='blue', marker='o', linestyle='-', label='$NO_3^-$ (A)'),
               Line2D([0], [0], color='red', marker='o', linestyle='-', label='$NO_2^-$ (I)')]
    fig.legend(handles=handles, loc='upper right')

    batch_num = id.split('batch')[-1]
    if free_K:
        fixed_line = 'Fixed: X_A(0) = X_I(0) = 1.0'
        inferred_line = (f'Inferred: r_A={fit["r_A"]:.4f}, r_I={fit["r_I"]:.4f}, delta={fit["delta"]:.4f}, '
                          f'Gamma_A={fit["Gamma_A"]:.4f}, Gamma_I={fit["Gamma_I"]:.4f}, '
                          f'K_A={fit["K_A"]:.3g}, K_I={fit["K_I"]:.3g}, RMSE={fit["rmse"]:.4f}')
    else:
        fixed_line = 'Fixed: K_A = K_I = 0.001, X_A(0) = X_I(0) = 1.0'
        inferred_line = (f'Inferred: r_A={fit["r_A"]:.4f}, r_I={fit["r_I"]:.4f}, delta={fit["delta"]:.4f}, '
                          f'Gamma_A={fit["Gamma_A"]:.4f}, Gamma_I={fit["Gamma_I"]:.4f}, RMSE={fit["rmse"]:.4f}')
    eq_line = (r"$\dot{A}=-r_A X_A \frac{A}{A+K_A}$,  $\dot{I}=-\dot{A}-r_I X_I \frac{I}{I+K_I}$,  "
               r"$\dot{X}_A=\Gamma_A X_A \frac{A}{A+K_A}-\delta X_A$,  $\dot{X}_I=\Gamma_I X_I \frac{I}{I+K_I}-\delta X_I$")
    fig.suptitle(f'Batch {batch_num} CHL- Global Fit (worst replicate for each condition)\n{fixed_line}\n{inferred_line}\n{eq_line}', fontsize=14)
    fig.tight_layout(rect=[0.02, 0.02, 0.95, 0.90])

    if plot_fn:
        fig.savefig(plot_fn, dpi=200)
        print(f'Saved {plot_fn}  (r_A={fit["r_A"]:.4f}, r_I={fit["r_I"]:.4f}, delta={fit["delta"]:.4f}, '
              f'Gamma_A={fit["Gamma_A"]:.4f}, Gamma_I={fit["Gamma_I"]:.4f}, RMSE={fit["rmse"]:.4f})')
    plt.close(fig)
    return fit


def fit_and_plot_batch(id, free_K=False, plot_fn=None):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)

    rmses = []
    for row, I_add in enumerate(add_conc):
        for col, A_add in enumerate(add_conc):
            ax = axes[row, col]
            time, A_reps, I_reps = get_condition_data(id, A_add, I_add)

            if time is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'A_add={A_add}, I_add={I_add}')
                continue

            fit = fit_condition(time, A_reps, I_reps, free_K=free_K)
            rmses.append(fit['rmse'])

            A_mean, A_std = A_reps.mean(axis=0), A_reps.std(axis=0)
            I_mean, I_std = I_reps.mean(axis=0), I_reps.std(axis=0)
            ax.errorbar(time, A_mean, yerr=A_std, fmt='o', color='blue', ms=4, capsize=2, linestyle='none')
            ax.errorbar(time, I_mean, yerr=I_std, fmt='o', color='red', ms=4, capsize=2, linestyle='none')

            t_plot = np.linspace(time[0], time[-1], 200)
            y_plot = solve_model(t_plot, fit['A0'], fit['I0'], fit['r_A'], fit['r_I'], fit['K_A'], fit['K_I'], fit['delta'])
            ax.plot(t_plot, y_plot[0], color='blue')
            ax.plot(t_plot, y_plot[1], color='red')

            title = f'A_add={A_add}, I_add={I_add}\nr_A={fit["r_A"]:.3f}, r_I={fit["r_I"]:.3f}, delta={fit["delta"]:.3f}'
            if free_K:
                title += f'\nK_A={fit["K_A"]:.3g}, K_I={fit["K_I"]:.3g}'
            title += f', RMSE={fit["rmse"]:.3f}'
            ax.set_title(title, fontsize=8)

    for col, A_add in enumerate(add_conc):
        axes[0, col].annotate(f'A_add = {A_add}', xy=(0.5, 1.28), xycoords='axes fraction',
                               ha='center', fontsize=11, fontweight='bold')
    for row, I_add in enumerate(add_conc):
        axes[row, 0].annotate(f'I_add = {I_add}', xy=(-0.35, 0.5), xycoords='axes fraction',
                               ha='center', va='center', rotation=90, fontsize=11, fontweight='bold')

    for row in range(4):
        axes[row, 0].set_ylabel('Concentration (mM)')
    for col in range(4):
        axes[3, col].set_xlabel('Time (days)')

    handles = [Line2D([0], [0], color='blue', marker='o', linestyle='-', label='$NO_3^-$ (A)'),
               Line2D([0], [0], color='red', marker='o', linestyle='-', label='$NO_2^-$ (I)')]
    fig.legend(handles=handles, loc='upper right')

    k_str = 'K_A=K_I=0.001 fixed' if not free_K else 'K_A, K_I free'
    fig.suptitle(f'{id} chl1 ODE fit with X decay ({k_str}), mean RMSE = {np.mean(rmses):.4f}', fontsize=14)
    fig.tight_layout(rect=[0.02, 0.02, 0.95, 0.94])

    if plot_fn:
        fig.savefig(plot_fn, dpi=200)
        print(f'Saved {plot_fn}  (mean RMSE = {np.mean(rmses):.4f})')
    plt.close(fig)
    return np.mean(rmses)


if __name__ == '__main__':
    today = datetime.date.today().strftime('%Y%m%d')
    for id in ids:
        fit_and_plot_batch_global(id, free_K=False, plot_fn=f'plots/{id}.chl1_ODE_global_Kfixed_{today}.png')
    for id in ids:
        fit_and_plot_batch_global_chl0(id, free_K=False, plot_fn=f'plots/{id}.chl0_ODE_global_Kfixed_{today}.png')
    for id in ids:
        fit_and_plot_batch_global_chl0(id, free_K=True, plot_fn=f'plots/{id}.chl0_ODE_global_Kfree_{today}.png')
