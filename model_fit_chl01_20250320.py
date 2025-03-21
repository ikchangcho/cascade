import sys
sys.path.append('./_functions')
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters
from scipy.interpolate import interp1d
import json
import pickle
import copy
from models import *

# Load your data
no3_chl0 = pd.read_csv(f'concentrations/no3_chl0_evap.csv', index_col=0)
no2_chl0 = pd.read_csv(f'concentrations/no2_chl0_evap.csv', index_col=0)
no3_chl1 = pd.read_csv(f'concentrations/no3_chl1_evap.csv', index_col=0)
no2_chl1 = pd.read_csv(f'concentrations/no2_chl1_evap.csv', index_col=0)

for row in ['A01', 'B01']:
    A_chl0 = no3_chl0.loc[row]
    I_chl0 = no2_chl0.loc[row]
    A_chl1 = no3_chl1.loc[row]
    I_chl1 = no2_chl1.loc[row]
    t_eval_chl0 = np.linspace(0, float(no3_chl0.columns[-1]) + 1, 100)
    t_eval_chl1 = np.linspace(0, float(no3_chl1.columns[-1]) + 1, 100)
    X0 = 0.01
    initial = [A_chl0.iloc[0], I_chl0.iloc[0], X0, A_chl1.iloc[0], I_chl1.iloc[0], X0]
    data_str = 'no3_no2'

    # Create lmfit Parameters with optional constraints
    model = model1
    model_str = 'model1'
    params = Parameters()
    #params.add('eps',  value=0.1, min=1e-3, max=1)
    params.add('K_A',  value=0.001, min=1e-4, max=10, vary=False)
    params.add('K_I',  value=0.001, min=1e-4, max=10, vary=False)
    params.add('r_A',  value=0.1, min=1e-4, max=1e3)
    params.add('r_I',  value=0.1, min=1e-4, max=1e3)
    params.add('gamA', value=10.0, min=1e-4, max=1e3)
    params.add('gamI', value=10.0, min=1e-4, max=1e3)
    #params.add('gamI', expr='r_A * gamA / r_I')  # Constraint: r_A * gamA = r_I * gamI
    fitter = Minimizer(residual_chl01, params, fcn_args=(model, initial, t_eval_chl0, t_eval_chl1, A_chl0, I_chl0, A_chl1, I_chl1))
    fit_str = f'{model_str}_chl01_rA_rI_gamA_gamI'
###################################################################################################################################################
    print(f'=====Brute fitting for row {row} started=====')
    result_brute = fitter.minimize(method='brute', Ns=3)
    print(f'=====Brute fitting for row {row} completed=====')
    best_result = copy.deepcopy(result_brute)
    num_iterations = 1
    for candidate in result_brute.candidates:
        print(f'=====Searching candidates {num_iterations}=====')
        trial = fitter.minimize(method='leastsq', params=candidate.params)
        if trial.chisqr < best_result.chisqr:
            best_result = trial
        num_iterations += 1

    result = best_result
    with open(f'fitting_results/{data_str}_{fit_str}_{row}.pkl', 'wb') as f:
        pickle.dump(result, f)
    params = result.params
    print('=====Best-fit values=====')
    for param_name, param in result.params.items():
        print(f'{param_name}: {param.value} ± {param.stderr}')

    # Extract best-fit values
    K_A_best = result.params['K_A'].value
    K_I_best = result.params['K_I'].value
    r_A_best = result.params['r_A'].value
    r_I_best = result.params['r_I'].value
    gamA_best = result.params['gamA'].value
    gamI_best = result.params['gamI'].value

    # Simulate with best-fit parameters (including X0)
    A_chl0_fit, I_chl0_fit = simulate(model, params, initial[0:3], t_eval_chl0)
    
    params_chl1 = params.copy()
    params_chl1['gamA'].value = 0.0
    params_chl1['gamI'].value = 0.0
    A_chl1_fit, I_chl1_fit = simulate(model, params_chl1, initial[3:6], t_eval_chl1)

    # CHL- plot
    plt.figure()
    plt.plot(A_chl0.index.astype(float), I_chl0, 'ro', label='I Data')
    plt.plot(t_eval_chl0, I_chl0_fit, 'r-', label='I Fit')
    plt.plot(A_chl0.index.astype(float), A_chl0, 'bo', label='A Data')
    plt.plot(t_eval_chl0, A_chl0_fit, 'b-', label='A Fit')
    plt.xlabel('Time (hours)', fontsize=15)
    plt.ylabel('Concentration (mM)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.title(f'{data_str}_chl0_{fit_str} ({row})\n'
            f'$r_A$={r_A_best:.3f}, $r_I$={r_I_best:.3f}, $\gamma_A$={gamA_best:.3f}, $\gamma_I$={gamI_best:.3f}\n'
            f'$K_A$={K_A_best:.3f}, $K_I$={K_I_best:.3f}, $X_0$={X0:.2f}',
            fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/model_fit_{data_str}_chl0_{fit_str}_{row}.png')
    plt.close()

    # CHL+ plot
    plt.figure()
    plt.plot(A_chl1.index.astype(float), I_chl1, 'ro', label='I Data')
    plt.plot(t_eval_chl1, I_chl1_fit, 'r-', label='I Fit')
    plt.plot(A_chl1.index.astype(float), A_chl1, 'bo', label='A Data')
    plt.plot(t_eval_chl1, A_chl1_fit, 'b-', label='A Fit')
    plt.xlabel('Time (hours)', fontsize=15)
    plt.ylabel('Concentration (mM)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.title(f'{data_str}_chl1_{fit_str} ({row})\n'
            f'$r_A$={r_A_best:.3f}, $r_I$={r_I_best:.3f}, $\gamma_A$=0.0, $\gamma_I$=0.0\n'
            f'$K_A$={K_A_best:.3f}, $K_I$={K_I_best:.3f}, $X_0$={X0:.2f}',
            fontsize=14)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/model_fit_{data_str}_chl1_{fit_str}_{row}.png')
    plt.close()
  


