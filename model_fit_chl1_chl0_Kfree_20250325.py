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
import lmfit
from model1 import *   ###############################################################################################################

# Load your data
no3_chl0 = pd.read_csv('concentrations/no3_chl0_evap.csv', index_col=0)
no2_chl0 = pd.read_csv('concentrations/no2_chl0_evap.csv', index_col=0)
no3_chl1 = pd.read_csv('concentrations/no3_chl1_evap.csv', index_col=0)
no2_chl1 = pd.read_csv('concentrations/no2_chl1_evap.csv', index_col=0)
# no2 = no2.iloc[:, :10]
# no3 = no3.iloc[:, :10]

for row in ['A01','B01']:
    # Fit CHL+ data
    X0 = 0.01
    A_chl1 = no3_chl1.loc[row]
    I_chl1 = no2_chl1.loc[row]
    t_eval_chl1 = np.linspace(0, float(no3_chl1.columns[-1]) + 1, 100)
    initial_chl1 = [A_chl1.iloc[0], I_chl1.iloc[0], X0] ###############################################################################################################
    data_str_chl1 = 'no3_no2_chl1'  ###############################################################################################################

    # Create lmfit Parameters with optional constraints
    params_chl1 = Parameters()
    params_chl1.add('K_A',  value=0.001, min=1e-3, max=1, vary=False)
    params_chl1.add('K_I',  value=0.001, min=1e-3, max=1, vary=False)
    params_chl1.add('r_A',  value=0.5, min=1e-3, max=1e3)
    params_chl1.add('r_I',  value=0.5, min=1e-3, max=1e3)
    params_chl1.add('gamA', value=0.0, min=1e-3, max=1e3, vary=False)
    params_chl1.add('gamI', value=0.0, min=1e-3, max=1e3, vary=False)
    #params.add('gamI', expr='r_A * gamA / r_I')  # Constraint: r_A * gamA = r_I * gamI
    fit_str_chl1 = f'model1_rA_rI_KA_KI'  ###############################################################################################################

    fitter = Minimizer(residual, params_chl1, fcn_args=(initial_chl1, t_eval_chl1, A_chl1, I_chl1))
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

    result_chl1 = best_result
    with open(f'fitting_results/{data_str_chl1}_{fit_str_chl1}_{row}.pkl', 'wb') as f:
        pickle.dump(result_chl1, f)

    params_chl1 = result_chl1.params
    print('=====Best-fit values=====')
    for param_name, param in result_chl1.params.items():
        print(f'{param_name}: {param.value} ± {param.stderr}')
    
    # CHL- data
    A_chl0 = no3_chl0.loc[row]
    I_chl0 = no2_chl0.loc[row]
    t_eval_chl0 = np.linspace(0, float(no3_chl0.columns[-1]) + 1, 100)
    initial_chl0 = [A_chl0.iloc[0], I_chl0.iloc[0], X0] ###############################################################################################################
    data_str_chl0 = 'no3_no2_chl0'  ###############################################################################################################

    # Use the best-fit parameters from CHL+ data
    K_A = params_chl1['K_A'].value
    K_I = params_chl1['K_I'].value
    r_A = params_chl1['r_A'].value
    r_I = params_chl1['r_I'].value
    gamA = params_chl1['gamA'].value
    gamI = params_chl1['gamI'].value

    params_chl0 = Parameters()
    params_chl0.add('K_A',  value=K_A, min=1e-4, max=10)
    params_chl0.add('K_I',  value=K_I, min=1e-4, max=10)
    params_chl0.add('r_A',  value=r_A, min=1e-4, max=1e3, vary=False)
    params_chl0.add('r_I',  value=r_I, min=1e-4, max=1e3, vary=False)
    params_chl0.add('gamA', value=1.0, min=1e-4, max=10)
    params_chl0.add('gamI', value=0.5, min=1e-4, max=10)
    #params.add('gamI', expr='r_A * gamA / r_I')  # Constraint: r_A * gamA = r_I * gamI
    fit_str_chl0 = f'model1_gamA_gamI_KA_KI'  ###############################################################################################################

    fitter = Minimizer(residual, params_chl0, fcn_args=(initial_chl0, t_eval_chl0, A_chl0, I_chl0))
    print(f'=====Brute fitting for CHL- row {row} started=====')
    result_brute = fitter.minimize(method='brute', Ns=3)
    print(f'=====Brute fitting for CHL- row {row} completed=====')
    best_result = copy.deepcopy(result_brute)
    num_iterations = 1
    for candidate in result_brute.candidates:
        print(f'=====Searching candidates {num_iterations}=====')
        trial = fitter.minimize(method='leastsq', params=candidate.params)
        if trial.chisqr < best_result.chisqr:
            best_result = trial
        num_iterations += 1

    result_chl0 = best_result
    with open(f'fitting_results/{data_str_chl0}_{fit_str_chl0}_{row}.pkl', 'wb') as f:
        pickle.dump(result_chl0, f)

    params_chl0 = result_chl0.params
    print(f'=====Best-fit values for CHL- {row}=====')
    for param_name, param in result_chl0.params.items():
        print(f'{param_name}: {param.value} ± {param.stderr}')

    params_chl1.add('K_A', value=params_chl0['K_A'].value)
    params_chl1.add('K_I', value=params_chl0['K_I'].value)

    # Simulate with best-fit parameters (including X0)
    A_chl0_fit, I_chl0_fit = simulate(params_chl0, initial_chl0, t_eval_chl0)
    A_chl1_fit, I_chl1_fit = simulate(params_chl1, initial_chl1, t_eval_chl1)
    
    # Optimized parameters
    K_A_chl0 = result_chl0.params['K_A'].value
    K_I_chl0 = result_chl0.params['K_I'].value
    r_A_chl0 = result_chl0.params['r_A'].value
    r_I_chl0 = result_chl0.params['r_I'].value
    gamA_chl0 = result_chl0.params['gamA'].value
    gamI_chl0 = result_chl0.params['gamI'].value

    K_A_chl1 = result_chl1.params['K_A'].value
    K_I_chl1 = result_chl1.params['K_I'].value
    r_A_chl1 = result_chl1.params['r_A'].value
    r_I_chl1 = result_chl1.params['r_I'].value
    gamA_chl1 = result_chl1.params['gamA'].value
    gamI_chl1 = result_chl1.params['gamI'].value

    # CHL- plot
    plt.figure()
    plt.plot(A_chl0.index.astype(float), I_chl0, 'ro', label='I Data')
    plt.plot(t_eval_chl0, I_chl0_fit, 'r-', label='I Fit')
    plt.plot(A_chl0.index.astype(float), A_chl0, 'bo', label='A Data')
    plt.plot(t_eval_chl0, A_chl0_fit, 'b-', label='A Fit')
    plt.xlabel('Time (hours)', fontsize=15)
    plt.ylabel('Concentration (mM)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.title(f'{data_str_chl0}_{fit_str_chl0} ({row})\n'
            f'$r_A$={r_A_chl0:.3f}, $r_I$={r_I_chl0:.3f}, $\gamma_A$={gamA_chl0:.3f}, $\gamma_I$={gamI_chl0:.3f},\n'
            f'$K_A$={K_A_chl0:.3f}, $K_I$={K_I_chl0:.3f}, $X_0$={X0:.2f}',
            fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/model_fit_{data_str_chl0}_{fit_str_chl0}_{row}.png')
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
    plt.title(f'{data_str_chl1}_{fit_str_chl1} ({row})\n'
            f'$r_A$={r_A_chl1:.3f}, $r_I$={r_I_chl1:.3f}, $\gamma_A$=0.0, $\gamma_I$=0.0\n'
            f'$K_A$={K_A_chl1:.3f}, $K_I$={K_I_chl1:.3f}, $X_0$={X0:.2f}',
            fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/model_fit_{data_str_chl1}_{fit_str_chl1}_{row}.png')
    plt.close()
