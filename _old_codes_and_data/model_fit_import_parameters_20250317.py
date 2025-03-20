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
from model1 import *

# Load your data
no2 = pd.read_csv('concentrations/no2_chl0_evap.csv', index_col=0)
no3 = pd.read_csv('concentrations/no3_chl0_evap.csv', index_col=0)

# no2 = no2.iloc[:, :10]
# no3 = no3.iloc[:, :10]

for row in no3.index[0:1]:
    # Import lmfit parameters from fitting_result_A01.pkl
    with open(f'fitting_results/no3_no2_chl1_model1_rA_rI_{row}.pkl', 'rb') as f:
        fitting_result = pickle.load(f)

    # Print imported parameters
    print('=====Imported parameters=====')
    for param_name, param in fitting_result.params.items():
        print(f'{param_name}: {param.value} ± {param.stderr}')

    # Save imported parameters
    params = fitting_result.params
    K_A = params['K_A'].value
    K_I = params['K_I'].value
    r_A = params['r_A'].value
    r_I = params['r_I'].value
    gamA = params['gamA'].value
    gamI = params['gamI'].value
    X0 = params['X0'].value
    
    A_data = no3.loc[row]
    I_data = no2.loc[row]
    t_eval = np.linspace(0, float(no3.columns[-1]) + 1, 100)

    #params.add('eps',  value=0.1, min=1e-3, max=1)
    params.add('K_A',  value=K_A, min=1e-5, max=1, vary=False)
    params.add('K_I',  value=K_I, min=1e-5, max=1, vary=False)
    params.add('r_A',  value=r_A, min=1e-3, max=1e3, vary=False)
    params.add('r_I',  value=r_I, min=1e-3, max=1e3, vary=False)
    params.add('gamA', value=1.0, min=1e-3, max=1e3)
    params.add('gamI', value=1.0, min=1e-3, max=1e3)
    #params.add('gamI', expr='r_A * gamA / r_I')  # Constraint: r_A * gamA = r_I * gamI
    params.add('X0',   value=X0, min=1e-3, max=10, vary=False)
    filename_str = 'no3_no2_chl0_model1_gamA_gamI'

###################################################################################################################################################

    fitter = Minimizer(residual, params, fcn_args=(t_eval, A_data, I_data))
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
    params = result.params
    print('=====Best-fit values=====')
    for param_name, param in result.params.items():
        print(f'{param_name}: {param.value} ± {param.stderr}')

    # Extract best-fit values
    #eps_best = result.params['eps'].value
    K_A_best = result.params['K_A'].value
    K_I_best = result.params['K_I'].value
    r_A_best = result.params['r_A'].value
    r_I_best = result.params['r_I'].value
    gamA_best = result.params['gamA'].value
    gamI_best = result.params['gamI'].value
    X0_best   = result.params['X0'].value

    # Simulate with best-fit parameters (including X0)
    A_fit, I_fit = simulate(params,
                            (t_eval[0], t_eval[-1]),
                            [A_data.iloc[0], I_data.iloc[0], X0_best],
                            t_eval)

    # Plot
    plt.figure()
    plt.plot(A_data.index.astype(float), I_data, 'ro', label='I Data')
    plt.plot(t_eval, I_fit, 'r-', label='I Fit')
    plt.plot(A_data.index.astype(float), A_data, 'bo', label='A Data')
    plt.plot(t_eval, A_fit, 'b-', label='A Fit')
    plt.xlabel('Time (hours)', fontsize=15)
    plt.ylabel('Concentration (mM)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.title(f'{filename_str} ({row})\n'
            f'$K_A$={K_A_best:.3f}, $K_I$={K_I_best:.3f}, $r_A$={r_A_best:.3f}, $r_I$={r_I_best:.3f},\n'
            f'$\gamma_A$={gamA_best:.3f}, $\gamma_I$={gamI_best:.3f}, $X_0$={X0_best:.3f}',
            fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/model_fit_{filename_str}_{row}.png')
    with open(f'fitting_results/{filename_str}_{row}.pkl', 'wb') as f:
        pickle.dump(result, f)
    plt.show()
    plt.close()
