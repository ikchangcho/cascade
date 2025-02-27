import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters
from scipy.interpolate import interp1d
import json
import pickle
import copy
from model2 import *

# Load your data
filepath = '20250114/'
no2 = pd.read_csv(f'{filepath}no2_chl-_evap.csv', index_col=0)
no3 = pd.read_csv(f'{filepath}/no3_chl-_evap.csv', index_col=0)

# Example usage
row = no3.index[0]  # Just as an example, fitting one row
A_data = no3.loc[row]
I_data = no2.loc[row]
t_eval = np.linspace(0, float(no3.columns[-1]) + 1, 100)

# Create lmfit Parameters with optional constraints
params = Parameters()
params.add('eps',  value=0.1, min=1e-3, max=1)
params.add('K_A',  value=0.001, min=1e-3, max=1, vary=False)
params.add('K_I',  value=0.001, min=1e-3, max=1, vary=False)
params.add('r_A',  value=1.0, min=1e-3, max=10)
params.add('r_I',  value=1.0, min=1e-3, max=10)
params.add('gamA', value=1.0, min=1e-3, max=10)
params.add('gamI', expr='r_A * gamA / r_I')  # Constraint: r_A * gamA = r_I * gamI
params.add('X0',   value=1.0, min=1e-3, max=10)
filename_str = '_model2_fixed_K_single_gam'

fitter = Minimizer(residual, params, fcn_args=(t_eval, A_data, I_data))
print('=====Brute fitting started=====')
result_brute = fitter.minimize(method='brute', Ns=3)
print('=====Brute fitting completed=====')
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
eps_best = result.params['eps'].value
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
plt.title(f'Row {row}{filename_str}\n'
          f'$K_A$={K_A_best:.3f}, $K_I$={K_I_best:.3f}, $r_A$={r_A_best:.3f}, $r_I$={r_I_best:.3f},\n'
          f'$\gamma_A$={gamA_best:.3f}, $\gamma_I$={gamI_best:.3f}, $X_0$={X0_best:.3f}, $\epsilon$={eps_best:.3f}',
          fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig(f'{filepath}plots_fitting/{row}{filename_str}.png')
with open(f'{filepath}fitting_result_{row}{filename_str}.pkl', 'wb') as f:
    pickle.dump(result, f)
plt.show()
