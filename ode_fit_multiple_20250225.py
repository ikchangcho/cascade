import sys
sys.path.append('./functions')
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
no2 = pd.read_csv('20250114/no2_chl-_evap.csv', index_col=0)
no3 = pd.read_csv('20250114/no3_chl-_evap.csv', index_col=0)

# Import lmfit parameters from fitting_result_A01.pkl
with open('20250114/fitting_result_A01.pkl', 'rb') as f:
    result = pickle.load(f)
params = result.params
for param_name, param in result.params.items():
    print(f'{param_name}: {param.value} ± {param.stderr}')
print(f'{param_name}: {param.value} ± {param.stderr}')

for row in no3.index[1:-1:10]:
    print(f'=====Fitting row {row}=====')
    A_data = no3.loc[row]
    I_data = no2.loc[row]
    t_eval = np.linspace(0, float(no3.columns[-1]) + 1, 100)

    # fitter = Minimizer(residual, params, fcn_args=(t_eval, A_data, I_data))
    # result = fitter.minimize(method='leastsq', params=params)

    # print('Final parameters:')
    # for param_name, param in result.params.items():
    #     print(f'{param_name}: {param.value} ± {param.stderr}')

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
    plt.title(f'Row {row} with A01 parameters\n'
          f'$K_A$={K_A_best:.3f}, $K_I$={K_I_best:.3f}, $r_A$={r_A_best:.3f}, $r_I$={r_I_best:.3f},\n'
          f'$\gamma_A$={gamA_best:.3f}, $\gamma_I$={gamI_best:.3f}, $X_0$={X0_best:.3f}',
          fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'20250114/plots_fitting/{row}_A01params.png')
    print(f'=====Saved plot {row}_A01params.png=====')
    #plt.show()


