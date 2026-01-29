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
from model1_20250414 import *

no3_conc = pd.read_csv('concentrations/no3_chl0_evap.csv', index_col=0)
no2_conc = pd.read_csv('concentrations/no2_chl0_evap.csv', index_col=0)
no3_cons = pd.read_csv('concentrations/no3_chl0_cons.csv', index_col=0)
no2_cons = pd.read_csv('concentrations/no2_chl0_cons.csv', index_col=0)
row = 'B03'
A_data_conc = no3_conc.loc[row]
I_data_conc = no2_conc.loc[row]
A_data_cons = no3_cons.loc[row]
I_data_cons = no2_cons.loc[row]
times = [float(col) for col in A_data_conc.index]

A0 = A_data_conc[0]
I0 = I_data_conc[0]
X0 = 0.01
initial = [A0, I0, X0]

params = Parameters()
#params.add('eps',  value=0.1, min=1e-3, max=1)
params.add('K_A',  value=0.001, min=1e-4, max=10)
params.add('K_I',  value=0.001, min=1e-4, max=10)
params.add('r_A',  value=0.528, min=1e-4, max=1e3)
params.add('r_I',  value=0.399, min=1e-4, max=1e3)
params.add('GamA', value=0.2, min=1e-4, max=1e3)
params.add('GamI', value=0.127, min=1e-4, max=1e3)

times_sim = np.linspace(0, times[-1], 100)
A_sim_conc, I_sim_conc = simulate(params, initial, times_sim)
A_sim_cons = A_sim_conc[0] - A_sim_conc
I_sim_cons = I_sim_conc[0] - I_sim_conc + A_sim_cons

plt.figure()
plt.plot(times, I_data_cons, 'ro', label=f'$-\Delta I \Delta A$ Data')
plt.plot(times_sim, I_sim_cons, 'r-', label='$-\Delta I \Delta A$ Fit')
plt.plot(times, A_data_cons, 'bo', label=f'$-\Delta A$ Data')
plt.plot(times_sim, A_sim_cons, 'b-', label='$-\Delta A$ Fit')
plt.xlabel('Time (hours)', fontsize=15)
plt.ylabel('Concentration (mM)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend()
r_A = params['r_A'].value
r_I = params['r_I'].value
GamA = params['GamA'].value
GamI = params['GamI'].value
plt.title(f'Simulation ({row})\n'
            f'$r_A X(0)$={r_A * X0:.2e}, $r_I X(0)$={r_I * X0:.2e},\n'
            f'$\Gamma_A$={GamA:.2e}, $\Gamma_I$={GamI:.2e}',
            fontsize=14)
plt.tight_layout()
plt.savefig(f'plots/simulate_{row}.png')
print(f'Plot saved as plots/simulate_{row}.png')
plt.show()
plt.close()

