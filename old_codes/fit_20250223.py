import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load the data
no2 = pd.read_csv('20250114/no2_chl-_evap.csv', index_col=0)
no3 = pd.read_csv('20250114/no3_chl-_evap.csv', index_col=0)

def ode_system(t, y, K_A, K_I, gamma, r_A, r_I):
    A, I, X = y
    dA_dt = -K_A/(K_A + A) * r_A * X
    dI_dt = -K_I/(K_I + I) * r_I * X
    dX_dt = gamma * X
    return [dA_dt, dI_dt, dX_dt]

def simulate(params, t_span, initial, t_eval, r_A, r_I):
    sol = solve_ivp(ode_system, t_span, initial, t_eval=t_eval, args=(*params, r_A, r_I))
    return sol.y[0], sol.y[1]

def loss(params, t_eval, A_data, I_data, r_A, r_I):
    A_model, I_model = simulate(params, (t_eval[0], t_eval[-1]), [A_data[0], I_data[0], 1.0], t_eval, r_A, r_I)
    # Interpolate model results to match the time points of the data
    A_interp = interp1d(t_eval, A_model, kind='linear', fill_value='extrapolate')
    I_interp = interp1d(t_eval, I_model, kind='linear', fill_value='extrapolate')
    A_model_interp = A_interp(A_data.index.astype(float))
    I_model_interp = I_interp(I_data.index.astype(float))
    return ((A_model_interp - A_data)**2 + (I_model_interp - I_data)**2).sum()

print(no3.index[0])

r_A, r_I = 0.1, 0.1
for row in [no3.index[0]]:
    print(f'Fitting row {row}')
    A_data = no3.loc[row]
    I_data = no2.loc[row]
    t_eval = np.linspace(0, float(no3.columns[-1]) + 1, 100)  # Extend slightly beyond the last time point
    res = minimize(loss, [1.0, 1.0, 0.1], args=(t_eval, A_data, I_data, r_A, r_I), method='Nelder-Mead')
    K_A, K_I, gamma = res.x
    A_fit, I_fit = simulate((K_A, K_I, gamma), (t_eval[0], t_eval[-1]), [A_data[0], I_data[0], 1.0], t_eval, r_A, r_I)
    plt.figure()
    plt.plot(A_data.index.astype(float), I_data, 'ro', label='I Data')
    plt.plot(t_eval, I_fit, 'r-', label='I Fit')
    plt.plot(A_data.index.astype(float), A_data, 'bo', label='A Data')
    plt.plot(t_eval, A_fit, 'b-', label='A Fit')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title(f'Row {row}: K_A={K_A:.2f}, K_I={K_I:.2f}, gamma={gamma:.2f}')
    plt.legend()
    plt.show()
