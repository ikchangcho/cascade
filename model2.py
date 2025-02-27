# Last Modified on 2025-02-27 by Ik
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters
from scipy.interpolate import interp1d
import json
import pickle
import copy

def ode_system(t, y, params):
    A, I, X = y
    eps = params['eps'].value
    K_A = params['K_A'].value
    K_I = params['K_I'].value
    r_A = params['r_A'].value
    r_I = params['r_I'].value
    gamA = params['gamA'].value
    gamI = params['gamI'].value
    dA_dt = -A/(A + eps * I + 1e-2) * A/(K_A + A) * r_A * X
    dI_dt = -dA_dt - eps * I/(A + eps * I + 1e-2) * I/(K_I + I) * r_I * X
    dX_dt = (A/(K_A + A) * r_A * gamA + I/(K_I + I) * r_I * gamI) * X
    return [dA_dt, dI_dt, dX_dt]

def simulate(params, t_span, initial, t_eval):
    sol = solve_ivp(ode_system, t_span, initial, t_eval=t_eval,
                    args=(params,),
                    method='BDF', 
                    rtol=1e-3)      # methods: 'RK45', 'RK23', 'Radau', 'BDF', 'LSODA', 'DOP853'
    print(sol.message)
    return sol.y[0], sol.y[1]  # Returns A(t) and I(t)

def residual(params, t_eval, A_data, I_data):
    """
    Returns residuals (Not squared) for A(t) and I(t).
    lmfit expects an array of residuals for least-squares fitting.
    """
    # Simulate
    A_model, I_model = simulate(params,
                                (t_eval[0], t_eval[-1]),
                                [A_data.iloc[0], I_data.iloc[0], params['X0'].value],
                                t_eval)

    # Interpolate model output at the measurement times
    A_interp = interp1d(t_eval, A_model, kind='linear', fill_value='extrapolate')
    I_interp = interp1d(t_eval, I_model, kind='linear', fill_value='extrapolate')

    # Compute residuals: difference between model and data
    delta_A = A_interp(A_data.index.astype(float)) - A_data
    delta_I = I_interp(I_data.index.astype(float)) - I_data

    # Return a 1D array of residuals
    return np.concatenate((delta_A.values, delta_I.values))

