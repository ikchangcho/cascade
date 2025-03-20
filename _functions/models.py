# Last Modified on 2025-03-20 by Ik
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters
from scipy.interpolate import interp1d
import json
import pickle
import copy

def model1(t, y, params):
    A, I, X = y
    K_A = params['K_A'].value
    K_I = params['K_I'].value
    r_A = params['r_A'].value
    r_I = params['r_I'].value
    gamA = params['gamA'].value
    gamI = params['gamI'].value
    dA_dt = -A/(K_A + A) * r_A * X
    dI_dt = -dA_dt - I/(K_I + I) * r_I * X
    dX_dt = (A/(K_A + A) * r_A * gamA + I/(K_I + I) * r_I * gamI) * X
    return [dA_dt, dI_dt, dX_dt]    

def model2(t, y, params):
    A, I, X_A, X_I = y
    K_A = params['K_A'].value
    K_I = params['K_I'].value
    r_A = params['r_A'].value
    r_I = params['r_I'].value
    gamA = params['gamA'].value
    gamI = params['gamI'].value
    dA_dt = - A/(K_A + A) * r_A * X_A
    dI_dt = -dA_dt - I/(K_I + I) * r_I * X_I
    dX_A_dt = A/(K_A + A) * r_A * gamA * X_A
    dX_I_dt = I/(K_I + I) * r_I * gamI * X_I
    return [dA_dt, dI_dt, dX_A_dt, dX_I_dt]

def simulate(model, params, initial, t_eval):
    sol = solve_ivp(model, (t_eval[0], t_eval[-1]), initial, t_eval=t_eval,
                    args=(params,),
                    method='BDF', 
                    rtol=1e-6)      # methods: 'RK45', 'RK23', 'Radau', 'BDF', 'LSODA', 'DOP853'
    #print(sol.message)
    return sol.y[0], sol.y[1]  # Returns A(t) and I(t)

def residual(params, model, initial, t_eval, A_data, I_data):
    """
    Returns residuals (Not squared) for A(t) and I(t).
    lmfit expects an array of residuals for least-squares fitting.
    """
    # Simulate
    A_model, I_model = simulate(model, params,
                                [initial[0], initial[1], initial[2]],
                                t_eval)

    # Interpolate model output at the measurement times
    A_interp = interp1d(t_eval, A_model, kind='linear', fill_value='extrapolate')
    I_interp = interp1d(t_eval, I_model, kind='linear', fill_value='extrapolate')

    # Compute residuals: difference between model and data
    delta_A = A_interp(A_data.index.astype(float)) - A_data
    delta_I = I_interp(I_data.index.astype(float)) - I_data

    # Return a 1D array of residuals
    return np.concatenate((delta_A.values, delta_I.values))

def residual_chl01(params, model, t_eval_chl0, t_eval_chl1, A_chl0, I_chl0, A_chl1, I_chl1):
    
    params_chl1 = params.copy()
    params_chl1['gamA'].value = 0.0
    params_chl1['gamI'].value = 0.0

    # Simulate
    A_model_chl0, I_model_chl0 = simulate(model, params,
                                [A_chl0.iloc[0], I_chl0.iloc[0], params['X0'].value],
                                t_eval_chl0)
    
    A_model_chl1, I_model_chl1 = simulate(model, params_chl1,
                                [A_chl1.iloc[0], I_chl1.iloc[0], params_chl1['X0'].value],
                                t_eval_chl1)

    # Interpolate model output at the measurement times
    A_interp_chl0 = interp1d(t_eval_chl0, A_model_chl0, kind='linear', fill_value='extrapolate')
    I_interp_chl0 = interp1d(t_eval_chl0, I_model_chl0, kind='linear', fill_value='extrapolate')
    
    A_interp_chl1 = interp1d(t_eval_chl1, A_model_chl1, kind='linear', fill_value='extrapolate')
    I_interp_chl1 = interp1d(t_eval_chl1, I_model_chl1, kind='linear', fill_value='extrapolate')

    # Compute residuals: difference between model and data
    delta_A_chl0 = A_interp_chl0(A_chl0.index.astype(float)) - A_chl0
    delta_I_chl0 = I_interp_chl0(I_chl0.index.astype(float)) - I_chl0
    
    delta_A_chl1 = A_interp_chl1(A_chl1.index.astype(float)) - A_chl1
    delta_I_chl1 = I_interp_chl1(I_chl1.index.astype(float)) - I_chl1

    # Return a 1D array of residual
    return np.concatenate((delta_A_chl0.values, delta_I_chl0.values, delta_A_chl1.values, delta_I_chl1.values))

