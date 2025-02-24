import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters
from scipy.interpolate import interp1d

# Load your data
no2 = pd.read_csv('20250114/no2_chl-_evap.csv', index_col=0)
no3 = pd.read_csv('20250114/no3_chl-_evap.csv', index_col=0)

def ode_system(t, y, K_A, K_I, r_A, r_I, gamma):
    A, I, X = y
    dA_dt = -A/(K_A + A) * r_A * X
    dI_dt = -dA_dt -I/(K_I + I) * r_I * X
    dX_dt = gamma * (A/(K_A + A) + I/(K_I + I)) * X
    return [dA_dt, dI_dt, dX_dt]

def simulate(K_A, K_I, r_A, r_I, gamma, t_span, initial, t_eval):
    sol = solve_ivp(ode_system, t_span, initial, t_eval=t_eval, args=(K_A, K_I, r_A, r_I, gamma))
    return sol.y[0], sol.y[1]  # Returns A(t) and I(t)

def residual(params, t_eval, A_data, I_data):
    """
    Returns residuals (Not squared) for A(t) and I(t).
    lmfit expects an array of residuals for least-squares fitting.
    """
    # Extract parameters
    K_A = params['K_A'].value
    K_I = params['K_I'].value
    r_A = params['r_A'].value
    r_I = params['r_I'].value
    gamma = params['gamma'].value

    # Simulate
    A_model, I_model = simulate(K_A, K_I, r_A, r_I, gamma,
                                (t_eval[0], t_eval[-1]),
                                [A_data[0], I_data[0], 1.0],
                                t_eval)

    # Interpolate model output at the measurement times
    A_interp = interp1d(t_eval, A_model, kind='linear', fill_value='extrapolate')
    I_interp = interp1d(t_eval, I_model, kind='linear', fill_value='extrapolate')

    # Compute residuals: difference between model and data
    delta_A = A_interp(A_data.index.astype(float)) - A_data
    delta_I = I_interp(I_data.index.astype(float)) - I_data

    # Return a 1D array of residuals to be minimized
    return np.concatenate((delta_A.values, delta_I.values))

# Example usage
row = no3.index[0]    # Just as an example, fitting one row
A_data = no3.loc[row]
I_data = no2.loc[row]

# Create time steps that cover the entire data range
t_eval = np.linspace(0, float(no3.columns[-1]) + 1, 100)

# Create lmfit Parameters with optional constraints
params = Parameters()
params.add('K_A', value=0.1, min=1e-3, max=1.0)
params.add('K_I', value=0.1, min=1e-3, max=1.0)
params.add('r_A', value=1, min=1e-3, max=10)
params.add('r_I', value=1, min=1e-3, max=10)
params.add('gamma', value=1, min=1e-3, max=10)

# Create Minimizer object
fitter = Minimizer(residual, params,
                   fcn_args=(t_eval, A_data, I_data))

# Perform the minimization (Levenberg-Marquardt by default)
result = fitter.minimize(method='leastsq')

# Extract best-fit values
K_A_best = result.params['K_A'].value
K_I_best = result.params['K_I'].value
r_A_best = result.params['r_A'].value
r_I_best = result.params['r_I'].value
gamma_best = result.params['gamma'].value

# Simulate with best-fit parameters
A_fit, I_fit = simulate(K_A_best, K_I_best, r_A_best, r_I_best, gamma_best,
                        (t_eval[0], t_eval[-1]),
                        [A_data[0], I_data[0], 1.0],
                        t_eval)

# Plot
plt.figure()
plt.plot(A_data.index.astype(float), I_data, 'ro', label='I Data')
plt.plot(t_eval, I_fit, 'r-', label='I Fit')
plt.plot(A_data.index.astype(float), A_data, 'bo', label='A Data')
plt.plot(t_eval, A_fit, 'b-', label='A Fit')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title(f'Row {row} Fit:\nK_A={K_A_best:.2f}, K_I={K_I_best:.2f}, gamma={gamma_best:.2f}')
plt.legend()
plt.show()

# If you want to see a summary of fit results
# print(result.fit_report())