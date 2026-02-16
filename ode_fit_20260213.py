import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt

class ODEfitter:
    """
    A class to fit ODEs to experimental data and infer optimized parameters.
    """
    
    def __init__(
            self,
            id: str, 
            model: Callable,
            input_dir: str = 'concentrations',
            result_dir: str = 'fitting_results',
            plot_dir: str = 'plots',
            meta_col_num: int = 4,
            use_conc_data: bool = True,
    ):
        self.model = model
        self.input_dir = input_dir
        self.result_dir = result_dir
        self.plot_dir = plot_dir
        self.meta_df = pd.read_csv(f'{input_dir}/{id}_no2_conc.csv', index_col=0).iloc[:, -meta_col_num:]
        if use_conc_data:
            self.no2_df = pd.read_csv(f'{input_dir}/{id}_no2_conc.csv', index_col=0).iloc[:, :-meta_col_num]
            self.no3_df = pd.read_csv(f'{input_dir}/{id}_no3_conc.csv', index_col=0).iloc[:, :-meta_col_num]
        else:
            self.no2_df = pd.read_csv(f'{input_dir}/{id}_no2_cons.csv', index_col=0).iloc[:, :-meta_col_num]
            self.no3_df = pd.read_csv(f'{input_dir}/{id}_no3_cons.csv', index_col=0).iloc[:, :-meta_col_num]
        self.row_labels = self.no2_df.index.tolist()
        self.time = self.no2_df.columns.values.astype(float)
        self.no2_init_df = self.no2_df.iloc[:, 0]
        self.no3_init_df = self.no3_df.iloc[:, 0]
        

    def _solve_ode(
            self,
            params,
            init_cond
    ):
        t_span = (self.time[0], self.time[-1])
        sol = solve_ivp(self.model, t_span, init_cond, 
                        t_eval=self.time, args=(params,), 
                        method='BDF', rtol=1e-6)        # methods: 'RK45', 'RK23', 'Radau', 'BDF', 'LSODA', 'DOP853'
        return sol.y.T

    def _solve_ode_for_plot(
            self,
            params,
            init_cond
    ):
        t_span = (self.time[0], self.time[-1])
        t_eval = np.linspace(self.time[0], self.time[-1], 100)
        sol = solve_ivp(self.model, t_span, init_cond, 
                        t_eval=t_eval, args=(params,), 
                        method='BDF', rtol=1e-6)        # methods: 'RK45', 'RK23', 'Radau', 'BDF', 'LSODA', 'DOP853'
        return sol.t, sol.y.T
    
    def _residual(
            self,
            row_label,
            params,
            init_cond,
            no2_index = 2,
            no3_index = 3
    ):
        no2 = self.no2_df.loc[row_label].values
        no3 = self.no3_df.loc[row_label].values
        sol = self._solve_ode(params, init_cond)
        
        residual_no2 = sol[:, no2_index] - no2
        residual_no3 = sol[:, no3_index] - no3
        
        return residual_no2, residual_no3
    
    def fit_for_selected_rows(
            self,
            row_labels: List[str],
            initial_guess: Dict[str, float],
            bounds: List[Tuple[float, float]],
            result_fn: str = None,
            plot_fn: str = None
    ):
        def objective(param_values):
            # Convert array back to dictionary
            params = dict(zip(initial_guess.keys(), param_values))
            
            total_error = 0
            for row_label in row_labels:
                init_cond = [1.0, self.no3_init_df.loc[row_label], self.no2_init_df.loc[row_label], 1.0]
                residual_no2, residual_no3 = self._residual(row_label, params, init_cond)
                total_error += np.sum(residual_no2**2) + np.sum(residual_no3**2)
            
            return total_error

        result = minimize(objective, list(initial_guess.values()), method='Nelder-Mead', bounds=bounds)
        # Print optimization progress
        print(f"Optimization completed. Success: {result.success}")
        print(f"Number of iterations: {result.nit}")
        print(f"Final objective value: {result.fun}")


        optimized_params = dict(zip(initial_guess.keys(), result.x))
        
        if result_fn is not None:
            optimized_params_df = pd.DataFrame([optimized_params])
            optimized_params_df.to_csv(f'{self.result_dir}/{id}_{result_fn}.csv', index=False)
            print(f'Saved optimized parameters to {self.result_dir}/{id}_{result_fn}.csv')

        if plot_fn is not None:
            for row_label in row_labels:
                time = self.time
                no2 = self.no2_df.loc[row_label].values
                no3 = self.no3_df.loc[row_label].values

                init_cond = [1.0, self.no3_init_df.loc[row_label], self.no2_init_df.loc[row_label], 1.0]
                t, y = self._solve_ode_for_plot(optimized_params, init_cond)
                plt.figure(figsize=(10, 5))
                plt.plot(t, y[:, 2], label='Fitted NO2', color='red')
                plt.plot(t, y[:, 3], label='Fitted NO3', color='blue')
                plt.scatter(time, no2, label='Data NO2', color='red')
                plt.scatter(time, no3, label='Data NO3', color='blue')
                plt.title(f'Model Fit for {row_label}')
                plt.xlabel('Time (hours)')
                plt.ylabel('Concentration (mM)')
                plt.legend()
                plt.savefig(f'{self.plot_dir}/{id}.{row_label}_{plot_fn}.png')
                print(f'Saved {self.plot_dir}/{id}.{row_label}_{plot_fn}.png')

        return optimized_params

    

def model1(t, y, params):
    X, A, I, C = y
    gamma = params['gamma']
    r_A = params['r_A']
    r_I = params['r_I']
    r_C = params['r_C']
    K_A = params['K_A']
    K_I = params['K_I']
    K_C = params['K_C']

    monod_AC = A / (K_A + A) * C / (K_C + C)
    monod_IC = I / (K_I + I) * C / (K_C + C)

    dXdt = (monod_AC + monod_IC) * gamma * X
    dAdt = -monod_AC * r_A * X
    dIdt = -monod_IC * r_I * X + monod_AC * r_A * X
    dCdt = -(monod_AC + monod_IC) * r_C * X
    
    return [dXdt,dAdt,dIdt,dCdt]


if __name__ == "__main__":
    id = '4.2.batch1'
    fitter = ODEfitter(id, model1)
    initial_guess = {
        'gamma': 1e-2,
        'r_A': 0.5,
        'r_I': 0.5,
        'r_C': 0.5,
        'K_A': 0.1,
        'K_I': 0.1,
        'K_C': 0.1
    }
    bounds = [
        (0.0, 1.0),  # gamma
        (0.0, 10.0),  # r_A
        (0.0, 10.0),  # r_I
        (0.0, 10.0),  # r_C
        (0.0, 1.0),  # K_A
        (0.0, 1.0),  # K_I
        (0.0, 1.0),  # K_C
    ]
    row_labels = ['A04', 'A05', 'A06']  # Example row labels to fit
    optimized_params = fitter.fit_for_selected_rows(
        row_labels, 
        initial_guess, 
        bounds,
        result_fn=None,
        plot_fn='model1_fit')