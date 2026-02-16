import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from lmfit import Minimizer, Parameters
from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt

class ODEfitter:
    def __init__(
            self,
            id: str, 
            model: Callable,
            input_dir: str = 'concentrations',
            result_dir: str = 'fitting_results',
            plot_dir: str = 'plots',
            meta_col_num: int = 4,
    ):
        self.model = model
        self.input_dir = input_dir
        self.result_dir = result_dir
        self.plot_dir = plot_dir
        self.meta_df = pd.read_csv(f'{input_dir}/{id}_no2_conc.csv', index_col=0).iloc[:, -meta_col_num:]
        self.no2_conc_df = pd.read_csv(f'{input_dir}/{id}_no2_conc.csv', index_col=0).iloc[:, :-meta_col_num]
        self.no3_conc_df = pd.read_csv(f'{input_dir}/{id}_no3_conc.csv', index_col=0).iloc[:, :-meta_col_num]
        self.no2_cons_df = pd.read_csv(f'{input_dir}/{id}_no2_cons.csv', index_col=0).iloc[:, :-meta_col_num]
        self.no3_cons_df = pd.read_csv(f'{input_dir}/{id}_no3_cons.csv', index_col=0).iloc[:, :-meta_col_num]
        self.row_labels = self.no2_conc_df.index.tolist()
        self.time = self.no2_conc_df.columns.values.astype(float)
        self.no2_init_df = self.no2_conc_df.iloc[:, 0]
        self.no3_init_df = self.no3_conc_df.iloc[:, 0]
        

    def _solve_ode(
            self,
            params,
            init_cond
    ):
        t_span = (self.time[0], self.time[-1])
        sol = solve_ivp(self.model, t_span, init_cond, 
                        t_eval=self.time, args=(params,), 
                        method='BDF', rtol=1e-6)        # methods: 'RK45', 'RK23', 'Radau', 'BDF', 'LSODA', 'DOP853'
        
        if not sol.success:
            print(f"Solver message: {sol.message}")
        if len(sol.t) != len(self.time):
            print(f"Warning: Solver returned {len(sol.t)} time points, expected {len(self.time)}.")
        
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
        no2_conc = self.no2_conc_df.loc[row_label].values
        no3_conc = self.no3_conc_df.loc[row_label].values
        sol = self._solve_ode(params, init_cond)
        
        residual_no2 = sol[:, no2_index] - no2_conc
        residual_no3 = sol[:, no3_index] - no3_conc
        
        return residual_no2, residual_no3
    
    def fit_for_selected_rows(
            self,
            row_labels: List[str],
            params: Parameters,
            result_fn: str = None,
            plot_fn: str = None,
            show_plot: bool = False
    ):
        def residuals(params):
            residuals = []
            for row_label in row_labels:
                init_cond = [1.0, self.no3_init_df.loc[row_label], self.no2_init_df.loc[row_label], 1.0]
                residual_no2, residual_no3 = self._residual(row_label, params, init_cond)
                residuals.extend(residual_no2)
                residuals.extend(residual_no3)
            return np.array(residuals)
        
        optimizer = Minimizer(residuals, params)
        print(f'=====Brute fitting started=====')
        results_brute = optimizer.minimize(method='brute', Ns=3)
        print(f"=====Brute fitting conpleted=====")
        best_result = results_brute.copy()
        num_iterations = 1
        for candidate in results_brute.candidates:
            print(f'=====Searching candidates {num_iterations}=====')
            trial = optimizer.minimize(method='leastsq', params=candidate.params)
            if trial.chisqr < best_result.chisqr:
                best_result = trial
            num_iterations += 1
        print(f"Fitting result for {row_labels}:")
        print(best_result.params.pretty_print())
        
        if result_fn is not None:
            result_dict = {name: param.value for name, param in fitting_result.params.items()}
            pd.Series(result_dict).to_csv(f'{self.result_dir}/{id}_{result_fn}.csv')
            print(f'Saved {self.result_dir}/{id}_{result_fn}.csv')

        if plot_fn is not None or show_plot:
            num_rpl = 1
            num_col = 3 
            num_row = int(np.ceil(len(row_labels) / num_col / num_rpl))
            fig, axes = plt.subplots(num_row, num_col)
            axes = axes.flatten()

            for i, row_label in enumerate(row_labels):
                ax = axes[i // num_rpl]
                time = self.time
                no2_cons = self.no2_cons_df.loc[row_label].values
                no3_cons = self.no3_cons_df.loc[row_label].values
                ax.scatter(time, no2_cons, color='red')
                ax.scatter(time, no3_cons, color='blue')

                init_cond = [1.0, self.no3_init_df.loc[row_label], self.no2_init_df.loc[row_label], 1.0]
                t, y = self._solve_ode_for_plot(fitting_result.params, init_cond)
                no3_cons_fit = y[0, 1] - y[:, 1]
                no2_cons_fit = y[0, 2] - y[:, 2] + no3_cons_fit
                ax.plot(t, no3_cons_fit, color='blue')
                ax.plot(t, no2_cons_fit, color='red')
            handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'$NO_3$ (A)'),
                    plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'$NO_2$ (I)')]
            fig.legend(handles=handles, loc='upper right', fontsize=20)
            fig.suptitle(f'Model Fit for {row_labels}')
            if plot_fn is not None:
                plt.savefig(f'{self.plot_dir}/{id}_{plot_fn}.png')
                print(f'Saved {self.plot_dir}/{id}_{plot_fn}.png')
            if show_plot:
                plt.show()

        return best_result.params

    

def model1(t, y, params):
    X, A, I, C = y
    gamma = params['gamma'].value
    r_A = params['r_A'].value
    r_I = params['r_I'].value
    r_C = params['r_C'].value
    K_A = params['K_A'].value
    K_I = params['K_I'].value
    K_C = params['K_C'].value

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
    
    params = Parameters()
    params.add('gamma', value=1e-2, min=0.0, max=1.0)
    params.add('r_A', value=0.5, min=1e-3, max=10.0)
    params.add('r_I', value=0.5, min=1e-3, max=10.0)
    params.add('r_C', value=0.5, min=1e-3, max=10.0)
    params.add('K_A', value=0.1, min=1e-3, max=1.0)
    params.add('K_I', value=0.1, min=1e-3, max=1.0)
    params.add('K_C', value=0.1, min=1e-3, max=1.0)

    row_labels = ['E04', 'E05', 'E06', 'F07', 'F08', 'F09', 'G10', 'G11', 'G12']  # Example row labels to fit
    fitting_result = fitter.fit_for_selected_rows(
        row_labels, 
        params,
        result_fn=None,
        plot_fn=None,
        show_plot=True)