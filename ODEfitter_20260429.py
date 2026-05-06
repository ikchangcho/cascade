import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from lmfit import Minimizer, Parameters
from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt
import copy

class ODEfitter:
    def __init__(
            self,
            ids,
            wells,
            model: Callable,
            no3_index: int,
            no2_index: int,
            input_dir: str = 'concentrations',
            result_dir: str = 'fitting_results',
            plot_dir: str = 'plots',
            meta_col_num: int = 4,
    ):
        def load_csv(ids, wells):
            data_dict = {}
            for id in ids:
                data_dict[id] = {}
                no3_conc_df = pd.read_csv(f'{input_dir}/{id}_no3_conc.csv', index_col=0)
                no2_conc_df = pd.read_csv(f'{input_dir}/{id}_no2_conc.csv', index_col=0)
                no3_cons_df = pd.read_csv(f'{input_dir}/{id}_no3_cons.csv', index_col=0)
                no2_cons_df = pd.read_csv(f'{input_dir}/{id}_no2_cons.csv', index_col=0)
                for well in wells:
                    well_data_df = pd.concat([no3_conc_df.loc[[well]].iloc[:,:-meta_col_num], no2_conc_df.loc[[well]].iloc[:,:-meta_col_num], no3_cons_df.loc[[well]].iloc[:,:-meta_col_num], no2_cons_df.loc[[well]].iloc[:,:-meta_col_num]], axis=0)
                    well_data_df.index = ['no3_conc', 'no2_conc', 'no3_cons', 'no2_cons']
                    data_dict[id][well] = well_data_df
            return data_dict

        self.data_dict = load_csv(ids, wells)       # data_dict structure: {id: {well: DataFrame of time points data}}
        self.model = model
        self.no3_index = no3_index
        self.no2_index = no2_index
        self.input_dir = input_dir
        self.result_dir = result_dir
        self.plot_dir = plot_dir
        # self.row_labels = self.no2_conc_df.index.tolist()
        # self.time = self.no2_conc_df.columns.values.astype(float)
        # self.no2_init_df = self.no2_conc_df.iloc[:, 0]
        # self.no3_init_df = self.no3_conc_df.iloc[:, 0]
        

    def _solve_ode(
            self,
            id,
            well,
            params,
            init_cond
    ):
        time = self.data_dict[id][well].columns.values.astype(float)
        t_span = (time[0], time[-1])
        sol = solve_ivp(self.model, t_span, init_cond, 
                        t_eval=time, args=(params,), 
                        method='BDF', rtol=1e-6)        # methods: 'RK45', 'RK23', 'Radau', 'BDF', 'LSODA', 'DOP853'

        if len(sol.t) != len(time):
            print(f"    Expected {len(time)} time points, but got {len(sol.t)}. Solver message: {sol.message}")
        
        return sol.t, sol.y.T

    def _solve_ode_for_plot(
            self,
            id,
            well,
            params,
            init_cond
    ):
        time = self.data_dict[id][well].columns.values.astype(float)
        t_span = (time[0], time[-1])
        t_eval = np.linspace(time[0], time[-1], 100)
        sol = solve_ivp(self.model, t_span, init_cond, 
                        t_eval=t_eval, args=(params,), 
                        method='BDF', rtol=1e-3)        # methods: 'RK45', 'RK23', 'Radau', 'BDF', 'LSODA', 'DOP853'
        
        return sol.t, sol.y.T
    
    def _residual(
            self,
            id,
            well,
            params,
            init_cond,
    ):
        time = self.data_dict[id][well].columns.values.astype(float)
        no2_conc = self.data_dict[id][well].loc['no2_conc'].values
        no3_conc = self.data_dict[id][well].loc['no3_conc'].values
        sol_time, sol = self._solve_ode(id, well, params, init_cond)
        if len(sol_time) < len(time):
            padding = sol[-1, :].reshape(1, -1).repeat(len(time) - len(sol_time), axis=0)
            sol = np.vstack([sol, padding])
            print(f"    Warning: ODE solver returned fewer time points than expected. Appending last solution value to match the length of time points.")
        
        residual_no2 = sol[:, self.no2_index] - no2_conc
        residual_no3 = sol[:, self.no3_index] - no3_conc
        
        return residual_no2, residual_no3
    
    def fit_for_selected_rows(
            self,
            ids,
            wells: List[str],
            params: Parameters,
            skip_fine_tuning: bool = True,
            result_fn: str = '',
            plot_fn: str = '',
            plot_cons: bool = True,
            show_plot: bool = True,
            num_rpl: int = 3,
            num_col: int = 6
    ):
        def _residuals(params):
            max_eval = 1000
            _residuals.call_count += 1
            if _residuals.call_count % 100 == 0:
                print(f"Residuals function called {_residuals.call_count} times")

            residuals = []
            for id in ids:
                for well in wells:
                    init_no3 = self.data_dict[id][well].iloc[0, 0]
                    init_no2 = self.data_dict[id][well].iloc[1, 0]
                    if self.model == model1:
                        init_cond = np.array([1.0, init_no3, init_no2, 1.0])
                    if self.model == model2:
                        init_cond = np.array([1.0, 1.0, init_no3, init_no2, 1.0])
                    if self.model == model3:
                        init_cond = np.array([1.0, 1.0, init_no3, init_no2])
                    residual_no2, residual_no3 = self._residual(id, well, params, init_cond)
                    residuals.extend(residual_no2)
                    residuals.extend(residual_no3)
            return np.array(residuals)
        
        optimizer = Minimizer(_residuals, params)
        print(f'=====Brute fitting started=====')
        _residuals.call_count = 0
        results_brute = optimizer.minimize(method='brute')
        best_result = copy.deepcopy(results_brute)

        if not skip_fine_tuning:
            num_iterations = 1
            for candidate in results_brute.candidates:
                print(f'=====Searching candidates {num_iterations}=====')
                _residuals.call_count = 0
                trial = optimizer.minimize(method='leastsq', params=candidate.params)  
                if _residuals.call_count > 1000:
                    print(f"    Iteration limit reached ({_residuals.call_count} calls), skipping to next candidate")
                    continue
                if trial.chisqr < best_result.chisqr:
                    best_result = trial
                num_iterations += 1
        print(f"Fitting result for {ids} {wells}:")
        print(best_result.params.pretty_print())
        
        params_df = pd.DataFrame({name: [param.value] for name, param in best_result.params.items()}, index=[', '.join(wells)])
        if result_fn != '':
            params_df.to_csv(f'{self.result_dir}/{id}{result_fn}.csv')
            print(f'Saved {self.result_dir}/{id}{result_fn}.csv')

        if plot_fn != '' or show_plot:
            num_row = int(np.ceil(len(ids) * len(wells) / num_col / num_rpl))
            fig, axes = plt.subplots(num_row, num_col, squeeze=False, figsize=(5*num_col, 3*num_row))
            axes = axes.flatten()

            for id in ids:
                for well in wells:
                    init_no3 = self.data_dict[id][well].iloc[0, 0]
                    init_no2 = self.data_dict[id][well].iloc[1, 0]
                    i = ids.index(id) * len(wells) + wells.index(well)
                    ax = axes[i // num_rpl]
                    time = self.data_dict[id][well].columns.values.astype(float)
                    if plot_cons:
                        no3_cons = self.data_dict[id][well].loc['no3_cons'].values
                        no2_cons = self.data_dict[id][well].loc['no2_cons'].values
                        ax.scatter(time, no3_cons, color='blue', s=20)
                        ax.scatter(time, no2_cons, color='red', s=10)
                    else:
                        no3_conc = self.data_dict[id][well].loc['no3_conc'].values
                        no2_conc = self.data_dict[id][well].loc['no2_conc'].values
                        ax.scatter(time, no3_conc, color='blue', s=20)
                        ax.scatter(time, no2_conc, color='red', s=10)
                        
                    if self.model == model1:
                        init_cond = np.array([1.0, init_no3, init_no2, 1.0])
                    if self.model == model2:
                        init_cond = np.array([1.0, 1.0, init_no3, init_no2, 1.0])
                    if self.model == model3:
                        init_cond = np.array([1.0, 1.0, init_no3, init_no2])
                    
                    t, y = self._solve_ode_for_plot(id, well, best_result.params, init_cond)
                    if plot_cons:
                        no3_cons_fit = y[0, self.no3_index] - y[:, self.no3_index]
                        no2_cons_fit = y[0, self.no2_index] - y[:, self.no2_index] + no3_cons_fit
                        ax.plot(t, no3_cons_fit, color='blue')
                        ax.plot(t, no2_cons_fit, color='red')
                    else:
                        ax.plot(t, y[:, self.no3_index], color='blue')
                        ax.plot(t, y[:, self.no2_index], color='red')
            handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'$NO_3$ (A)'),
                    plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'$NO_2$ (I)')]
            fig.legend(handles=handles, loc='upper right')
            fig.suptitle(f'Model Fit for {ids} {wells}\n' + 
                         ', '.join([f'{name}={param.value:.4f}' for name, param in best_result.params.items()]))
            # fig.tight_layout()
            if plot_fn != '':
                plt.savefig(f'{self.plot_dir}/{plot_fn}.png', dpi=300)
                print(f'Saved {self.plot_dir}/{plot_fn}.png')
            if show_plot:
                plt.show()

        return best_result.params


def model1(t, y, params):       # no3_index = 1, no2_index = 2
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

def model2(t, y, params):       # no3_index = 2, no2_index = 3
    X_A, X_I, A, I, C = y
    gamma_A = params['gamma_A'].value
    gamma_I = params['gamma_I'].value
    r_A = params['r_A'].value
    r_I = params['r_I'].value
    r_C = params['r_C'].value
    K_A = params['K_A'].value
    K_I = params['K_I'].value
    K_C = params['K_C'].value

    monod_AC = A / (K_A + A) * C / (K_C + C)
    monod_IC = I / (K_I + I) * C / (K_C + C)

    dX_Adt = monod_AC * gamma_A * X_A
    dX_Idt = monod_IC * gamma_I * X_I
    dAdt = -monod_AC * r_A * X_A
    dIdt = -monod_IC * r_I * X_I + monod_AC * r_A * X_A
    dCdt = -(monod_AC * r_C * X_A + monod_IC * r_C * X_I)

    return [dX_Adt, dX_Idt, dAdt, dIdt, dCdt]

def model3(t, y, params):       # no3_index = 2, no2_index = 3
    X_A, X_I, A, I = y
    gamma_A = params['gamma_A'].value
    gamma_I = params['gamma_I'].value
    r_A = params['r_A'].value
    r_I = params['r_I'].value
    K_A = params['K_A'].value
    K_I = params['K_I'].value

    monod_A = A / (K_A + A)
    monod_I = I / (K_I + I)

    dX_Adt = monod_A * gamma_A * X_A
    dX_Idt = monod_I * gamma_I * X_I
    dAdt = -monod_A * r_A * X_A
    dIdt = -monod_I * r_I * X_I + monod_A * r_A * X_A

    return [dX_Adt, dX_Idt, dAdt, dIdt]

if __name__ == "__main__":
    ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5', '4.2.batch6']
    wells = ['H01', 'H02', 'H03']
    fitter = ODEfitter(ids, wells, model2, no3_index=2, no2_index=3)
    
    global_params = Parameters()
    global_params.add('gamma_A', value=0.03, min=0.01, max=0.05)
    global_params.add('gamma_I', value=0.03, min=0.03, max=0.07)
    global_params.add('r_A', value=0.03, min=0.02, max=0.06)
    global_params.add('r_I', value=0.03, min=0.02, max=0.06)
    global_params.add('r_C', value=5e-3, min=1e-3, max=0.01)
    global_params.add('K_A', value=1e-3, vary=False)
    global_params.add('K_I', value=1e-3, vary=False)
    global_params.add('K_C', value=1e-3, vary=False)

    # global_params = fitter.fit_for_selected_rows(ids, wells, global_params, 
    #     skip_fine_tuning=False, 
    #     plot_fn='model2_global_fit', 
    #     show_plot=False, 
    #     num_rpl=3, num_col=3)

    for id in ids:
        for i in range(0, len(wells), 3):
            wells = wells[i:i+3]
            global_params.add('gamma_A', value=0.01, vary=False)
            global_params.add('gamma_I', value=0.055, vary=False)
            global_params.add('r_A', value=0.0453, vary=False)
            global_params.add('r_I', value=0.0269, vary=False)
            global_params.add('r_C', value=0.0066, min=1e-4, max=0.01, brute_step=3e-3)
            fitter.fit_for_selected_rows([id], wells, global_params, 
                skip_fine_tuning=False, 
                plot_fn=f'{id}_model2_r_C_individual_fit', 
                show_plot=False, 
                num_rpl=3, num_col=2)



    


