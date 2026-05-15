import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from lmfit import Minimizer, Parameters
from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt
import copy
import pickle

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
        
        _residual_no2 = sol[:, self.no2_index] - no2_conc
        _residual_no3 = sol[:, self.no3_index] - no3_conc
        
        return _residual_no2, _residual_no3
    
    def fit_for_selected_rows(
            self,
            ids,
            wells: List[str],
            params: Parameters,
            result_fn: str = '',
            plot_fn: str = '',
            plot_cons: bool = True,
            show_plot: bool = True,
            num_rpl: int = 3,
            num_col: int = 6
    ):
        def _residuals(params):
            residuals = []
            for id in ids:
                for well in wells:
                    init_no3 = self.data_dict[id][well].iloc[0, 0]
                    init_no2 = self.data_dict[id][well].iloc[1, 0]
                    if self.model == model1:
                        init_cond = np.array([1.0, init_no3, init_no2])
                    if self.model == model2:
                        init_cond = np.array([1.0, init_no3, init_no2, 1.0])
                    if self.model == model3:
                        init_cond = np.array([1.0, 1.0, init_no3, init_no2])
                    if self.model == model4:
                        init_cond = np.array([1.0, 1.0, init_no3, init_no2, 1.0])
                    _residual_no2, _residual_no3 = self._residual(id, well, params, init_cond)
                    residuals.extend(_residual_no2)
                    residuals.extend(_residual_no3)
            return np.array(residuals)
        
        optimizer = Minimizer(_residuals, params)
        results = optimizer.minimize(method='leastsq')
        print(f"Residual: {results.chisqr}")
        print(results.params.pretty_print())
        
        if result_fn != '':
            params_df = pd.DataFrame({name: [param.value] for name, param in results.params.items()}, index=[', '.join(wells)])
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
                        init_cond = np.array([1.0, init_no3, init_no2])
                    if self.model == model2:
                        init_cond = np.array([1.0, init_no3, init_no2, 1.0])
                    if self.model == model3:
                        init_cond = np.array([1.0, 1.0, init_no3, init_no2])
                    if self.model == model4:
                        init_cond = np.array([1.0, 1.0, init_no3, init_no2, 1.0])
                    
                    t, y = self._solve_ode_for_plot(id, well, results.params, init_cond)
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
                         ', '.join([f'{name}={param.value:.4f}' for name, param in results.params.items()]))
            plt.tight_layout()
            
            if plot_fn != '':
                plt.savefig(f'{self.plot_dir}/{plot_fn}.png', dpi=300)
                print(f'Saved {self.plot_dir}/{plot_fn}.png')
            if show_plot:
                plt.show()

        return results

# When you modify the model, make sure to update the initial conditions in fit_for_selected_rows accordingly in line 122 and 168.
def model1(t, y, params):       # no3_index = 1, no2_index = 2
    X, A, I = y
    Gamma = params['Gamma'].value
    r_A = params['r_A'].value
    r_I = params['r_I'].value
    K_A = params['K_A'].value
    K_I = params['K_I'].value

    A = max(A, 0)
    I = max(I, 0)

    monod_A = A / (K_A + A)
    monod_I = I / (K_I + I)

    dXdt = 0.5 * (monod_A + monod_I) * Gamma * X
    dAdt = -monod_A * r_A * X
    dIdt = -monod_I * r_I * X - dAdt

    return [dXdt, dAdt, dIdt]

def model2(t, y, params):       # no3_index = 1, no2_index = 2
    X, A, I, C = y
    Gamma = params['Gamma'].value
    r_A = params['r_A'].value
    r_I = params['r_I'].value
    r_C = params['r_C'].value
    K_A = params['K_A'].value
    K_I = params['K_I'].value
    K_C = params['K_C'].value

    A = max(A, 0)
    I = max(I, 0)
    C = max(C, 0)

    monod_A = A / (K_A + A)
    monod_I = I / (K_I + I)
    monod_C = C / (K_C + C)

    dXdt = 0.5 * (monod_A + monod_I) * Gamma * X
    dAdt = -monod_A * r_A * X
    dIdt = -monod_I * r_I * X - dAdt
    dCdt = -monod_C * r_C * X
    
    return [dXdt,dAdt,dIdt,dCdt]

def model3(t, y, params):       # no3_index = 2, no2_index = 3
    X_A, X_I, A, I = y
    Gamma_A = params['Gamma_A'].value
    Gamma_I = params['Gamma_I'].value
    r_A = params['r_A'].value
    r_I = params['r_I'].value
    K_A = params['K_A'].value
    K_I = params['K_I'].value

    A = max(A, 0)
    I = max(I, 0)

    monod_A = A / (K_A + A)
    monod_I = I / (K_I + I)

    dX_Adt = monod_A * Gamma_A * X_A
    dX_Idt = monod_I * Gamma_I * X_I
    dAdt = -monod_A * r_A * X_A
    dIdt = -monod_I * r_I * X_I -dAdt

    return [dX_Adt, dX_Idt, dAdt, dIdt]

def model4(t, y, params):       # no3_index = 2, no2_index = 3
    X_A, X_I, A, I, C = y

    A = max(A, 0)
    I = max(I, 0)
    C = max(C, 0)

    Gamma_A = params['Gamma_A'].value
    Gamma_I = params['Gamma_I'].value
    r_A = params['r_A'].value
    r_I = params['r_I'].value
    r_C = params['r_C'].value
    K_A = params['K_A'].value
    K_I = params['K_I'].value
    K_C = params['K_C'].value

    monod_A = A / (K_A + A)
    monod_I = I / (K_I + I)
    monod_C = C / (K_C + C)

    dX_Adt = monod_A * monod_C * Gamma_A * X_A
    dX_Idt = monod_I * monod_C * Gamma_I * X_I
    dAdt = -monod_A * r_A * X_A
    dIdt = -monod_I * r_I * X_I - dAdt 
    dCdt = -monod_C * r_C * (X_A + X_I)

    return [dX_Adt, dX_Idt, dAdt, dIdt, dCdt]



if __name__ == "__main__":
    fitter = ODEfitter(ids=['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5', '4.2.batch6'],
             wells=['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
                    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12',
                    'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12',
                    'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12',
                    'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12',
                    'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12',
                    'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12',
                    'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12'], 
             model=model3, no3_index=1, no2_index=2)
    
    ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5', '4.2.batch6']
    wells_chl1 = ['A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
                    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12',
                    'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12',
                    'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12']
    wells_chl0 = ['E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12',
                    'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12',
                    'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12',
                    'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12']
    results_df = pd.DataFrame(columns=['id', 'well_chl1', 'red_chi2_chl1', 'well_chl0', 'red_chi2_chl0', 'r_A', 'r_I', 'Gamma_A', 'Gamma_I', 'K_A', 'K_I'])
    results_dict = {}

    for id in ids:
        results_dict[id] = {}
        for (well_chl1, well_chl0) in zip(wells_chl1, wells_chl0):    
            # CHL+ fitting
            initial_guess = Parameters()
            initial_guess.add('Gamma_A', value=0, vary=False)
            initial_guess.add('Gamma_I', value=0, vary=False)
            initial_guess.add('r_A', value=0.05, min=1e-3, max=1.0)
            initial_guess.add('r_I', value=0.05, min=1e-3, max=1.0)
            initial_guess.add('K_A', value=1e-3, vary=False)
            initial_guess.add('K_I', value=1e-3, vary=False)
            initial_guess.add('K_C', value=1e-3, vary=False)

            chl1_result = fitter.fit_for_selected_rows([id], [well_chl1], initial_guess,
                show_plot=True, num_rpl=1, num_col=1)
            results_dict[id][well_chl1] = chl1_result

            params = chl1_result.params
            params.add('Gamma_A', value=0.05, min=1e-3, max=1.0)
            params.add('Gamma_I', value=0.05, min=1e-3, max=1.0)
            params.add('r_A', value=params['r_A'].value, vary=False)
            params.add('r_I', value=params['r_I'].value, vary=False)

            chl0_result = fitter.fit_for_selected_rows([id], [well_chl0], params,
                show_plot=True, num_rpl=1, num_col=1)
            results_dict[id][well_chl0] = chl0_result
            
            results_df.loc[len(results_df)] = [id, well_chl1, chl1_result.redchi, well_chl0, chl0_result.redchi, chl0_result.params['r_A'].value, chl0_result.params['r_I'].value, chl0_result.params['Gamma_A'].value, chl0_result.params['Gamma_I'].value, chl0_result.params['K_A'].value, chl0_result.params['K_I'].value]
    
    filename = f'{fitter.result_dir}/20260511_model3_fitting_results'
    results_df.to_csv(f'{fitter.result_dir}/{filename}.csv', index=False)
    print(f'Saved {fitter.result_dir}/{filename}.csv')
    with open(f'{fitter.result_dir}/{filename}.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    print(f'Saved {fitter.result_dir}/{filename}.pkl')


                                                              
                                                                

    


