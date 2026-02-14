import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Callable, Dict, List, Tuple

class ODEfitter:
    """
    A class to fit ODEs to experimental data and infer optimized parameters.
    """
    
    def __init__(
            self,
            id: str, 
            input_dir: str,
            model: Callable,
            meta_col_num: int = 4,
            use_conc_data: bool = True,
    ):
        self.model = model
        self.meta_df = pd.read_csv(f'{input_dir}/{id}_no2_conc.csv', index_col=0).iloc[:, -meta_col_num:]
        if use_conc_data:
            self.no2_df = pd.read_csv(f'{input_dir}/{id}_no2_conc.csv', index_col=0).iloc[:, :-meta_col_num]
            self.no3_df = pd.read_csv(f'{input_dir}/{id}_no3_conc.csv', index_col=0).iloc[:, :-meta_col_num]
        else:
            self.no2_df = pd.read_csv(f'{input_dir}/{id}_no2_cons.csv', index_col=0).iloc[:, :-meta_col_num]
            self.no3_df = pd.read_csv(f'{input_dir}/{id}_no3_cons.csv', index_col=0).iloc[:, :-meta_col_num]
        self.row_labels = self.no2_df.index.tolist()
        self.time = self.no2_df.columns.values.astype(float)
        

    def _solve_ode(
            self,
            params,
            init_cond
    ):
        sol = odeint(self.model, init_cond, self.time, args=(params,))
        return sol
    
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
        
        # Calculate the residuals (difference between model and data)
        residual_no2 = sol[:, no2_index] - no2
        residual_no3 = sol[:, no3_index] - no3
        
        return residual_no2, residual_no3
    
    def fit_for_selected_rows():

        


        
    

def model1(y, t, params):
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

