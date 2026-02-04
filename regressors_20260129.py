import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class LinearRegressor:
    def __init__(
            self,
            exp_num: float,
            id: str,
            meta_col_num: int=4,
            input_dir: str="concentrations",
            output_dir: str="fitting_results",
            conc: bool=True
    ):
        print(f"Linear Regression on Exp {exp_num} {id} data")
        self.input_dir = input_dir
        self.output_dir = output_dir
        if conc:
            self.no2_df = pd.read_csv(f'{self.input_dir}/{exp_num}.{id}_no2_conc.csv', index_col=0).iloc[:, :-meta_col_num]
            self.no3_df = pd.read_csv(f'{self.input_dir}/{exp_num}.{id}_no3_conc.csv', index_col=0).iloc[:, :-meta_col_num]
        else:
            self.no2_df = pd.read_csv(f'{self.input_dir}/{exp_num}.{id}_no2_cons.csv', index_col=0).iloc[:, :-meta_col_num]
            self.no3_df = pd.read_csv(f'{self.input_dir}/{exp_num}.{id}_no3_cons.csv', index_col=0).iloc[:, :-meta_col_num]

        if not os.path.exists(self.output_dir):
            raise ValueError(f"Output directory {self.output_dir} does not exist.")
        if self.no2_df is None or self.no2_df.empty:
            raise ValueError(f"NO2 data could not be loaded from {self.input_dir}/{exp_num}.{id}_no2_conc.csv or is empty.")
        if self.no3_df is None or self.no3_df.empty:
            raise ValueError(f"NO3 data could not be loaded from {self.input_dir}/{exp_num}.{id}_no3_conc.csv or is empty.")
        
        self.rows = self.no2_df.index.tolist()
        self.time = self.no2_df.columns.values.astype(float)
        self.regression_results = {}

    def _get_time_and_values_for_row(
            self,
            row_label: str,
            column_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if row_label not in self.no2_df.index:
            raise ValueError(f"Row {row_label} not found in dataframe index.")

        x = self.time
        no2 = self.no2_df.loc[row_label].values.astype(float)
        no3 = self.no3_df.loc[row_label].values.astype(float)
        if column_indices is not None:
            if any(idx < 0 or idx >= len(x) for idx in column_indices):
                raise ValueError("One or more indices in column_indices are out of bounds.")
            x = x[column_indices]
            no2 = no2[column_indices]
            no3 = no3[column_indices]

        return x, no2, no3
    
    def fit_for_row(
            self,
            row_label: str,
            column_indices: Optional[List[int]] = None,
            show_plot: bool = False
    ) -> Tuple[float, float, float, float]:
        x, no2, no3 = self._get_time_and_values_for_row(row_label, column_indices)
        
        x = x.reshape(-1, 1)
        time = self.time.reshape(-1, 1)
        no2_fit = LinearRegression()
        no3_fit = LinearRegression()
        no2_fit.fit(x, no2)
        no3_fit.fit(x, no3)
        
        no2_slope = no2_fit.coef_[0]
        no2_intercept = no2_fit.intercept_
        no3_slope = no3_fit.coef_[0]
        no3_intercept = no3_fit.intercept_
        self.regression_results[row_label] = (no2_slope, no2_intercept, no3_slope, no3_intercept)

        print(f"Row {row_label} | NO2: ({no2_slope:.3g}, {no2_intercept:.3g}), NO3: ({no3_slope:.3g}, {no3_intercept:.3g}) | Fit on {column_indices}")
    
        if show_plot:
            y_min = min(np.min(self.no2_df.loc[row_label]), np.min(self.no3_df.loc[row_label]))
            y_max = max(np.max(self.no2_df.loc[row_label]), np.max(self.no3_df.loc[row_label]))
            plt.ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))
            plt.scatter(time, self.no2_df.loc[row_label], color='red', label='NO2 Data')
            plt.plot(time, no2_fit.predict(time), color='red', label='NO2 Fit')
            plt.scatter(time, self.no3_df.loc[row_label], color='blue', label='NO3 Data')
            plt.plot(time, no3_fit.predict(time), color='blue', label='NO3 Fit')
            plt.title(f'Linear Regression on {row_label}', fontsize=15)
            plt.xlabel('Time (hours)', fontsize=14)
            plt.ylabel('Concentration (mM)', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend()
            plt.show()
        
        return (no2_slope, no2_intercept, no3_slope, no3_intercept)
    
    def fit_for_entire_data(
            self,
            time_threshold: float,
            conc_threshold: float = 0.001,
            filename: Optional[str] = None
    ) -> Dict[str, Tuple[float, float, float, float]]:                
        self.regression_results = {}
        for row_label in self.rows:
            no3_values = self.no3_df.loc[row_label].values.astype(float)
            indices_no3_below_threshold = np.where(no3_values <= conc_threshold)[0]
            index_no3_become_zero = indices_no3_below_threshold[0] if len(indices_no3_below_threshold) > 0 else len(no3_values) - 1
            max_time_no3_nonzero = min(self.time[index_no3_become_zero], time_threshold) if index_no3_become_zero > 0 else time_threshold
            column_indices = np.where(self.time <= max_time_no3_nonzero)[0].tolist()
            self.fit_for_row(row_label, column_indices)
                
        if filename is not None:
            output_path = os.path.join(self.output_dir, filename)
            results_df = pd.DataFrame.from_dict(
                self.regression_results, 
                orient='index', 
                columns=['NO2 Slope', 'NO2 Intercept', 'NO3 Slope', 'NO3 Intercept']
            )
            results_df.to_csv(output_path)
            print(f"Linear regression results saved to {output_path}")
                
        return self.regression_results
    
class PolynomialRegressor:
    def __init__(
            self,
            exp_num: float,
            id: str,
            poly_deg: int=2,
            meta_col_num: int=4,
            input_dir: str="concentrations",
            output_dir: str="fitting_results",
    ):
        print(f"Polynomial Regression on Exp {exp_num} {id} data")
        self.poly_deg = poly_deg
        #self.meta_col_num = meta_col_num
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.no2_conc_df = pd.read_csv(f'{self.input_dir}/{exp_num}.{id}_no2_conc.csv', index_col=0).iloc[:, :-meta_col_num]
        self.no3_conc_df = pd.read_csv(f'{self.input_dir}/{exp_num}.{id}_no3_conc.csv', index_col=0).iloc[:, :-meta_col_num]
        self.no2_cons_df = pd.read_csv(f'{self.input_dir}/{exp_num}.{id}_no2_cons.csv', index_col=0).iloc[:, :-meta_col_num]
        self.no3_cons_df = pd.read_csv(f'{self.input_dir}/{exp_num}.{id}_no3_cons.csv', index_col=0).iloc[:, :-meta_col_num]

        if not os.path.exists(self.output_dir):
            raise ValueError(f"Output directory {self.output_dir} does not exist.")
        if self.no2_conc_df is None or self.no2_conc_df.empty or self.no2_cons_df is None or self.no2_cons_df.empty:
            raise ValueError(f"NO2 cdata could not be loaded from {self.input_dir} or is empty.")
        if self.no3_conc_df is None or self.no3_conc_df.empty or self.no3_cons_df is None or self.no3_cons_df.empty:
            raise ValueError(f"NO3 data could not be loaded from {self.input_dir} or is empty.")
        
        self.rows = self.no2_conc_df.index.tolist()
        self.time = self.no2_conc_df.columns.values.astype(float)
        self.regression_results = {}

    def choose_columns_for_regression(
            self,
            row_label: str,
            epsilon: float = 0.01
    ) -> Tuple[List[int], List[int]]:
        init_no3_value = self.no3_conc_df.loc[row_label].values.astype(float)[0]
        indices_for_no3_fit = np.where(self.no3_cons_df.loc[row_label].values.astype(float) < init_no3_value - epsilon)[0]
        init_no2_value = self.no2_conc_df.loc[row_label].values.astype(float)[0]
        indices_for_no2_fit = np.where(self.no2_cons_df.loc[row_label].values.astype(float) < init_no2_value + init_no3_value - epsilon)[0]

        return indices_for_no2_fit.tolist(), indices_for_no3_fit.tolist()
    
    def fit_for_row(
            self,
            row_label: str,
            show_plot: bool = False
    ):
        indices_for_no2_fit, indices_for_no3_fit = self.choose_columns_for_regression(row_label)
        x_no2 = self.time[indices_for_no2_fit].flatten()
        no2_cons = self.no2_cons_df.loc[row_label].values.astype(float)[indices_for_no2_fit].flatten()
        x_no3 = self.time[indices_for_no3_fit].flatten()
        no3_cons = self.no3_cons_df.loc[row_label].values.astype(float)[indices_for_no3_fit].flatten()

        no2_fit = np.poly1d(np.polyfit(x_no2, no2_cons, self.poly_deg))
        no3_fit = np.poly1d(np.polyfit(x_no3, no3_cons, self.poly_deg))

        self.regression_results[row_label] = (no2_fit.coef.tolist(), no3_fit.coef.tolist())

        if show_plot:
            time = self.time.flatten()
            time_array = np.linspace(np.min(time), np.max(time), 100)
            y_min = min(np.min(self.no2_cons_df.loc[row_label]), np.min(self.no3_cons_df.loc[row_label]))
            y_max = max(np.max(self.no2_cons_df.loc[row_label]), np.max(self.no3_cons_df.loc[row_label]))
            plt.ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))

            plt.scatter(time, self.no2_cons_df.loc[row_label], color='red', label='NO2 Data')
            plt.plot(time_array, no2_fit(time_array), color='red', label='NO2 Fit')
            plt.scatter(time, self.no3_cons_df.loc[row_label], color='blue', label='NO3 Data')
            plt.plot(time_array, no3_fit(time_array), color='blue', label='NO3 Fit')
            plt.title(f'{self.poly_deg}nd Polynomial Regression on {row_label}', fontsize=15)
            plt.xlabel('Time (hours)', fontsize=14)
            plt.ylabel('Consumption (mM)', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend()
            plt.show()

if __name__ == "__main__":
    exp_num = 4.2
    ids = ['batch1', 'batch2', 'batch3', 'batch4', 'batch5']
    time_threshold=20
    for id in ids[0:5]:
        # regressor = LinearRegressor(exp_num, id)
        # result = regressor.fit_for_row('E04', [0, 1, 2, 3], show_plot=True)
        # results = regressor.fit_for_entire_data(time_threshold, filename=f'{exp_num}.{id}_linear_regression_results.csv')
    
        regressor = PolynomialRegressor(exp_num, id)
        regressor.fit_for_row('E04', show_plot=True)
        