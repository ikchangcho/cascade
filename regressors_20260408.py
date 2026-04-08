import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy.polynomial import polynomial as P
import pickle
import seaborn as sns

def calculate_mean_and_var(
        filepath: str,
        groupby_cols: List[str] = ['Nitrite_input', 'Nitrate_input', 'Chloramphenicol'],
        drop_cols: List[str] = ['Sample_type'],
        ):
    regression_results_df = pd.read_csv(f'{filepath}.csv', index_col=0)
    # Consider only rows where all drop_cols are NaN (i.e., exclude rows with specific Sample_type)
    regression_results_df = regression_results_df[regression_results_df[drop_cols].isnull().all(axis=1)]
    mean_df = regression_results_df.drop(drop_cols, axis=1).groupby(groupby_cols, as_index=False).mean()
    var_df = regression_results_df.drop(drop_cols, axis=1).groupby(groupby_cols, as_index=False).var()
    mean_and_var = mean_df.merge(var_df, on=groupby_cols, suffixes=('_mean', '_var'))
    mean_and_var = mean_and_var.sort_values(['Chloramphenicol', 'Nitrite_input', 'Nitrate_input'], ascending=False)
    mean_and_var.to_csv(f"{filepath}_mean_var.csv", index=False)
    print(f"Saved {filepath}_mean_var.csv")


class Interpolator:
    def __init__(
            self,
            id: str,
            meta_col_num: int=4,
            input_dir: str="concentrations",
            results_dir: str="fitting_results",
            plots_dir: str="plots"
    ):
        self.id = id
        self.meta_col_num = meta_col_num
        self.input_dir = input_dir
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.meta_df = pd.read_csv(f'{self.input_dir}/{id}_no2_cons.csv', index_col=0).iloc[:, -meta_col_num:]
        self.no3_df = pd.read_csv(f'{self.input_dir}/{id}_no3_cons.csv', index_col=0).iloc[:, :-meta_col_num]
        self.no2_df = pd.read_csv(f'{self.input_dir}/{id}_no2_cons.csv', index_col=0).iloc[:, :-meta_col_num]
        self.no3_conc_df = pd.read_csv(f'{self.input_dir}/{id}_no3_conc.csv', index_col=0).iloc[:, :-meta_col_num]
        self.no2_conc_df = pd.read_csv(f'{self.input_dir}/{id}_no2_conc.csv', index_col=0).iloc[:, :-meta_col_num]
        
        if not os.path.exists(self.results_dir):
            raise ValueError(f"Output directory {self.results_dir} does not exist.")
        if self.no2_df is None or self.no2_df.empty:
            raise ValueError(f"NO2 data could not be loaded from {self.input_dir} or is empty.")
        if self.no3_df is None or self.no3_df.empty:
            raise ValueError(f"NO3 data could not be loaded from {self.input_dir} or is empty.")
        
        self.row_labels = self.no2_df.index.tolist()
        self.time = self.no2_df.columns.values.astype(float)
    
    def values_at_time_points_for_row(
        self,
        row_label: str,
        time_points_no3: List[float],
        time_points_no2: List[float],
        show_plot: bool = False
    ):
        time = self.time
        no3 = self.no3_df.loc[row_label].values.astype(float)
        no2 = self.no2_df.loc[row_label].values.astype(float)

        values_at_time_points_no3 = np.interp(time_points_no3, time, no3)
        values_at_time_points_no2 = np.interp(time_points_no2, time, no2)

        if show_plot:
            plt.plot(time, no3, color='blue', marker='o', s=10, linestyle='--', label='NO3')
            plt.plot(time, no2, color='red', marker='o', s=10, linestyle='--', label='NO2')
            plt.scatter(time_points_no3, values_at_time_points_no3, color='green', marker='+', s=100)
            plt.scatter(time_points_no2, values_at_time_points_no2, color='black', marker='+', s=100)
            plt.xlabel('Time (hours)')
            plt.ylabel('Consumption (mM)')
            plt.title(f'Consumption at Time Points for {self.id} {row_label}')
            plt.legend()
            plt.show()
        
        return values_at_time_points_no3, values_at_time_points_no2
        
    def values_at_time_points_for_selected_rows(
        self,
        row_labels: List[str],
        time_points_no3: List[float],
        time_points_no2: List[float],
        output_fn: str = ''
    ):
        values_dict = {
            'row_label': []
        }
        for time_point in time_points_no3:
            values_dict[f'no3_cons_{time_point}hrs'] = []
        for time_point in time_points_no2:
            values_dict[f'no2_cons_{time_point}hrs'] = []
        
        for row_label in row_labels:
            values_dict['row_label'].append(row_label)
            values_at_time_points_no3, values_at_time_points_no2 = self.values_at_time_points_for_row(row_label, time_points_no3, time_points_no2)
            for i, time_point in enumerate(time_points_no3):
                values_dict[f'no3_cons_{time_point}hrs'].append(values_at_time_points_no3[i])
            for i, time_point in enumerate(time_points_no2):
                values_dict[f'no2_cons_{time_point}hrs'].append(values_at_time_points_no2[i])

        values_df = pd.DataFrame(values_dict).set_index('row_label')
        values_df = values_df.join(self.meta_df.loc[values_df.index])

        if output_fn != '':
            values_df.to_csv(f'{self.results_dir}/{output_fn}.csv')
            print(f"Saved {self.results_dir}/{output_fn}.csv")
            calculate_mean_and_var(f'{self.results_dir}/{output_fn}')
        
        return values_df

    def half_life_for_row(
            self,
            row_label: str,
            half: float = 0.5,
            epsilon: float = 0.01,
            show_plot: bool = False,
            outpuf_fn: str = ''
    ):
        time = self.time
        no2 = self.no2_df.loc[row_label].values.astype(float)
        no3 = self.no3_df.loc[row_label].values.astype(float)
        no2_half = np.max(no2) * half - epsilon
        no3_half = np.max(no3) * half - epsilon
        
        smallest_index_no2_above_half = np.where(no2 > no2_half)[0][0]
        smallest_index_no3_above_half = np.where(no3 > no3_half)[0][0]
        
        if smallest_index_no2_above_half == 0:
            no2_half_life = None
        else:
            no2_half_life = np.interp(no2_half, [no2[smallest_index_no2_above_half - 1], no2[smallest_index_no2_above_half]], [time[smallest_index_no2_above_half - 1], time[smallest_index_no2_above_half]])
        if smallest_index_no3_above_half == 0:
            no3_half_life = None
        else:
            no3_half_life = np.interp(no3_half, [no3[smallest_index_no3_above_half - 1], no3[smallest_index_no3_above_half]], [time[smallest_index_no3_above_half - 1], time[smallest_index_no3_above_half]])

        time_interp = np.linspace(np.min(time), np.max(time), 100)
        no2_interp = np.interp(time_interp, time, no2)
        no3_interp = np.interp(time_interp, time, no3)
        if show_plot or outpuf_fn != '':
            plt.scatter(time, no2, color='red', marker='o', label='NO2 Data')
            plt.plot(time_interp, no2_interp, color='red', linestyle='-')
            plt.axhline(no2_half, color='red', linestyle='--', label='NO2 Half Level')
            plt.axvline(no2_half_life, color='red', linestyle=':', label=f'NO2 Half Time: {no2_half_life:.2f} hrs')
            plt.scatter(time, no3, color='blue', marker='o', label='NO3 Data')
            plt.plot(time_interp, no3_interp, color='blue', linestyle='-')
            plt.axhline(no3_half, color='blue', linestyle='--', label='NO3 Half Level')
            plt.axvline(no3_half_life, color='blue', linestyle=':', label=f'NO3 Half Time: {no3_half_life:.2f} hrs')
            plt.xlabel('Time (hours)')
            plt.ylabel('Concentration (mM)')
            plt.title(f'Half-life Interpolation for {row_label}')
            plt.legend()
            if outpuf_fn != '':
                plt.savefig(f'{self.plots_dir}/{outpuf_fn}.png', dpi=300, bbox_inches='tight')
                print(f'Saved {self.plots_dir}/{outpuf_fn}.png')
            if show_plot:
                plt.show()
        
        return no2_half_life, no3_half_life
    
    def half_life_for_selected_rows(
            self,
            row_labels: List[str],
            half: float = 0.5,
            output_fn: Optional[str] = None
    ):
        half_life_data = []
        for row_label in row_labels:
            no2_half_life, no3_half_life = self.half_life_for_row(row_label, half)
            half_life_data.append({'row_label': row_label, 'no2_half_life': no2_half_life, 'no3_half_life': no3_half_life})
        half_life_df = pd.DataFrame(half_life_data).set_index('row_label')
        half_life_df = half_life_df.join(self.meta_df.loc[half_life_df.index])

        if output_fn is not None:
            half_life_df.to_csv(f'{self.results_dir}/{output_fn}.csv')
            print(f"Saved {self.results_dir}/{output_fn}.csv")
        
        return half_life_df
    
    def half_rate_for_row(
            self,
            row_label: str,
            half: float = 0.5,
            epsilon: float = 0.01,
            show_plot: bool = False,
            output_fn: str = ''
    ):
        time = self.time
        no2 = self.no2_df.loc[row_label].values.astype(float)
        no2_conc = self.no2_conc_df.loc[row_label].values.astype(float)
        no3 = self.no3_df.loc[row_label].values.astype(float)
        no3_conc = self.no3_conc_df.loc[row_label].values.astype(float)

        no3_half = no3_conc[0] * half - epsilon
        no2_half = (no2_conc[0] + no3_conc[0]) * half - epsilon

        if np.sum(no2 > no2_half) == 0:
            RuntimeError(f"Warning: NO2 consumption for row {row_label} never goes above half level. Cannot calculate half-life and half-rate for NO2.")
        smallest_index_no2_above_half = np.where(no2 > no2_half)[0][0]
        if np.sum(no3 > no3_half) == 0:
            RuntimeError(f"Warning: NO3 consumption for row {row_label} never goes above half level. Cannot calculate half-life and half-rate for NO3.")
        smallest_index_no3_above_half = np.where(no3 > no3_half)[0][0]

        if smallest_index_no2_above_half == 0:
            no2_half_life = None
            no2_half_rate = None
        else:
            no2_half_life = np.interp(no2_half, [no2[smallest_index_no2_above_half - 1], no2[smallest_index_no2_above_half]], [time[smallest_index_no2_above_half - 1], time[smallest_index_no2_above_half]])
            if no2_half_life < 0:
                print(f"Warning: Calculated NO2 half-life for row {row_label} is negative.")
            no2_half_rate = no2_half / no2_half_life
        if smallest_index_no3_above_half == 0:
            no3_half_life = None
            no3_half_rate = None
        else:
            no3_half_life = np.interp(no3_half, [no3[smallest_index_no3_above_half - 1], no3[smallest_index_no3_above_half]], [time[smallest_index_no3_above_half - 1], time[smallest_index_no3_above_half]])
            if no3_half_life < 0:
                print(f"Warning: Calculated NO3 half-life for row {row_label} is negative.")
            no3_half_rate = no3_half / no3_half_life
        
        return no3_half_rate, no2_half_rate
    
    def half_rate_for_selected_rows(
            self,
            row_labels: List[str],
            half: float = 0.5,
            output_fn: Optional[str] = None
    ):
        half_rate_data = []
        for row_label in row_labels:
            no3_half_rate, no2_half_rate = self.half_rate_for_row(row_label, half)
            half_rate_data.append({'row_label': row_label, 'no3_half_rate': no3_half_rate, 'no2_half_rate': no2_half_rate})
        half_rate_df = pd.DataFrame(half_rate_data).set_index('row_label')
        half_rate_df = half_rate_df.join(self.meta_df.loc[half_rate_df.index])

        if output_fn is not None:
            half_rate_df.to_csv(f'{self.results_dir}/{output_fn}.csv')
            print(f"Saved {self.results_dir}/{output_fn}.csv")
        
        return half_rate_df
            
            
    
    def auc_for_row(
            self,
            row_label: str,
            time_range: float,
            show_plot: bool = False
    ):
        time = self.time
        no3 = self.no3_df.loc[row_label].values.astype(float)
        no2 = self.no2_df.loc[row_label].values.astype(float)

        mask = np.where(time <= time_range)[0]
        next_indices = np.where(time > time_range)[0]
        if len(next_indices) > 0:
            mask = np.append(mask, next_indices[0])
        
        time_interp = np.linspace(0, time_range, 100)
        no3_interp = np.interp(time_interp, time, no3)
        no2_interp = np.interp(time_interp, time, no2)
        
        no3_auc = np.trapezoid(no3_interp, time_interp)
        no2_auc = np.trapezoid(no2_interp, time_interp)

        if show_plot:
            plt.scatter(time, no3, color='blue', marker='o', label='NO3')
            plt.plot(time_interp, no3_interp, color='blue', linestyle='-')
            plt.scatter(time, no2, color='red', marker='o', label='NO2')
            plt.plot(time_interp, no2_interp, color='red', linestyle='-')
            plt.axvline(time_range, color='green', linestyle='--')
            plt.fill_between(time_interp, no3_interp, where=(time_interp <= time_range), alpha=0.3, color='blue')
            plt.fill_between(time_interp, no2_interp, where=(time_interp <= time_range), alpha=0.3, color='red')
            plt.xlabel('Time (hours)')
            plt.ylabel('Consumption (mM)')
            plt.title(f'AUC for {self.id} {row_label} data up to {time_range} hours \nNO3 AUC: {no3_auc:.2f}, NO2 AUC: {no2_auc:.2f}')
            plt.legend()
            plt.show()

        return no3_auc, no2_auc

    def auc_for_selected_rows(
        self,
        row_labels: List[str],
        time_ranges_no3: List[float],
        time_ranges_no2: List[float],
        output_fn: str = ''
    ):
        auc_dict = {
            'row_label': []
        }
        for time_range in time_ranges_no3:
            auc_dict[f'no3_auc_{time_range}hrs'] = []
        for time_range in time_ranges_no2:
            auc_dict[f'no2_auc_{time_range}hrs'] = []

        for row_label in row_labels:
            auc_dict['row_label'].append(row_label)
            for time_range in time_ranges_no3:
                no3_auc, _ = self.auc_for_row(row_label, time_range)
                auc_dict[f'no3_auc_{time_range}hrs'].append(no3_auc)
            for time_range in time_ranges_no2:
                _, no2_auc = self.auc_for_row(row_label, time_range)
                auc_dict[f'no2_auc_{time_range}hrs'].append(no2_auc)
        
        auc_df = pd.DataFrame(auc_dict).set_index('row_label')
        auc_df = auc_df.join(self.meta_df.loc[auc_df.index])

        if output_fn != '':
            auc_df.to_csv(f'{self.results_dir}/{output_fn}.csv')
            print(f"AUC results saved to {self.results_dir}/{output_fn}.csv")
            calculate_mean_and_var(f'{self.results_dir}/{output_fn}')
        
        return auc_df
        



class LinearRegressor:
    '''
    Fits a linear model on early time points of concentration data for each row. The time points used for fitting are determined by the time when NO3 concentration becomes zero or
    a specified time threshold, whichever is earlier. The regression results include the slope and intercept for both NO2 and NO3, as well as the calculated rates of NO2 reduction 
    and NO3 reduction. The results can be saved to a CSV file and visualized with optional plots.
    '''
    def __init__(
            self,
            id: str,
            time_threshold: float,
            time_interval: (Optional[Tuple[float, float]]) = None,
            meta_col_num: int=4,
            input_dir: str="concentrations",
            results_dir: str="fitting_results",
            plots_dir: str="plots",
            use_conc_data: bool=True
    ):
        self.meta_col_num = meta_col_num
        self.time_threshold = time_threshold
        self.time_interval = time_interval
        self.input_dir = input_dir
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.meta_df = pd.read_csv(f'{self.input_dir}/{id}_no2_conc.csv', index_col=0).iloc[:, -meta_col_num:]  # Load metadata columns only
        if use_conc_data:
            self.no2_df = pd.read_csv(f'{self.input_dir}/{id}_no2_conc.csv', index_col=0).iloc[:, :-meta_col_num]
            self.no3_df = pd.read_csv(f'{self.input_dir}/{id}_no3_conc.csv', index_col=0).iloc[:, :-meta_col_num]
        else:
            self.no2_df = pd.read_csv(f'{self.input_dir}/{id}_no2_cons.csv', index_col=0).iloc[:, :-meta_col_num]
            self.no3_df = pd.read_csv(f'{self.input_dir}/{id}_no3_cons.csv', index_col=0).iloc[:, :-meta_col_num]

        if not os.path.exists(self.results_dir):
            raise ValueError(f"Output directory {self.results_dir} does not exist.")
        if self.no2_df is None or self.no2_df.empty:
            raise ValueError(f"NO2 data could not be loaded from {self.input_dir}/{id}_no2_conc.csv or is empty.")
        if self.no3_df is None or self.no3_df.empty:
            raise ValueError(f"NO3 data could not be loaded from {self.input_dir}/{id}_no3_conc.csv or is empty.")
        
        print(f"======Linear Regression on Exp {id} data======")
        self.row_labels = self.no2_df.index.tolist()
        self.time = self.no2_df.columns.values.astype(float)
        self.regression_results = {}

    def _get_time_and_values_for_row(
            self,
            row_label: str,
            column_indices: Optional[List[int]] = None
    ):
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
            show_plot: bool = False,
            epsilon: float = 0.05
    ):
        x, no2, no3 = self._get_time_and_values_for_row(row_label, column_indices)
        x = x.reshape(-1, 1)
        time = self.time.reshape(-1, 1)

        if no3[0] < epsilon:
            print(f"Initial NO3 concentration for row {row_label} is zero. Save None for NO3 fit parameters.")
            no3_slope = 0
            no3_intercept = 0
            no3_rate = None
        else:
            no3_fit = LinearRegression()
            no3_fit.fit(x, no3)
            no3_slope = no3_fit.coef_[0]
            no3_intercept = no3_fit.intercept_
            no3_rate = -no3_slope

        if no2[0] < epsilon and no3[0] < epsilon:
            print(f"Initial NO2 and NO3 concentrations for row {row_label} are zero. Save None for NO2 fit parameters.")
            no2_slope = 0
            no2_intercept = 0
            no2_rate = None
        else:     
            no2_fit = LinearRegression()
            no2_fit.fit(x, no2)
            no2_slope = no2_fit.coef_[0] 
            no2_intercept = no2_fit.intercept_
            no2_rate = -no2_slope -no3_slope

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
        
        return (no3_rate, no3_slope, no3_intercept, no2_rate, no2_slope, no2_intercept)
    
    def fit_for_selected_rows(
            self,
            output_fn,
            row_labels: List[str] = None,
            conc_threshold: float = 0.1            
    ):                
        if row_labels is None:
            row_labels = self.row_labels
        self.regression_results = {}
        for row_label in row_labels:
            no3_values = self.no3_df.loc[row_label].values.astype(float)
            if no3_values[0] > 0.3:
                indices_no3_below_threshold = np.where(no3_values <= conc_threshold)[0]
                index_no3_become_zero = indices_no3_below_threshold[0] if len(indices_no3_below_threshold) > 0 else len(no3_values) - 1
                max_time_no3_nonzero = min(self.time[index_no3_become_zero], self.time_threshold) if index_no3_become_zero > 0 else self.time_threshold
                column_indices = np.where(self.time <= max_time_no3_nonzero)[0].tolist()
            else:
                no2_values = self.no2_df.loc[row_label].values.astype(float)
                indices_no2_below_threshold = np.where(no2_values <= conc_threshold)[0]
                index_no2_become_zero = indices_no2_below_threshold[0] if len(indices_no2_below_threshold) > 0 else len(no2_values) - 1
                max_time_no2_nonzero = min(self.time[index_no2_become_zero], self.time_threshold) if index_no2_become_zero > 0 else self.time_threshold
                column_indices = np.where(self.time <= max_time_no2_nonzero)[0].tolist()
            no3_rate_early, no3_slope_early, no3_intercept_early, no2_rate_early, no2_slope_early, no2_intercept_early = self.fit_for_row(row_label, column_indices)
            
            if self.time_interval is None:
                self.regression_results[row_label] = (no3_rate_early, no3_slope_early, no3_intercept_early, no2_rate_early, no2_slope_early, no2_intercept_early)         

                regression_results_df = pd.DataFrame.from_dict(
                    self.regression_results, 
                    orient='index', 
                    columns=['no3_rate_early', 'no3_slope', 'no3_intercept', 'no2_rate_early', 'no2_slope', 'no2_intercept']
                )
                regression_results_df = regression_results_df.join(self.meta_df.loc[regression_results_df.index])
                regression_results_df.to_csv(f'{self.results_dir}/{output_fn}.csv')
                print(f"Linear regression results saved to {self.results_dir}/{output_fn}.csv")
            
            else:
                column_indices = np.where((self.time >= self.time_interval[0]) & (self.time <= self.time_interval[1]))[0].tolist()
                no3_rate_late, no3_slope_late, no3_intercept_late, no2_rate_late, no2_slope_late, no2_intercept_late = self.fit_for_row(row_label, column_indices)
                self.regression_results[row_label] = (no3_rate_early, no3_slope_early, no3_intercept_early, no2_rate_early, no2_slope_early, no2_intercept_early, no3_rate_late, no3_slope_late, no3_intercept_late, no2_rate_late, no2_slope_late, no2_intercept_late)

                regression_results_df = pd.DataFrame.from_dict(
                    self.regression_results, 
                    orient='index', 
                    columns=['no3_rate_early', 'no3_slope_early', 'no3_intercept_early', 'no2_rate_early', 'no2_slope_early', 'no2_intercept_early', 'no3_rate_late', 'no3_slope_late', 'no3_intercept_late', 'no2_rate_late', 'no2_slope_late', 'no2_intercept_late']
                )
                regression_results_df = regression_results_df.join(self.meta_df.loc[regression_results_df.index])
                regression_results_df.to_csv(f'{self.results_dir}/{output_fn}.csv')
                print(f"Linear regression results saved to {self.results_dir}/{output_fn}.csv")
            calculate_mean_and_var(f'{self.results_dir}/{output_fn}')

        return self.regression_results

    
    def plot_selected_rows(
            self,
            title: str,
            output_fn: str,
            row_labels: List[str] = None,
            show_plot: bool = False
    ):
        if row_labels is None:
            row_labels = self.row_labels
        
        num_rpl = 3
        num_col = 4
        num_row = int(np.ceil(len(row_labels) / num_col / num_rpl))
        fig, axes = plt.subplots(num_row, num_col, figsize=(4*num_row, 4*num_col))
        axes = axes.flatten()
        
        time = self.time.flatten()
        time_array = np.linspace(np.min(time), np.max(time), 100)
        
        excluded_rows = ['A01', 'A02', 'A03', 'E01', 'E02', 'E03']
        all_values = pd.concat([self.no2_df.drop(excluded_rows, errors='ignore'), self.no3_df.drop(excluded_rows, errors='ignore')])
        y_min = all_values.min().min()
        y_max = all_values.max().max()

        marker_styles = ['o', 's', '^']
        for i, row in enumerate(row_labels):
            marker = marker_styles[i % num_rpl]
            ax = axes[i // num_rpl]
            ax.set_ylim(y_min, y_max)
            ax.scatter(time, self.no2_df.loc[row], color='r', marker=marker)
            ax.scatter(time, self.no3_df.loc[row], color='b', marker=marker)
            
            if self.time_interval is None:
                _, no3_slope_early, no3_intercept_early, _, no2_slope_early, no2_intercept_early = self.regression_results[row]
                ax.plot(time_array, no2_slope_early * time_array + no2_intercept_early, 'r-')
                ax.plot(time_array, no3_slope_early * time_array + no3_intercept_early, 'b-')
                suptitle = f'{title}\nLinear Regression on (0, {self.time_threshold})'
            
            else:
                _, no3_slope_early, no3_intercept_early, _, no2_slope_early, no2_intercept_early, _, no3_slope_late, no3_intercept_late, _, no2_slope_late, no2_intercept_late = self.regression_results[row]
                ax.plot(time_array, no2_slope_early * time_array + no2_intercept_early, 'r-')
                ax.plot(time_array, no3_slope_early * time_array + no3_intercept_early, 'b-')
                ax.plot(time_array, no2_slope_late * time_array + no2_intercept_late, 'r--')
                ax.plot(time_array, no3_slope_late * time_array + no3_intercept_late, 'b--')
                suptitle = f'{title}\nLinear Regression on (0, {self.time_threshold}) and ({self.time_interval[0]}, {self.time_interval[1]})'


        handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'$NO_3$ (A)'),
                    plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'$NO_2$ (I)')]
        fig.legend(handles=handles, loc='upper right', fontsize=15)
        fig.suptitle(suptitle, fontsize=20, fontweight='bold')
        plt.savefig(f'{self.plots_dir}/{output_fn}.png', dpi=300, bbox_inches='tight')
        print(f'Saved {self.plots_dir}/{output_fn}.png')
        if show_plot:
            plt.show()
        plt.close()

        return

class PolynomialRegressor:
    def __init__(
            self,
            id: str,
            poly_deg: int=2,
            meta_col_num: int=4,
            input_dir: str="concentrations",
            results_dir: str="fitting_results",
            plots_dir: str="plots"
    ):
        print(f"======Polynomial Regression on Exp {id} Consumption data======")
        self.poly_deg_no2 = poly_deg
        self.poly_deg_no3 = poly_deg
        self.input_dir = input_dir
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.meta_col_num = meta_col_num
        self.meta_df = pd.read_csv(f'{self.input_dir}/{id}_no2_cons.csv', index_col=0).iloc[:, -meta_col_num:]  # Load metadata columns only
        self.no2_cons_df = pd.read_csv(f'{self.input_dir}/{id}_no2_cons.csv', index_col=0).iloc[:, :-meta_col_num]    # Exclude metadata columns
        self.no3_cons_df = pd.read_csv(f'{self.input_dir}/{id}_no3_cons.csv', index_col=0).iloc[:, :-meta_col_num]

        if not os.path.exists(self.results_dir):
            raise ValueError(f"Output directory {self.results_dir} does not exist.")
        if self.no2_cons_df is None or self.no2_cons_df.empty:
            raise ValueError(f"NO2 consumption data could not be loaded from {self.input_dir} or is empty.")
        if self.no3_cons_df is None or self.no3_cons_df.empty:
            raise ValueError(f"NO3 consumption data could not be loaded from {self.input_dir} or is empty.")
        
        self.no2_total_cons = self.no2_cons_df.iloc[:, -1]  # Total consumption values for NO2
        self.no3_total_cons = self.no3_cons_df.iloc[:, -1]
        self.rows = self.no2_cons_df.index.tolist()
        self.time = self.no2_cons_df.columns.values.astype(float)

    def _choose_columns_for_regression(
            self,
            row_label: str,
            mask: Optional[List[int]] = None,
            epsilon: float = 0.02
    ):
        indices_for_no3_fit = np.where(self.no3_cons_df.loc[row_label].values.astype(float) < self.no3_total_cons[row_label] - epsilon)[0].tolist()            
        poly_deg_no3 = self.poly_deg_no3
        if 0 < len(indices_for_no3_fit) < 3:
            indices_for_no3_fit = [0, 1]
            poly_deg_no3 = 1
        if len(indices_for_no3_fit) == 0:
            indices_for_no3_fit = [0]
            poly_deg_no3 = 0

        indices_for_no2_fit = np.where(self.no2_cons_df.loc[row_label].values.astype(float) < self.no2_total_cons[row_label] - epsilon)[0].tolist()
        poly_deg_no2 = self.poly_deg_no2
        if 0 < len(indices_for_no2_fit) < 3:
            indices_for_no2_fit = [0, 1]
            poly_deg_no2 = 1
        if len(indices_for_no2_fit) == 0:
            indices_for_no2_fit = [0]
            poly_deg_no2 = 0
        
        if mask is not None:
            indices_for_no2_fit = [idx for idx in indices_for_no2_fit if idx not in mask]
            indices_for_no3_fit = [idx for idx in indices_for_no3_fit if idx not in mask]
        return indices_for_no2_fit, poly_deg_no2, indices_for_no3_fit, poly_deg_no3
    
    def fit_for_row(
            self,
            row_label: str,
            mask: Optional[List[int]] = None,
            show_plot: bool = False
    ):
        indices_for_no2_fit, poly_deg_no2, indices_for_no3_fit, poly_deg_no3 = self._choose_columns_for_regression(row_label, mask)
        x_no2 = self.time[indices_for_no2_fit].flatten()
        no2_cons = self.no2_cons_df.loc[row_label].values.astype(float)[indices_for_no2_fit].flatten()
        x_no3 = self.time[indices_for_no3_fit].flatten()
        no3_cons = self.no3_cons_df.loc[row_label].values.astype(float)[indices_for_no3_fit].flatten()

        # Debugging prints
        print(f"Fitting row {row_label}:")
        print(f"  NO2 fit indices: {indices_for_no2_fit}")
        print(f"  NO3 fit indices: {indices_for_no3_fit}")

        no2_fit = np.poly1d(np.polyfit(x_no2, no2_cons, poly_deg_no2))
        no3_fit = np.poly1d(np.polyfit(x_no3, no3_cons, poly_deg_no3))

        if show_plot:
            plt.close('all')
            time = self.time.flatten()
            time_array = np.linspace(np.min(time), np.max(time), 100)
            y_min = min(np.min(self.no2_cons_df.loc[row_label]), np.min(self.no3_cons_df.loc[row_label]))
            y_max = max(np.max(self.no2_cons_df.loc[row_label]), np.max(self.no3_cons_df.loc[row_label]))
            plt.ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))

            plt.scatter(time, self.no2_cons_df.loc[row_label], color='red', label='NO2 Data')
            plt.plot(time_array, no2_fit(time_array), color='red', label='NO2 Fit')
            plt.scatter(time, self.no3_cons_df.loc[row_label], color='blue', label='NO3 Data')
            plt.plot(time_array, no3_fit(time_array), color='blue', label='NO3 Fit')
            plt.title(f'Polynomial Regression on {row_label}', fontsize=15)
            plt.xlabel('Time (hours)', fontsize=14)
            plt.ylabel('Consumption (mM)', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend()
            plt.show()

        return no2_fit, no3_fit

    def rates_for_first_and_second_half(
            self,
            row_label: str,
            no2_fit: np.poly1d,
            no3_fit: np.poly1d,
            epsilon: float = 0.05
    ):
        def _get_smallest_positive_real_root(poly, target_value):
            roots = (poly - target_value).roots
            positive_real_roots = roots[np.isreal(roots) & (np.real(roots) > 0)]
            return np.real(min(positive_real_roots)) if len(positive_real_roots) > 0 else 0
        
        no2_total_cons = self.no2_total_cons[row_label] - epsilon
        no3_total_cons = self.no3_total_cons[row_label] - epsilon

        no2_half_time = _get_smallest_positive_real_root(no2_fit, no2_total_cons / 2)
        no2_full_time = _get_smallest_positive_real_root(no2_fit, no2_total_cons)
        no3_half_time = _get_smallest_positive_real_root(no3_fit, no3_total_cons / 2)
        no3_full_time = _get_smallest_positive_real_root(no3_fit, no3_total_cons)

        print(f"  NO2 half-consumption time: {no2_half_time}")
        print(f"  NO2 full-consumption time: {no2_full_time}")
        print(f"  NO3 half-consumption time: {no3_half_time}")
        print(f"  NO3 full-consumption time: {no3_full_time}")

        no2_first_rate = (no2_total_cons / 2) / no2_half_time if no2_half_time > 0 else 0
        no2_second_rate = (no2_total_cons / 2) / (no2_full_time - no2_half_time) if no2_full_time > no2_half_time else 0
        no3_first_rate = (no3_total_cons / 2) / no3_half_time if no3_half_time > 0 else 0
        no3_second_rate = (no3_total_cons / 2) / (no3_full_time - no3_half_time) if no3_full_time > no3_half_time else 0

        return no2_first_rate, no2_second_rate, no3_first_rate, no3_second_rate

    def fit_for_selected_rows(
            self,
            row_labels: List[str],
            masks: Optional[Dict[str, List[int]]] = None,
            output_fn: Optional[str] = None,
            save_plot: bool = False
    ):
        regression_results = {}
        for row_label in row_labels:
            mask = None
            if masks is not None and row_label in masks:
                mask = masks[row_label]

            no2_fit, no3_fit = self.fit_for_row(row_label, mask)
            no2_first_rate, no2_second_rate, no3_first_rate, no3_second_rate = self.rates_for_first_and_second_half(row_label, no2_fit, no3_fit)
            regression_results[row_label] = {
                'no2_second_coef': no2_fit.coef[-3] if len(no2_fit.coef) > 2 else 0.0,
                'no2_first_coef': no2_fit.coef[-2] if len(no2_fit.coef) > 1 else 0.0,
                'no2_zeroth_coef': no2_fit.coef[-1],
                'no2_rate_first_half': no2_first_rate,
                'no2_rate_second_half': no2_second_rate,
                'no3_second_coef': no3_fit.coef[-3] if len(no3_fit.coef) > 2 else 0.0,
                'no3_first_coef': no3_fit.coef[-2] if len(no3_fit.coef) > 1 else 0.0,
                'no3_zeroth_coef': no3_fit.coef[-1],
                'no3_rate_first_half': no3_first_rate,
                'no3_rate_second_half': no3_second_rate}
            
        regression_results_df = pd.DataFrame.from_dict(regression_results, orient='index').astype(float)
        regression_results_df = regression_results_df.join(self.meta_df.loc[regression_results_df.index])

        if output_fn is not None:
            output_path = os.path.join(self.results_dir, f'{output_fn}.csv')
            regression_results_df.to_csv(output_path)
            print(f"Polynomial regression results saved to {output_path}")

        return regression_results_df
    
    def consumption_plot_for_selected_rows(
                self,
                row_labels: List[str],
                output_fn: str,
                regression_results_df: Optional[Dict] = None,
                show_plot: bool = False
        ):
            all_values = pd.concat([self.no3_cons_df.loc[row_labels], self.no2_cons_df.loc[row_labels]])
            y_min = all_values.min().min()
            y_max = all_values.max().max()
            num_rpl = 3
            num_col = 4
            num_row = int(np.ceil(len(row_labels) / num_col / num_rpl))
            fig, axes = plt.subplots(num_row, num_col, figsize=(5*num_row, 4*num_col))
            axes = axes.flatten()

            x = self.time.flatten()
            x_fit = np.linspace(min(x), max(x), 100)
            for i, row_label in enumerate(row_labels):
                no2_cons = self.no2_cons_df.loc[row_label].values.astype(float)
                no3_cons = self.no3_cons_df.loc[row_label].values.astype(float)

                ax = axes[i // num_rpl + 1]
                marker_styles = ['o', 's', '^']
                marker = marker_styles[i % num_rpl]
                
                ax.scatter(x, no2_cons, color='r', marker=marker)
                ax.scatter(x, no3_cons, color='b', marker=marker)

                if regression_results_df is None:
                    no2_fit, no3_fit = self.fit_for_row(row_label)
                    ax.plot(x_fit, no2_fit(x_fit), 'r-')
                    ax.plot(x_fit, no3_fit(x_fit), 'b-')
                else:
                    no2_fit = np.poly1d([regression_results_df.loc[row_label]['no2_second_coef'], regression_results_df.loc[row_label]['no2_first_coef'], regression_results_df.loc[row_label]['no2_zeroth_coef']])
                    no3_fit = np.poly1d([regression_results_df.loc[row_label]['no3_second_coef'], regression_results_df.loc[row_label]['no3_first_coef'], regression_results_df.loc[row_label]['no3_zeroth_coef']])
                    ax.plot(x_fit, no2_fit(x_fit), 'r-')
                    ax.plot(x_fit, no3_fit(x_fit), 'b-')

                #ax.set_xticks([0, 20, 40, 60, 80])
                #ax.tick_params(axis='x', labelsize=25)
                ax.set_ylim(y_min, y_max)
                #ax.set_yticks([0, 1, 2, 3])
                #ax.tick_params(axis='y', labelsize=25)
            fig.text(0.55, 0.05, 'Time (hours)', ha='center', fontsize=30)
            fig.text(0.145, 0.9, f'A(0) = 2.0 mM', fontsize=25)
            fig.text(0.35, 0.9, f'A(0) = 1.4 mM', fontsize=25)
            fig.text(0.55, 0.9, f'A(0) = 0.7 mM', fontsize=25)
            fig.text(0.75, 0.9, f'A(0) = 0.0 mM', fontsize=25)
            fig.text(0.08, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize=30)
            fig.text(0.91, 0.77, f'I(0) =\n2.0 mM', fontsize=25)
            fig.text(0.91, 0.575, f'I(0) =\n1.4 mM', fontsize=25)
            fig.text(0.91, 0.37, f'I(0) =\n0.7 mM', fontsize=25)
            fig.text(0.91, 0.165, f'I(0) =\n0.0 mM', fontsize=25)
            handles = [plt.Line2D([0], [0], color='b', marker='o', label=f'$NO_3$ Cosumption'),
                        plt.Line2D([0], [0], color='r', marker='o', label=f'$NO_2$ Consumption')]
            fig.legend(handles=handles, loc='upper right', fontsize=20)
            fig.suptitle(f'{id}, CHL-\nPolynomial Regression', fontsize=30, fontweight='bold')

            plt.savefig(f'plots/{output_fn}_polynomial_regression.png', dpi=300, bbox_inches='tight')
            print(f'Saved plots/{output_fn}_polynomial_regression.png')
            if show_plot:
                plt.show()
            plt.close()

if __name__ == "__main__":
    row_labels_chl1 = ['A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12']
    row_labels_chl0 = ['E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12']
    ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5', '4.2.batch6']
    for id in ids[:]:
        interpolator = Interpolator(id)

        chl0_half_rate_df = interpolator.half_rate_for_selected_rows(row_labels=row_labels_chl0)
        
        no3_conc_df = pd.read_csv(f'concentrations/{id}_no3_conc.csv', index_col=0)
        chl1_init_no3 = no3_conc_df.loc[row_labels_chl1, '0.0'].values.astype(float)
        chl0_init_no3 = no3_conc_df.loc[row_labels_chl0, '0.0'].values.astype(float)
        no2_conc_df = pd.read_csv(f'concentrations/{id}_no2_conc.csv', index_col=0)
        chl1_init_no2 = no2_conc_df.loc[row_labels_chl1, '0.0'].values.astype(float)
        chl0_init_no2 = no2_conc_df.loc[row_labels_chl0, '0.0'].values.astype(float)

        df_for_phase_diagram = pd.DataFrame()
        df_for_phase_diagram['chl0_init_no3'] = chl0_init_no3
        df_for_phase_diagram['chl0_init_no2'] = chl0_init_no2
        df_for_phase_diagram['chl0_no3_half_rate'] = chl0_half_rate_df['no3_half_rate'].values
        df_for_phase_diagram['chl0_no2_half_rate'] = chl0_half_rate_df['no2_half_rate'].values

        no3_thrs = 0.31
        no2_thrs = 0.1
        df_for_phase_diagram.loc[df_for_phase_diagram['chl0_init_no3'] < no3_thrs, 'chl0_no3_half_rate'] = np.nan
        df_for_phase_diagram.loc[(df_for_phase_diagram['chl0_init_no3'] < no3_thrs) & (df_for_phase_diagram['chl0_init_no2'] < no2_thrs), 'chl0_no2_half_rate'] = np.nan
        
        df_for_phase_diagram.to_csv(f'fitting_results/{id}_half_rate_for_phase_diagram.csv')
        print(f'Saved fitting_results/{id}_half_rate_for_phase_diagram.csv')