import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy.polynomial import polynomial as P
import pickle
import seaborn as sns

class LinearRegressor:
    def __init__(
            self,
            id: str,
            meta_col_num: int=4,
            input_dir: str="concentrations",
            results_dir: str="fitting_results",
            conc: bool=True
    ):
        print(f"Linear Regression on Exp {id} data")
        self.input_dir = input_dir
        self.results_dir = results_dir
        if conc:
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
            output_path = os.path.join(self.results_dir, filename)
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

            plt.savefig(f'plots/{output_fn}.png', dpi=300, bbox_inches='tight')
            print(f'Saved plots/{output_fn}.png')
            if show_plot:
                plt.show()
            plt.close()


    def heatmaps_for_selected_columns(
            self,
            input_fn: str,
            col_labels: List[str],
            output_fn: str,
            conv_factor: float = 24.0,
            show_plot: bool = False
    ):
        file_path = os.path.join(self.results_dir, f'{input_fn}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        regression_results_df = pd.read_csv(file_path, index_col=0)
        regression_results_df.iloc[:, :-self.meta_col_num] = regression_results_df.iloc[:, :-self.meta_col_num] * conv_factor    # Convert rates to mM/day
        
        # Group by and calculate mean/std
        groupby_cols = ['Nitrite_input', 'Nitrate_input', 'Chloramphenicol']
        drop_cols = ['Sample_type']
        regression_results_mean = regression_results_df.drop(drop_cols, axis=1).groupby(groupby_cols, as_index=False).mean()
        regression_results_var = regression_results_df.drop(drop_cols, axis=1).groupby(groupby_cols, as_index=False).var()
        mean_and_var = regression_results_mean.merge(regression_results_var, on=groupby_cols, suffixes=('_mean', '_var'))
        mean_and_var = mean_and_var.sort_values(['Chloramphenicol', 'Nitrate_input', 'Nitrite_input'], ascending=False)
        mean_and_var.to_csv(f"{self.results_dir}/{output_fn}_polynomial_regression_mean_var.csv", index=False)

        # Function to create pivot table
        def create_pivot_table(df, chl, column_name):
            return df.query(f'Chloramphenicol == {chl}').pivot(
                index='Nitrite_input', 
                columns='Nitrate_input', 
                values=column_name
            ).sort_index(ascending=False).sort_index(axis=1, ascending=False)

        for col_label in col_labels:
            if col_label not in regression_results_mean.columns:
                raise ValueError(f"Column {col_label} not found in regression results.")
            pivot_table_mean = create_pivot_table(regression_results_mean, 0, col_label)
            pivot_table_std = create_pivot_table(regression_results_var, 0, col_label).pow(0.5)
            pivot_table_annot = pivot_table_mean.round(2).astype(str) + "\n±" + pivot_table_std.round(2).astype(str)

            sns.heatmap(pivot_table_mean, annot=pivot_table_annot, fmt='', cmap='binary')
            plt.title(f'{col_label} ({id} CHL-)', fontsize=16)
            plt.xlabel('Nitrate Input (mM)', fontsize=14)
            plt.ylabel('Nitrite Input (mM)', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.savefig(f'{self.plots_dir}/{output_fn}_{col_label}_heatmap.png', dpi=300, bbox_inches='tight')    
            print(f'Saved {self.plots_dir}/{output_fn}_{col_label}_heatmap.png')
            if show_plot:
                plt.show()
            plt.close()


if __name__ == "__main__":
    ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5']
    for id in ids[0:5]:
        #time_threshold=20
        # regressor = LinearRegressor(id)
        # result = regressor.fit_for_row('E04', [0, 1, 2, 3], show_plot=True)
        # results = regressor.fit_for_entire_data(time_threshold, filename=f'{id}_linear_regression_results.csv')
    
        regressor = PolynomialRegressor(id)
        #regressor.fit_for_row('F01', show_plot=True)
        rows_chl0 = ['E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12']
        masks = None
        if id == '4.2.batch3':
            masks = {'E12': [9], 'F08': [9], 'H09': [9], 'H10': [9]}
        regression_results_df = regressor.fit_for_selected_rows(row_labels=rows_chl0, masks=masks, output_fn=f'{id}.chl0_polynomial_regression_results')
        # regressor.consumption_plot_for_selected_rows(row_labels=rows_chl0, regression_results_df=regression_results_df, output_fn=f'{id}.chl0_polynomial_regression_plots')
        

        regressor.heatmaps_for_selected_columns(
            input_fn=f'{id}.chl0_polynomial_regression_results',
            col_labels=['no2_second_coef', 'no3_second_coef'],
            output_fn=f'{id}.chl0',
            conv_factor=1000,
            show_plot=False
        )