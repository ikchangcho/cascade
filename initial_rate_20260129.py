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
            input_dir: str="concentrations",
            output_dir: str="fitting_results"
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.no2_data = pd.read_csv(f'{self.input_dir}/{exp_num}.{id}_no2_conc.csv', index_col=0)
        self.no3_data = pd.read_csv(f'{self.input_dir}/{exp_num}.{id}_no3_conc.csv', index_col=0)
        self.times = self.df.columns.values.astype(float)
        self.regression_results = {}
        if not os.path.exists(self.output_dir):
            raise ValueError(f"Output directory {self.output_dir} does not exist.")
        if self.no2_data is None or self.no2_data.empty:
            raise ValueError(f"NO2 data could not be loaded from {self.input_dir}/{exp_num}.{id}_no2_conc.csv or is empty.")
        if self.no3_data is None or self.no3_data.empty:
            raise ValueError(f"NO3 data could not be loaded from {self.input_dir}/{exp_num}.{id}_no3_conc.csv or is empty.")

    def _get_time_and_values_for_row(
            self,
            row_label: str,
            column_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if row_label not in self.df.index:
            raise ValueError(f"Row {row_label} not found in dataframe index.")

        x = self.times
        y = self.df.loc[row_label].values.astype(float) 
        if column_indices is not None:
            if any(idx < 0 or idx >= len(x) for idx in column_indices):
                raise ValueError("One or more indices in column_indices are out of bounds.")
            x = x[column_indices]
            y = y[column_indices]

        return x, y
    
    def fit_for_row(
            self,
            row_label: str,
            column_indices: Optional[List[int]] = None,
            show_plot: bool = False
    ) -> Tuple[float, float]:
        x, y = self._get_time_and_values_for_row(row_label, column_indices)
        
        x = x.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        self.regression_results[row_label] = (slope, intercept)
    
        if show_plot:
            plt.scatter(x, y, color='blue', label='Data Points')
            plt.plot(x, model.predict(x), color='red', label='Fitted Line')
            plt.title(f'Linear Regression for {row_label}')
            plt.xlabel('Time (hours)')
            plt.legend()
            plt.show()
        
        return slope, intercept
    
    def fit_for_data(
            self,
            rows_and_columns: Optional[Dict[str, List[int]]] = None,
            filename: Optional[str] = None
    ) -> Dict[str, Tuple[float, float]]:                
        for row, column_indices in rows_and_columns.items():
            slope, intercept = self.fit_for_row(row, column_indices)
            self.regression_results[row] = (slope, intercept)
                
        if filename is not None:
            output_path = os.path.join(self.output_dir, filename)
            results_df = pd.DataFrame.from_dict(
                self.regression_results, 
                orient='index', 
                columns=['slope', 'intercept']
            )
            results_df.to_csv(output_path)
                
        return self.regression_results
    
rows_and_columns_batch1 = = {
        'A04': [0, 1, ],
    }



if __name__ == "__main__":
    regressor = LinearRegressor("concentrations/4.2.batch5_no3_conc.csv")
    rows_and_columns =
    results = regressor.fit_for_data(rows_and_columns)
    #result = regressor.fit_for_row('A01', [0, 1, 3, 4], show_plot=True)
    
        