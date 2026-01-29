import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class LinearRegressor:
    def __init__(
            self,
            filename: str,
            output_dir: str="fitting_results"
    ):
        self.filename = filename
        self.output_dir = output_dir
        self.df = pd.read_csv(self.filename, index_col=0)
        self.times = self.df.columns.values.astype(float)
        self.regression_results = {}
        if not os.path.exists(self.output_dir):
            raise ValueError(f"Output directory {self.output_dir} does not exist.")
        if self.df is None or self.df.empty:
            raise ValueError(f"Dataframe could not be loaded from {self.filename} or is empty.")

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
    
    def fit_regression_for_row(
            self,
            row_label: str,
            column_indices: Optional[List[int]] = None
    ) -> Tuple[float, float]:
        x, y = self._get_time_and_values_for_row(row_label, column_indices)
        
        x = x.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        self.regression_results[row_label] = (slope, intercept)
        
        return slope, intercept
    
if __name__ == "__main__":
    regressor = LinearRegressor("concentrations/4.2.batch5_no3_conc.csv")
    slope, intercept = regressor.fit_regression_for_row("A01", column_indices=[0, 1, 2, 3])
    print(f"Slope: {slope}, Intercept: {intercept}")
        