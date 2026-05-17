import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns

df = pd.read_csv('fitting_results/model3_fitting_results_20260517.csv')
x_label = 'init_no3'
y_label = 'Gamma_A'

x_values = df[x_label].values
x_values = np.sort(x_values)
x_range = x_values[-5] - x_values[5]
x_min = x_values[5] - 0.1 * x_range
x_max = x_values[-5] + 0.1 * x_range

y_values = df[y_label].values
y_values = np.sort(y_values)
y_range = y_values[-5] - y_values[5]
y_min = y_values[5] - 0.1 * y_range
y_max = y_values[-5] + 0.1 * y_range

plt.plot(df[x_label], df[y_label], 'o', alpha=0.5)
plt.xlabel(x_label)
plt.xlim(x_min, x_max)
plt.ylabel(y_label)
plt.ylim(y_min, y_max)
plt.show()