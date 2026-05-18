import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns

df = pd.read_csv('fitting_results/model3_fitting_results_20260517.csv')
x_labels = ['init_no3', 'init_no2']
y_labels = ['r_A', 'r_I', 'Gamma_A', 'Gamma_I']

# r_A, Gamma_A : Exclude init_no3 = 0 conditions
# r_I, Gamma_I : Exclude init_no3 = 0 and init_no2 = 0 conditions

for x_label in x_labels:
    for y_label in y_labels:
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

        colors = ['red', 'darkorange', 'green', 'blue', 'purple', 'black']
        fig, ax = plt.subplots(figsize=(8, 6))

        for i in range(len(df)):
            if y_label == 'r_A' or y_label == 'Gamma_A':
                if x_label == 'init_no3':
                    x_min = 0.25
                if df.iloc[i].loc['init_no3'] < 0.3:
                    continue
            if y_label == 'r_I' or y_label == 'Gamma_I':
                if df.iloc[i].loc['init_no3'] < 0.3 and df.iloc[i].loc['init_no2'] < 0.3:
                    continue
            batch_id = df.iloc[i].loc['id']
            batch_num = int(batch_id.split('.')[-1].replace('batch', '')) - 1
            color = colors[batch_num]

            ax.scatter(df.iloc[i].loc[x_label], df.iloc[i].loc[y_label], color=color, alpha=0.5)
            ax.set_xlabel(x_label)
            ax.set_xlim(x_min, x_max)
            ax.set_ylabel(y_label)
            ax.set_ylim(y_min, y_max)
        handles = [plt.Line2D([0], [0], color='red', marker='o', label=f'batch 1'),
                        plt.Line2D([0], [0], color='darkorange', marker='o', label=f'batch 2'),
                        plt.Line2D([0], [0], color='green', marker='o', label=f'batch 3'),
                        plt.Line2D([0], [0], color='blue', marker='o', label=f'batch 4'),
                        plt.Line2D([0], [0], color='purple', marker='o', label=f'batch 5'),
                        plt.Line2D([0], [0], color='black', marker='o', label=f'batch 6')]
        ax.legend(handles=handles, loc='upper right')
        plt.savefig(f'plots/{y_label}_vs_{x_label}_20260517.png')
        print(f'Plot saved: plots/{y_label}_vs_{x_label}_20260517.png')