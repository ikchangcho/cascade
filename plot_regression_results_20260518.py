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

r_A_mean = []
r_A_std = []
r_I_mean = []
r_I_std = []
Gamma_A_mean = []
Gamma_A_std = []
Gamma_I_mean = []
Gamma_I_std = []

ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5', '4.2.batch6']
for id in ids:
    r_A_values = df[df['id'] == id]['r_A'].values
    r_I_values = df[df['id'] == id]['r_I'].values
    Gamma_A_values = df[df['id'] == id]['Gamma_A'].values
    Gamma_I_values = df[df['id'] == id]['Gamma_I'].values

    r_A_mean.append(np.mean(r_A_values))
    r_A_std.append(np.std(r_A_values))
    r_I_mean.append(np.mean(r_I_values))
    r_I_std.append(np.std(r_I_values))
    Gamma_A_mean.append(np.mean(Gamma_A_values))
    Gamma_A_std.append(np.std(Gamma_A_values))
    Gamma_I_mean.append(np.mean(Gamma_I_values))
    Gamma_I_std.append(np.std(Gamma_I_values))

fig, ax = plt.subplots(1, 1, figsize=(12, 10), squeeze=True)
labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width/2, Gamma_A_mean, yerr=Gamma_A_std, width=width, label=r'$\Gamma_A$', capsize=5, color='blue')
ax.bar(x + width/2, Gamma_I_mean, yerr=Gamma_I_std, width=width, label=r'$\Gamma_I$', capsize=5, color='red')
ax.set_ylim(0, None)
ax.set_xticks(x, labels)
ax.set_title(r'Mean Values of $\Gamma_A$ and $\Gamma_I$ (/hour)', fontsize = 15)
ax.legend()
plt.savefig('plots/Gamma_A_Gamma_I_mean_values_20260518.png')
