import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns

df = pd.read_csv('fitting_results/model3_fitting_results_20260517.csv')

# r_A, Gamma_A : Exclude init_no3 = 0 conditions
A_mask = df['init_no3'] > 0.3

# r_I, Gamma_I : Exclude init_no3 = 0 and init_no2 = 0 conditions
I_mask = ((df['init_no3'] > 0.3) | (df['init_no2'] > 0.3))


for (x_label, y_label) in [('r_A', 'r_I'), ('r_A', 'Gamma_A'), ('r_A', 'Gamma_I'), ('Gamma_A', 'r_I'), ('Gamma_A', 'Gamma_I'), ('r_I', 'Gamma_I')]:
    if x_label == 'r_A' or x_label == 'Gamma_A':
        df = df[A_mask]
    else:
        df = df[I_mask]

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

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['red', 'darkorange', 'green', 'blue', 'purple', 'black']
    for i in range(len(df)):
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
    fig.legend(handles=handles, loc='upper right')
    fig.suptitle(f'{y_label} vs {x_label}', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'plots/{y_label}_vs_{x_label}_20260518.png')
    plt.close()




# r_A_mean = []
# r_A_std = []
# r_I_mean = []
# r_I_std = []
# Gamma_A_mean = []
# Gamma_A_std = []
# Gamma_I_mean = []
# Gamma_I_std = []

# ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3', '4.2.batch4', '4.2.batch5', '4.2.batch6']
# for id in ids:
#     r_A_values = df[(df['id'] == id) & A_mask]['r_A'].values
#     r_I_values = df[(df['id'] == id) & I_mask]['r_I'].values
#     Gamma_A_values = df[(df['id'] == id) & Gamma_A_mask]['Gamma_A'].values
#     Gamma_I_values = df[(df['id'] == id) & Gamma_I_mask]['Gamma_I'].values

#     r_A_mean.append(np.mean(r_A_values))
#     r_A_std.append(np.std(r_A_values))
#     r_I_mean.append(np.mean(r_I_values))
#     r_I_std.append(np.std(r_I_values))
#     Gamma_A_mean.append(np.mean(Gamma_A_values))
#     Gamma_A_std.append(np.std(Gamma_A_values))
#     Gamma_I_mean.append(np.mean(Gamma_I_values))
#     Gamma_I_std.append(np.std(Gamma_I_values))

# # Bar plots
# labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5', 'Batch 6']
# x = np.arange(len(labels))
# width = 0.35

# fig, ax = plt.subplots(1, 1, figsize=(12, 10), squeeze=True)
# ax.bar(x - width/2, Gamma_A_mean, yerr=Gamma_A_std, width=width, label=r'$\Gamma_A$', capsize=5, color='blue')
# ax.bar(x + width/2, Gamma_I_mean, yerr=Gamma_I_std, width=width, label=r'$\Gamma_I$', capsize=5, color='red')
# ax.set_ylim(0, None)
# ax.set_xticks(x, labels)
# ax.set_title(r'Mean Values of $\Gamma_A$ and $\Gamma_I$ (/hour)', fontsize = 15)
# ax.legend()
# plt.savefig('plots/Gamma_A_Gamma_I_mean_values_20260518.png')
# plt.close()

# fig, ax = plt.subplots(1, 1, figsize=(12, 10), squeeze=True)
# ax.bar(x - width/2, r_A_mean, yerr=r_A_std, width=width, label=r'$r_A$', capsize=5, color='blue')
# ax.bar(x + width/2, r_I_mean, yerr=r_I_std, width=width, label=r'$r_I$', capsize=5, color='red')
# ax.set_ylim(0, None)
# ax.set_xticks(x, labels)
# ax.set_title(r'Mean Values of $r_A$ and $r_I$ (mM/hour)', fontsize = 15)
# ax.legend()
# plt.savefig('plots/r_A_r_I_mean_values_20260518.png')
# plt.close()
