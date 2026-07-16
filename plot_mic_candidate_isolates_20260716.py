import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

df = pd.read_csv('absorbances/20260711_Ik_OD600_96F_chl_no_lid_avg.csv', skiprows=7, usecols=[0, 2], names=['Well', 'OD600'], index_col=0)
chl_11 = np.array([1.843 * 323.13 / (2 ** i) for i in range(8)] + [0])
chl_13 = np.array([1.786 * 323.13 / (2 ** i) for i in range(8)] + [0])

blank = np.mean([df.loc['C1':'C9', 'OD600'].values, df.loc['F1':'F9', 'OD600'].values])

strain_11 = df.loc['A1':'A9', 'OD600'].values - blank
strain_12 = df.loc['B1':'B9', 'OD600'].values - blank
strain_31 = df.loc['G1':'G9', 'OD600'].values - blank
strain_32 = df.loc['H1':'H9', 'OD600'].values - blank

df1 = pd.read_csv('absorbances/20260716_Ik_OD600_96F_bacteroidota_123_no_lid.csv', skiprows=8, index_col=0)
df1 = df1.iloc[:, :-1]
df2 = pd.read_csv('absorbances/20260716_Ik_OD600_96F_bacteroidota_456_no_lid.csv', skiprows=8, index_col=0)
df2 = df2.iloc[:, :-1]

chl_2 = np.array([4 * 100 / 220 * 323.13 / (2 ** i) for i in range(11)] + [0])
chl_5 = np.array([4 * 100 / 220 * 323.13 / (2 ** i) for i in range(11)] + [0])

blank_mean_1 = np.mean([df1.loc['C'].values, df1.loc['F'].values])
blank_std_1 = np.std([df1.loc['C'].values, df1.loc['F'].values])
blank_mean_2 = np.mean([df2.loc['C'].values, df2.loc['F'].values])
blank_std_2 = np.std([df2.loc['C'].values, df2.loc['F'].values])

strain_21 = df1.loc['D'].values - blank_mean_1
strain_22 = df1.loc['E'].values - blank_mean_1
strain_51 = df2.loc['D'].values - blank_mean_2
strain_52 = df2.loc['E'].values - blank_mean_2

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
fig.suptitle('MIC Curves for Candidate Isolates', fontsize=16)
for ax in axes.flatten():
    ax.set_xlim(0, 50)

ax1 = axes[0, 0]
ax1.plot(chl_11, strain_11, 'o-', color='blue', alpha=0.7)
ax1.plot(chl_11, strain_12, 'o-', color='orange', alpha=0.7)
ax1.set_title('Massilia, Pseudomonadota')
ax1.set_ylabel('OD600')

ax2 = axes[0, 1]
ax2.plot(chl_13, strain_31, 'o-', color='blue', alpha=0.7)
ax2.plot(chl_13, strain_32, 'o-', color='orange', alpha=0.7)
ax2.set_title('Paenibacillus (G7), Bacillota')

ax3 = axes[1, 0]
ax3.plot(chl_2, strain_21, 'o-', color='blue', alpha=0.7)
ax3.plot(chl_2, strain_22, 'o-', color='orange', alpha=0.7)
ax3.set_title('Chryseobacterium (I8), Bacteroidota')
ax3.set_ylabel('OD600')
ax3.set_xlabel('Chloramphenicol Concentration (µg/mL)')

ax4 = axes[1, 1]
ax4.plot(chl_5, strain_51, 'o-', color='blue', alpha=0.7)
ax4.plot(chl_5, strain_52, 'o-', color='orange', alpha=0.7)
ax4.set_title('Chryseobacterium (C8), Bacteroidota')
ax4.set_xlabel('Chloramphenicol Concentration (µg/mL)')

plt.tight_layout()
plt.savefig(f'plots/candidate_isolates_mic_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
