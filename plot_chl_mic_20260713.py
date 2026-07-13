import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

df = pd.read_csv('absorbances/20260711_Ik_OD600_96F_chl_no_lid_avg.csv', skiprows=7, usecols=[0, 2], names=['Well', 'OD600'], index_col=0)
chl_1 = np.array([1.843 * 323.13 / (2 ** i) for i in range(8)] + [0])
chl_2 = np.array([1.843 * 323.13 / (2 ** i) for i in range(8)] + [0])
chl_3 = np.array([1.786 * 323.13 / (2 ** i) for i in range(8)] + [0])

blank = np.mean([df.loc['C1':'C9', 'OD600'].values, df.loc['F1':'F9', 'OD600'].values])
print(blank)

strain_11 = df.loc['A1':'A9', 'OD600'].values - blank
strain_12 = df.loc['B1':'B9', 'OD600'].values - blank
strain_21 = df.loc['D1':'D9', 'OD600'].values - blank
strain_22 = df.loc['E1':'E9', 'OD600'].values - blank
strain_31 = df.loc['G1':'G9', 'OD600'].values - blank
strain_32 = df.loc['H1':'H9', 'OD600'].values - blank

fig, axes = plt.subplots(3, 1, figsize=(8, 6))
ax1 = axes[0]
ax1.plot(chl_1, strain_11, 'o-', color='blue', alpha=0.7)
ax1.plot(chl_1, strain_12, 'o-', color='orange', alpha=0.7)
ax1.set_title('Strain 1: Massilia')
ax1.set_ylabel('OD600')
ax1.set_xlim(0, 100)

ax2 = axes[1]
ax2.plot(chl_2, strain_21, 'o-', color='blue', alpha=0.7)
ax2.plot(chl_2, strain_22, 'o-', color='orange', alpha=0.7)
ax2.set_title('Strain 2: Chitinophaga')
ax2.set_ylabel('OD600')
ax2.set_xlim(0, 100)

ax3 = axes[2]
ax3.plot(chl_3, strain_31, 'o-', color='blue', alpha=0.7)
ax3.plot(chl_3, strain_32, 'o-', color='orange', alpha=0.7)
ax3.set_title('Strain 3: Paenibacillus (G7)')
ax3.set_xlabel('Chloramphenicol Concentration (µg/mL)')
ax3.set_ylabel('OD600')
ax3.set_xlim(0, 100)

plt.tight_layout()
plt.savefig(f'isolate_chl_mic_closeup_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
