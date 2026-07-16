import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

df1 = pd.read_csv('absorbances/20260716_Ik_OD600_96F_bacteroidota_123_no_lid.csv', skiprows=8, index_col=0)
df1 = df1.iloc[:, :-1]
df2 = pd.read_csv('absorbances/20260716_Ik_OD600_96F_bacteroidota_456_no_lid.csv', skiprows=8, index_col=0)
df2 = df2.iloc[:, :-1]

chl_1 = np.array([4 * 100 / 237 * 323.13 / (2 ** i) for i in range(11)] + [0])
chl_2 = np.array([4 * 100 / 220 * 323.13 / (2 ** i) for i in range(11)] + [0])
chl_3 = np.array([4 * 100 / 220 * 323.13 / (2 ** i) for i in range(11)] + [0])
chl_4 = np.array([4 * 100 / 223 * 323.13 / (2 ** i) for i in range(11)] + [0])
chl_5 = np.array([4 * 100 / 220 * 323.13 / (2 ** i) for i in range(11)] + [0])
chl_6 = np.array([4 * 100 / 246 * 323.13 / (2 ** i) for i in range(11)] + [0])

blank_mean_1 = np.mean([df1.loc['C'].values, df1.loc['F'].values])
blank_std_1 = np.std([df1.loc['C'].values, df1.loc['F'].values])
blank_mean_2 = np.mean([df2.loc['C'].values, df2.loc['F'].values])
blank_std_2 = np.std([df2.loc['C'].values, df2.loc['F'].values])

strain_11 = df1.loc['A'].values - blank_mean_1
strain_12 = df1.loc['B'].values - blank_mean_1
strain_21 = df1.loc['D'].values - blank_mean_1
strain_22 = df1.loc['E'].values - blank_mean_1
strain_31 = df1.loc['G'].values - blank_mean_1
strain_32 = df1.loc['H'].values - blank_mean_1
strain_41 = df2.loc['A'].values - blank_mean_2
strain_42 = df2.loc['B'].values - blank_mean_2
strain_51 = df2.loc['D'].values - blank_mean_2
strain_52 = df2.loc['E'].values - blank_mean_2
strain_61 = df2.loc['G'].values - blank_mean_2
strain_62 = df2.loc['H'].values - blank_mean_2

fig, axes = plt.subplots(3, 2, figsize=(8, 8))
fig.suptitle('MIC Curves for Bacteroidota Isolates', fontsize=16)

ax1 = axes[0, 0]
ax1.plot(chl_1, strain_11, 'o-', color='blue', alpha=0.7)
ax1.plot(chl_1, strain_12, 'o-', color='orange', alpha=0.7)
# ax1.errorbar(0, blank_mean_1, yerr=blank_std_1, fmt='.', color='black', alpha=0.7, label='Blank')
ax1.set_title('Strain 1: Chitinophaga')
ax1.set_ylabel('OD600')

ax2 = axes[1, 0]
ax2.plot(chl_2, strain_21, 'o-', color='blue', alpha=0.7)
ax2.plot(chl_2, strain_22, 'o-', color='orange', alpha=0.7)
ax2.set_title('Strain 2: Chryseobacterium (I8)')
ax2.set_ylabel('OD600')

ax3 = axes[2, 0]
ax3.plot(chl_3, strain_31, 'o-', color='blue', alpha=0.7)
ax3.plot(chl_3, strain_32, 'o-', color='orange', alpha=0.7)
ax3.set_title('Strain 3: Chryseobacterium (A2)')
ax3.set_xlabel('Chloramphenicol (µg/mL)')
ax3.set_ylabel('OD600')

ax4 = axes[0, 1]
ax4.plot(chl_4, strain_41, 'o-', color='blue', alpha=0.7)
ax4.plot(chl_4, strain_42, 'o-', color='orange', alpha=0.7)
ax4.set_title('Strain 4: Pedobacter')

ax5 = axes[1, 1]
ax5.plot(chl_5, strain_51, 'o-', color='blue', alpha=0.7)
ax5.plot(chl_5, strain_52, 'o-', color='orange', alpha=0.7)
ax5.set_title('Strain 5: Chryseobacterium (C8)')

ax6 = axes[2, 1]
ax6.plot(chl_6, strain_61, 'o-', color='blue', alpha=0.7)
ax6.plot(chl_6, strain_62, 'o-', color='orange', alpha=0.7)
ax6.set_title('Strain 6: Dyadobacter')

plt.tight_layout()

plt.savefig(f'plots/bacteroidota_mic_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
# for ax in axes.flatten():
#     ax.set_xlim(0, 50)
# plt.savefig(f'plots/bacteroidota_mic_close_up_{datetime.datetime.now().strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
