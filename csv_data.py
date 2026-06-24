import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ext_fig1_df = pd.read_csv('concentrations/lee_etal_ext_fig1.csv')
ext_fig1_chl_df = ext_fig1_df[ext_fig1_df['Chloramphenicol'] == 'CHL']
ext_fig1_chl_phase2_df = ext_fig1_chl_df[(ext_fig1_chl_df['Unit'] >= -50) & (ext_fig1_chl_df['Unit'] <= 10)]

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_ylabel(r'$\tilde{C}(t) = \frac{A(0) - A(t)}{A(0) - A(t_{last})}$')
ax.set_xlabel(r'$\frac{t}{t_{last}}$')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=1.0, label='Linear')

times = []
no3_conc = []
count = 0
for i in range(len(ext_fig1_chl_phase2_df)):
    no3 = ext_fig1_chl_phase2_df['NO3_mM'].iloc[i]
    time = ext_fig1_chl_phase2_df['Time_hours'].iloc[i]
    if time == 0 and i > 0:
        if no3_conc[-1] < no3_conc[0]:
            count += 1
            no3_norm_cons = [(no3_conc[0] - no3) / (no3_conc[0] - no3_conc[-1]) for no3 in no3_conc]
            norm_time = [t / times[-1] for t in times]
            ax.plot(norm_time, no3_norm_cons, color='blue', linestyle='-', alpha=0.05)
        times = []
        no3_conc = []
    times.append(time)
    no3_conc.append(no3)
print(f'Number of curves plotted: {count} out of {len(ext_fig1_chl_phase2_df) // 10}')
ax.set_title('Lee et al. Extended Figure 1 Source Data'
              + '\n' + f'drug conditions | phase 2 (-50 ~ +10) | {count} curves')
plt.savefig('plots/lee_etal_ext_fig1_chl_norm_cons.png', dpi=300)
