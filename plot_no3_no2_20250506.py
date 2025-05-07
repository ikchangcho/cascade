import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
date = 20250428
key = ''
meta = pd.read_csv(f'absorbances/{date}_samples_metadata.csv', index_col=0).dropna(how='all')
no3_conc = pd.read_csv(f'concentrations/{date}_{key}_no3_conc.csv', index_col=0)
no2_conc = pd.read_csv(f'concentrations/{date}_{key}_no2_conc.csv', index_col=0)
no3_cons = pd.read_csv(f'concentrations/{date}_{key}_no3_cons.csv', index_col=0)
no2_cons = pd.read_csv(f'concentrations/{date}_{key}_no2_cons.csv', index_col=0)
times = no3_conc.columns.astype(float).tolist()

# # scatter plot, no3_no2_cons vs time, one condition, chl0 / chl1
# row = 'A01'
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.plot(times_chl0, no3_chl0_cons.loc[row], 'bo', label='$-\Delta A$')
# ax.plot(times_chl0, no2_chl0_cons.loc[row], 'ro', label='$-\Delta I -\Delta A$')
# ax.set_xlabel('Time (hours)', fontsize=15)
# ax.set_ylabel('Concentration (mM)', fontsize=15)
# ax.tick_params(axis='x', labelsize=15)
# ax.tick_params(axis='y', labelsize=15)
# ax.legend(fontsize=15)
# filename = f'scatter_no3_no2_chl0_cons_{row}.png'
# plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
# print(f'Saved {filename}')
# plt.close()

# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.plot(times_chl1, no3_chl1_cons.loc[row], 'bo', label='$-\Delta A$')
# ax.plot(times_chl1, no2_chl1_cons.loc[row], 'ro', label='$-\Delta I -\Delta A$')
# ax.set_xlabel('Time (hours)', fontsize=15)
# ax.set_ylabel('Concentration (mM)', fontsize=15)
# ax.tick_params(axis='x', labelsize=15)
# ax.tick_params(axis='y', labelsize=15)
# ax.legend(fontsize=15)
# filename = f'scatter_no3_no2_chl1_cons_{row}.png'
# plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
# print(f'Saved {filename}')
# plt.close()



# # Compare previous CHL result and new one
# rows_old = ['A01', 'A02', 'A03', 'B01', 'B02', 'B03', 'D01', 'D02', 'D03', 'D07', 'D08', 'D09']
# rows_new = ['A05', 'A06', 'B05', 'B06', 'C05', 'C06', 'D05', 'D06']
# titles = ['A(0) = 2.0 mM, I(0) = 2.0 mM', 'A(0) = 2.0 mM, I(0) = 0.0 mM', 'A(0) = 1.0 mM, I(0) = 1.0 mM', 'A(0) = 1.0 mM, I(0) = 0.0 mM']

# for idx in np.arange(0, len(rows_old) // 3):
#     fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#     for row in rows_old[idx*3:(idx+1)*3]:
#         ax.plot(times_chl1[:5], no3_chl1_cons.loc[row][:5], 'b.-')
#         ax.plot(times_chl1[:5], no2_chl1_cons.loc[row][:5], 'r.-')
#     for row in rows_new[idx*2:(idx+1)*2]:
#         ax.plot(times_anti, no3_anti_cons.loc[row], 'bo--', alpha=0.5)
#         ax.plot(times_anti, no2_anti_cons.loc[row], 'ro--', alpha=0.5)
#     ax.set_xlabel('Time (hours)', fontsize=15)
#     ax.set_ylabel('Concentration (mM)', fontsize=15)
#     ax.tick_params(axis='x', labelsize=15)
#     ax.tick_params(axis='y', labelsize=15)
#     handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'old $-\Delta A$'),
#             plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'old $\Delta I -\Delta A$'),
#             plt.Line2D([0], [0], color='b', marker='o', linestyle='--', label=f'new $-\Delta A$'),
#             plt.Line2D([0], [0], color='r', marker='o', linestyle='--', label=f'new $\Delta I -\Delta A$')]
#     ax.legend(handles=handles, loc='upper right', fontsize=12)
#     ax.set_title(titles[idx], fontsize=15)
#     #plt.show()
#     filename = f'no3_no2_chl1_cons_old_new_{idx}.png'
#     plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
#     print(f'Saved {filename}')


# # no3_no2_chl1_cons vs time, one condition
# rows = ['C07', 'C08', 'C09']
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# for row in rows:
#     ax.plot(times_chl1[:], no3_chl1_cons.loc[row][:], 'b.-')
#     ax.plot(times_chl1[:], no2_chl1_cons.loc[row][:], 'r.-')
# ax.set_xlabel('Time (hours)', fontsize=20)
# ax.set_ylabel('Concentration (mM)', fontsize=20)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'$-\Delta A$'),
#             plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'$-\Delta I -\Delta A$')]
# ax.legend(handles=handles, fontsize=15)
# ax.set_title('A(0) = 1 mM, I(0) = 2 mM, drug', fontsize=20)
# filename = f'no3_no2_chl1_cons_(1,2).png'
# plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
# print(f'Saved plots/{filename}')

# # no3_no2_chl0_cons vs time, all conditions withour carbon addition
# rows = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06']
# all_values = pd.concat([no3_chl0_cons.loc[rows], no2_chl0_cons.loc[rows]])
# y_min = all_values.min().min()
# y_max = all_values.max().max()

# num_rpl = 3
# num_col = 5
# num_row = int(np.ceil(len(rows) / num_col / num_rpl))
# fig, axes = plt.subplots(num_row, num_col, figsize=(4*num_row, 3*num_col))
# axes = axes.flatten()
# for i ,row in enumerate(rows):
#     ax = axes[i//num_rpl]
#     ax.plot(times_chl0[:-3], no3_chl0_cons.loc[row][:-3], 'b.-')
#     ax.plot(times_chl0[:-3], no2_chl0_cons.loc[row][:-3], 'r.-')
#     ax.set_xticks([0, 20, 40])
#     ax.tick_params(axis='x', labelsize=25)
#     ax.set_ylim(y_min, y_max)
#     ax.set_yticks([0, 1, 2, 3, 4])
#     ax.tick_params(axis='y', labelsize=25)
# fig.text(0.55, 0.05, 'Time (hours)', ha='center', fontsize=30)
# fig.text(0.14, 0.9, f'I(0) = 2.0 mM', fontsize=25)
# fig.text(0.29, 0.9, f'I(0) = 1.5 mM', fontsize=25)
# fig.text(0.45, 0.9, f'I(0) = 1.0 mM', fontsize=25)
# fig.text(0.61, 0.9, f'I(0) = 0.5 mM', fontsize=25)
# fig.text(0.77, 0.9, f'I(0) = 0.0 mM', fontsize=25)
# fig.text(0.08, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize=30)
# fig.text(0.91, 0.80, f'A(0) = 2.0 mM', fontsize=25)
# fig.text(0.91, 0.65, f'A(0) = 1.5 mM', fontsize=25)
# fig.text(0.91, 0.48, f'A(0) = 1.0 mM', fontsize=25)
# fig.text(0.91, 0.32, f'A(0) = 0.5 mM', fontsize=25)
# fig.text(0.91, 0.16, f'A(0) = 0.0 mM', fontsize=25)
# #fig.suptitle('Title', fontsize=20, y=0.95)    
# handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'$-\Delta A$'),
#             plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'$-\Delta I -\Delta A$')]
# fig.legend(handles=handles, loc='upper right', fontsize=20)
# filename = 'no3_no2_chl0_cons_all.png'
# plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
# print(f'Saved plots/{filename}')

# # no3_no2_nh4_chl1 vs time, carbon addition
# rows_carbon = ['G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'D10', 'D11', 'D12']
# rows_neg = ['A01', 'A02', 'A03', 'B01', 'B02', 'B03', 'F04', 'F05', 'F06', 'F10', 'F11', 'F12', 'G04', 'G05', 'G06']
# num_rpl = 3

# for idx in np.arange(0, len(rows_carbon) // num_rpl):
#     fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#     for row in rows_carbon[idx*num_rpl:(idx+1)*num_rpl]:
#         ax.plot(times_chl1, no3_chl1_evap.loc[row], 'b.-')
#         ax.plot(times_chl1, no2_chl1_evap.loc[row], 'r.-')
#         ax.plot(times_chl1, nh4_chl1_evap.loc[row], 'g.-')
#     for row in rows_neg[idx*num_rpl:(idx+1)*num_rpl]:
#         ax.plot(times_chl1, no3_chl1_evap.loc[row], 'bo--', alpha=0.5)
#         ax.plot(times_chl1, no2_chl1_evap.loc[row], 'ro--', alpha=0.5)
#         ax.plot(times_chl1, nh4_chl1_evap.loc[row], 'go--', alpha=0.5)
#     ax.set_xlabel('Time (hours)', fontsize=15)
#     ax.set_ylabel('Concentration (mM)', fontsize=15)
#     ax.tick_params(axis='x', labelsize=15)
#     ax.tick_params(axis='y', labelsize=15)
#     handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'$NO_3$ (A)'),
#             plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'$NO_2$ (I)'),
#             plt.Line2D([0], [0], color='g', marker='.', linestyle='-', label=f'$NH_4$')]
#     ax.legend(handles=handles, loc='upper right', fontsize=12)
#     #plt.show()
#     filename = f'no3_no2_nh4_chl1_carbon{idx}.png'
#     plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
#     print(f'Saved {filename}')


# no3_no2_conc vs time, all conditions
rows_chl0 = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09']
rows_chl1 = ['E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'H09']
all_values_chl0 = pd.concat([no3_conc.loc[rows_chl0], no2_conc.loc[rows_chl0]])
all_values_chl1 = pd.concat([no3_conc.loc[rows_chl1], no2_conc.loc[rows_chl1]])
y_min_chl0 = all_values_chl0.min().min()
y_min_chl1 = all_values_chl1.min().min()
y_max_chl0 = all_values_chl0.max().max()
y_max_chl1 = all_values_chl1.max().max()

# No drug data
num_rpl = 3
num_col = 5
num_row = int(np.ceil(len(rows_chl0) / num_col / num_rpl))
fig, axes = plt.subplots(num_row, num_col, figsize=(8*num_row, 3*num_col))
axes = axes.flatten()
for i ,row in enumerate(rows_chl0):
    ax = axes[i//num_rpl]
    ax.plot(times, no3_conc.loc[row], 'b.-')
    ax.plot(times, no2_conc.loc[row], 'r.-')
    ax.set_xticks([0, 20, 40])
    ax.tick_params(axis='x', labelsize=25)
    ax.set_ylim(y_min_chl0, y_max_chl0)
    ax.set_yticks([0, 1, 2, 3])
    ax.tick_params(axis='y', labelsize=25)
fig.text(0.55, 0.05, 'Time (hours)', ha='center', fontsize=30)
fig.text(0.14, 0.9, f'I(0) = 2.0 mM', fontsize=25)
fig.text(0.29, 0.9, f'I(0) = 1.5 mM', fontsize=25)
fig.text(0.45, 0.9, f'I(0) = 1.0 mM', fontsize=25)
fig.text(0.61, 0.9, f'I(0) = 0.5 mM', fontsize=25)
fig.text(0.77, 0.9, f'I(0) = 0.0 mM', fontsize=25)
fig.text(0.08, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize=30)
fig.text(0.91, 0.75, f'A(0) =\n2.0 mM', fontsize=25)
fig.text(0.91, 0.48, f'A(0) =\n1.0 mM', fontsize=25)
fig.text(0.91, 0.21, f'A(0) =\n0.0 mM', fontsize=25)
handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'$NO_3$ (A)'),
            plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'$NO_2$ (I)')]
fig.legend(handles=handles, loc='upper right', fontsize=20)
fig.suptitle(f'$NO_3$, $NO_2$ Concentration, No Drug', fontsize=30)
filename = f'{date}_{key}_no3_no2_conc_chl0.png'
plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
print(f'Saved plots/{filename}')

# Under drug data
num_rpl = 3
num_col = 5
num_row = int(np.ceil(len(rows_chl1) / num_col / num_rpl))
fig, axes = plt.subplots(num_row, num_col, figsize=(8*num_row, 3*num_col))
axes = axes.flatten()
for i ,row in enumerate(rows_chl1):
    ax = axes[i//num_rpl]
    ax.plot(times, no3_conc.loc[row], 'b.-')
    ax.plot(times, no2_conc.loc[row], 'r.-')
    ax.set_xticks([0, 20, 40])
    ax.tick_params(axis='x', labelsize=25)
    ax.set_ylim(y_min_chl1, y_max_chl1)
    ax.set_yticks([0, 1, 2])
    ax.tick_params(axis='y', labelsize=25)
fig.text(0.55, 0.05, 'Time (hours)', ha='center', fontsize=30)
fig.text(0.14, 0.9, f'I(0) = 2.0 mM', fontsize=25)
fig.text(0.29, 0.9, f'I(0) = 1.5 mM', fontsize=25)
fig.text(0.45, 0.9, f'I(0) = 1.0 mM', fontsize=25)
fig.text(0.61, 0.9, f'I(0) = 0.5 mM', fontsize=25)
fig.text(0.77, 0.9, f'I(0) = 0.0 mM', fontsize=25)
fig.text(0.08, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize=30)
fig.text(0.91, 0.75, f'A(0) =\n2.0 mM', fontsize=25)
fig.text(0.91, 0.48, f'A(0) =\n1.0 mM', fontsize=25)
fig.text(0.91, 0.21, f'A(0) =\n0.0 mM', fontsize=25)
handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'$NO_3$ (A)'),
            plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'$NO_2$ (I)')]
fig.legend(handles=handles, loc='upper right', fontsize=20)
fig.suptitle(f'$NO_3$, $NO_2$ Concentration, Under Drug', fontsize=30)
filename = f'{date}_{key}_no3_no2_conc_chl1.png'
plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
print(f'Saved plots/{filename}')

# no3_no2_cons vs time, all conditions
rows_chl0 = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09']
rows_chl1 = ['E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'H09']
all_values_chl0 = pd.concat([no3_cons.loc[rows_chl0], no2_cons.loc[rows_chl0]])
all_values_chl1 = pd.concat([no3_cons.loc[rows_chl1], no2_cons.loc[rows_chl1]])
y_min_chl0 = all_values_chl0.min().min()
y_min_chl1 = all_values_chl1.min().min()
y_max_chl0 = all_values_chl0.max().max()
y_max_chl1 = all_values_chl1.max().max()

# No drug data
num_rpl = 3
num_col = 5
num_row = int(np.ceil(len(rows_chl0) / num_col / num_rpl))
fig, axes = plt.subplots(num_row, num_col, figsize=(8*num_row, 3*num_col))
axes = axes.flatten()
for i ,row in enumerate(rows_chl0):
    ax = axes[i//num_rpl]
    ax.plot(times, no3_cons.loc[row], 'b.-')
    ax.plot(times, no2_cons.loc[row], 'r.-')
    ax.set_xticks([0, 20, 40])
    ax.tick_params(axis='x', labelsize=25)
    ax.set_ylim(y_min_chl0, y_max_chl0)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.tick_params(axis='y', labelsize=25)
fig.text(0.55, 0.05, 'Time (hours)', ha='center', fontsize=30)
fig.text(0.14, 0.9, f'I(0) = 2.0 mM', fontsize=25)
fig.text(0.29, 0.9, f'I(0) = 1.5 mM', fontsize=25)
fig.text(0.45, 0.9, f'I(0) = 1.0 mM', fontsize=25)
fig.text(0.61, 0.9, f'I(0) = 0.5 mM', fontsize=25)
fig.text(0.77, 0.9, f'I(0) = 0.0 mM', fontsize=25)
fig.text(0.08, 0.5, 'Consumption (mM)', va='center', rotation='vertical', fontsize=30)
fig.text(0.91, 0.75, f'A(0) =\n2.0 mM', fontsize=25)
fig.text(0.91, 0.48, f'A(0) =\n1.0 mM', fontsize=25)
fig.text(0.91, 0.21, f'A(0) =\n0.0 mM', fontsize=25)
handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'$-\Delta A$'),
            plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'$-\Delta I -\Delta A$')]
fig.legend(handles=handles, loc='upper right', fontsize=20)
fig.suptitle(f'$NO_3$, $NO_2$ Consumption, No Drug', fontsize=30)
filename = f'{date}_{key}_no3_no2_cons_chl0.png'
plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
print(f'Saved plots/{filename}')

# Under drug data
num_rpl = 3
num_col = 5
num_row = int(np.ceil(len(rows_chl1) / num_col / num_rpl))
fig, axes = plt.subplots(num_row, num_col, figsize=(8*num_row, 3*num_col))
axes = axes.flatten()
for i ,row in enumerate(rows_chl1):
    ax = axes[i//num_rpl]
    ax.plot(times, no3_cons.loc[row], 'b.-')
    ax.plot(times, no2_cons.loc[row], 'r.-')
    ax.set_xticks([0, 20, 40])
    ax.tick_params(axis='x', labelsize=25)
    ax.set_ylim(y_min_chl1, y_max_chl1)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='y', labelsize=25)
fig.text(0.55, 0.05, 'Time (hours)', ha='center', fontsize=30)
fig.text(0.14, 0.9, f'I(0) = 2.0 mM', fontsize=25)
fig.text(0.29, 0.9, f'I(0) = 1.5 mM', fontsize=25)
fig.text(0.45, 0.9, f'I(0) = 1.0 mM', fontsize=25)
fig.text(0.61, 0.9, f'I(0) = 0.5 mM', fontsize=25)
fig.text(0.77, 0.9, f'I(0) = 0.0 mM', fontsize=25)
fig.text(0.08, 0.5, 'Consumption (mM)', va='center', rotation='vertical', fontsize=30)
fig.text(0.91, 0.75, f'A(0) =\n2.0 mM', fontsize=25)
fig.text(0.91, 0.48, f'A(0) =\n1.0 mM', fontsize=25)
fig.text(0.91, 0.21, f'A(0) =\n0.0 mM', fontsize=25)
handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'$-\Delta A$'),
            plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'$-\Delta I -\Delta A$')]
fig.legend(handles=handles, loc='upper right', fontsize=20)
fig.suptitle(f'$NO_3$, $NO_2$ Consumption, Under Drug', fontsize=30)
filename = f'{date}_{key}_no3_no2_cons_chl1.png'
plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
print(f'Saved plots/{filename}')


# # no3_no2_nh4 vs time, one condition
# rows = ['A01', 'A02', 'A03']
# fig, axes = plt.subplots(1, 1, figsize=(8, 6))
# ax = axes
# for row in rows:
#     ax.plot(times_chl0, no3_chl0_evap.loc[row], 'b.-')
#     ax.plot(times_chl0, no2_chl0_evap.loc[row], 'r.-')
#     ax.plot(times_chl0, nh4_chl0_evap.loc[row], 'g.-')
# ax.set_xlabel('Time (hours)', fontsize=15)
# ax.set_ylabel('Concentration (mM)', fontsize=15)
# ax.tick_params(axis='x', labelsize=15)
# ax.tick_params(axis='y', labelsize=15)
# handles = [plt.Line2D([0], [0], color='b', marker='.', linestyle='-', label=f'$NO_3$ (A)'),
#             plt.Line2D([0], [0], color='r', marker='.', linestyle='-', label=f'$NO_2$ (I)'),
#             plt.Line2D([0], [0], color='g', marker='.', linestyle='-', label=f'$NH_4$')]
# ax.legend(handles=handles, loc='upper right', fontsize=12)
# plt.savefig('plots/no3_no2_nh4_chl0_(2,2).png', dpi=300, bbox_inches='tight')
# print('Figure saved as plots/no3_no2_nh4_chl0_(2,2).png')
