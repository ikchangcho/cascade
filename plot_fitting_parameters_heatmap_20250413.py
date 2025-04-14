import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

all_rows = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', 'G01', 'G02', 'G03']

A20_rows = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03']
A15_rows = ['B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06']
A10_rows = ['C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09']
A05_rows = ['E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'F01', 'F02', 'F03']
A00_rows = ['F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', 'G01', 'G02', 'G03']

I20_rows = ['A01', 'A02', 'A03', 'B04', 'B05', 'B06', 'C07', 'C08', 'C09', 'E01', 'E02', 'E03', 'F04', 'F05', 'F06']
I15_rows = ['A04', 'A05', 'A06', 'B07', 'B08', 'B09', 'C10', 'C11', 'C12', 'E04', 'E05', 'E06', 'F07', 'F08', 'F09']
I10_rows = ['A07', 'A08', 'A09', 'B10', 'B11', 'B12', 'D01', 'D02', 'D03', 'E07', 'E08', 'E09', 'F10', 'F11', 'F12']
I05_rows = ['A10', 'A11', 'A12', 'C01', 'C02', 'C03', 'D04', 'D05', 'D06', 'E10', 'E11', 'E12', 'G01', 'G02', 'G03']
I00_rows = ['B01', 'B02', 'B03', 'C04', 'C05', 'C06', 'D07', 'D08', 'D09', 'F01', 'F02', 'F03']

data_dict = {'r_A': [], 'r_I': [], 'GamA': [], 'GamI': []}

for row in all_rows:
    with open(f"fitting_results/no3_no2_chl1_model2_rA_rI_{row}.pkl", "rb") as file:
        data = pickle.load(file)
        data_dict['r_A'].append(data.params['r_A'].value)
        data_dict['r_I'].append(data.params['r_I'].value)
    with open(f"fitting_results/no3_no2_chl0_model2_GamA_GamI_{row}.pkl", "rb") as file:
        data = pickle.load(file)
        data_dict['GamA'].append(data.params['GamA'].value)
        data_dict['GamI'].append(data.params['GamI'].value)

params = pd.DataFrame(data_dict, index=all_rows)
params.to_csv("fitting_results/parameters.csv")
# Load metadata
metadata = pd.read_csv("concentrations/samples_metadata_chl01.csv", index_col=0)

# Extract initial conditions
A0_values = metadata.loc[params.index, 'Nitrate_input']
I0_values = metadata.loc[params.index, 'Nitrite_input']

# Add initial conditions to the params DataFrame
params['A0'] = A0_values
params['I0'] = I0_values

# Group by initial conditions (A0, I0) and compute averages and medians
grouped = params.groupby(['A0', 'I0']).agg(['mean', 'median'])

# Extract the averages and medians for each parameter
parameters = ['r_A', 'r_I', 'GamA', 'GamI']
for param in parameters:
    for stat in ['mean', 'median']:
        # Create a pivot table for the parameter
        heatmap_data = grouped[param][stat].unstack()

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': param})
        plt.title(f"Heatmap of {param} ({stat})")
        plt.xlabel("A(0) (Nitrate_input)")
        plt.ylabel("I(0) (Nitrite_input)")
        plt.tight_layout()
        plt.savefig(f"heatmaps/{param}_{stat}_heatmap.png")
        plt.show()


