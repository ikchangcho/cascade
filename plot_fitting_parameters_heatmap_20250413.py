import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

all_rows = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', 'G01', 'G02', 'G03']

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
# Load metadata
metadata = pd.read_csv("concentrations/samples_metadata_chl01.csv", index_col=0)

# Extract initial conditions
A0_values = metadata.loc[params.index, 'Nitrate_input']
I0_values = metadata.loc[params.index, 'Nitrite_input']

# Add initial conditions to the params DataFrame
params['A0'] = A0_values
params['I0'] = I0_values
params.to_csv("fitting_results/parameters.csv")

# Group by initial conditions (A0, I0) and compute averages and medians
grouped = params.groupby(['A0', 'I0']).agg(['mean', 'median'])

# Extract the averages and medians for each parameter
parameters = ['r_A', 'r_I', 'GamA', 'GamI']
labels = ['$r_A$', '$r_I$', '$\Gamma_A$', '$\Gamma_I$']
for param, label in zip(parameters, labels):
    for stat in ['mean', 'median']:
        # Create a pivot table for the parameter
        heatmap_data = grouped[param][stat].unstack()

        # Exclude A(0) = 0 cases for r_A and GamA
        if param in ['r_A', 'GamA']:
            heatmap_data = heatmap_data.loc[heatmap_data.index != 0]

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="gray")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(f"{label} {stat} values", fontsize=20)
        plt.ylabel("A(0) (mM)", fontsize=15)
        plt.xlabel("I(0) (mM)", fontsize=15)
        plt.savefig(f"plots/heatmap_{param}_{stat}.png")
        print(f"Saved plots/heatmap_{param}_{stat}.png")


