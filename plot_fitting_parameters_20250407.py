import pickle
import pandas as pd
import matplotlib.pyplot as plt

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

for rows_to_plot in [A20_rows, A15_rows, A10_rows, A05_rows, A00_rows]:
    for param, color in zip(['r_A', 'r_I', 'GamA', 'GamI'], ['blue', 'red', 'blue', 'red']):
        y_values = params.loc[rows_to_plot, param]
        A0_values = metadata.loc[rows_to_plot, 'Nitrate_input']
        I0_values = metadata.loc[rows_to_plot, 'Nitrite_input']

        # Create a scatter plot for parameter values on same A(0)
        plt.figure(figsize=(8, 6))
        plt.scatter(I0_values, y_values, color=color)
        plt.xlabel('I(0)', fontsize=15)
        plt.ylabel(param, fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.title(f'A(0)={A0_values.iloc[0]:.1f} mM', fontsize=15)
        plt.grid(True)
        plt.savefig(f'plots/{param}_A0_{A0_values.iloc[0]:.1f}.png')
        plt.close()
        print(f'Saved plot: {param}_A0_{A0_values.iloc[0]:.1f}.png')


for rows_to_plot in [I20_rows, I15_rows, I10_rows, I05_rows, I00_rows]:
    for param, color in zip(['r_A', 'r_I', 'GamA', 'GamI'], ['blue', 'red', 'blue', 'red']):
        y_values = params.loc[rows_to_plot, param]
        A0_values = metadata.loc[rows_to_plot, 'Nitrate_input']
        I0_values = metadata.loc[rows_to_plot, 'Nitrite_input']

        # Create a scatter plot for parameter values on same A(0)
        plt.figure(figsize=(8, 6))
        plt.scatter(A0_values, y_values, color=color)
        plt.xlabel('A(0)', fontsize=15)
        plt.ylabel(param, fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.title(f'I(0)={I0_values.iloc[0]:.1f} mM', fontsize=15)
        plt.grid(True)
        plt.savefig(f'plots/{param}_I0_{I0_values.iloc[0]:.1f}.png')
        plt.close()
        print(f'Saved plot: {param}_I0_{I0_values.iloc[0]:.1f}.png')

