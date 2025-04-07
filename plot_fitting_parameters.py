import pickle
import pandas as pd
import matplotlib.pyplot as plt

rows = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'B01', 'B02', 'B03']
data_dict = {'r_A': [], 'r_I': [], 'gamA': [], 'gamI': []}

for row in rows:
    with open(f"fitting_results/no3_no2_chl1_model2_rA_rI_{row}.pkl", "rb") as file:
        data = pickle.load(file)
        data_dict['r_A'].append(data.params['r_A'].value)
        data_dict['r_I'].append(data.params['r_I'].value)
    with open(f"fitting_results/no3_no2_chl0_model2_gamA_gamI_{row}.pkl", "rb") as file:
        data = pickle.load(file)
        data_dict['gamA'].append(data.params['gamA'].value)
        data_dict['gamI'].append(data.params['gamI'].value)

params = pd.DataFrame(data_dict, index=rows)

# Load metadata
metadata = pd.read_csv("concentrations/samples_metadata.csv", index_col=0)

# Extract x and y values
x_values = metadata.loc[rows, 'Nitrite_input']
y_values = params['r_A']

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, color='blue', label='r_A')
plt.xlabel('I(0)', fontsize=15)
plt.ylabel('r_A', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.title('A(0)=2.0 mM', fontsize=15)
plt.grid(True)
plt.show()
