import pandas as pd
import numpy as np

# Read the CSV file (adjust the path accordingly)
df = pd.read_csv('data/20240914_Ik_NO2_standard_900.csv', header=None)

# Get the number of rows in the DataFrame
num_rows = df.shape[0] - 6

# Generate row indices for rows 6, 7, 10, 11, 14, 15, ...
rows_to_extract = []
for n in range(num_rows // 4):  # Ensures we don't go out of bounds
    rows_to_extract.append(6 + 4 * n)
    rows_to_extract.append(7 + 4 * n)

# Extract only the specified columns (0 and 1) and the rows
extracted_df = df.loc[rows_to_extract, [0, 1]]

# Convert the extracted data to float values using pd.to_numeric
extracted_df[0] = pd.to_numeric(extracted_df[0], errors='coerce')
extracted_df[1] = pd.to_numeric(extracted_df[1], errors='coerce')

# Convert the extracted data into a NumPy array
extracted_array = extracted_df.to_numpy()

flat_data_900 = extracted_array.flatten()

q25 = np.percentile(flat_data_900, 25)
q75 = np.percentile(flat_data_900, 75)

# Calculate the outlier thresholds
upper_threshold = q75
lower_threshold = 2 * q25 - q75

# Create a boolean mask where True indicates outliers
mask = (flat_data_900 > upper_threshold) | (flat_data_900 < lower_threshold)

flat_data_900[mask] = np.NaN
print(flat_data_900)