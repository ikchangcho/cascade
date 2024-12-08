import pandas as pd

# Import the DataFrame from the CSV file
csv_file_path = '20241103/plots/consumption_linear_fit/linear_regression_details.csv'
df = pd.read_csv(csv_file_path)

# # Import the metadata DataFrame from the CSV file
# metadata_file_path = '20241103/sample_metadata.csv'
# # Set the index of df to the 'Row' column
# df.set_index('Row', inplace=True)
# metadata_df = pd.read_csv(metadata_file_path, index_col=0)

# # Merge the dataframes on the index (assuming the index is the common key)
# df = df.merge(metadata_df[['Nitrate_input', 'Nitrite_input']], left_index=True, right_index=True, how='left')
# print(df.head())

# Add two columns to the DataFrame
df['NO3 Initial'] = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
df['NO2 Initial'] = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]

# Exclude the specified rows
rows_to_exclude = ['D02', 'D04', 'D06', 'D08', 'D10', 'D12', 'H02', 'H04', 'H06', 'H08', 'H10', 'H12']
df = df[~df['Row'].isin(rows_to_exclude)]

# Scatter plots of NO3 Initial vs. NO3 Slope for A~D rows
import matplotlib.pyplot as plt
import seaborn as sns

# Filter the DataFrame for A~D rows and exclude rows where NO3 Initial is zero
df_ad = df[df['Row'].str.startswith(('A', 'B', 'C', 'D')) & (df['NO3 Initial'] != 0)]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO3 Initial', y='NO3 Slope', hue='NO2 Initial', data=df_ad, palette='viridis')
plt.title('La Bagh Wood 2, pH 6.35', fontsize=20)
plt.xlabel('Initial NO3 (mM)', fontsize=20)
plt.ylabel('NO3 Consumption Rate (mM/hr)', fontsize=20)
plt.legend(title='Initial NO2 (mM)', loc='upper left', fontsize=15, title_fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no3_vs_no3_rate_0.png')

# Scatter plots of NO3 Initial vs. NO3 Slope for E~H rows and exclude rows where NO3 Initial is zero
# Filter the DataFrame for E~H rows
df_eh = df[df['Row'].str.startswith(('E', 'F', 'G', 'H')) & (df['NO3 Initial'] != 0)]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO3 Initial', y='NO3 Slope', hue='NO2 Initial', data=df_eh, palette='viridis')
plt.title('ELG_10 (Elder loam grassland)', fontsize=20)
plt.xlabel('Initial NO3 (mM)', fontsize=20)
plt.ylabel('NO3 Consumption Rate (mM/hr)', fontsize=20)
plt.legend(title='Initial NO2 (mM)', loc='upper left', fontsize=15, title_fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no3_vs_no3_rate_1.png')

# Scatter plots of NO2 Initial vs. NO3 Slope for A~D and E~H rows seperately. Exclude rows where NO3 Initial is zero
plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO2 Initial', y='NO3 Slope', hue='NO3 Initial', data=df_ad, palette='viridis')
plt.title('La Bagh Wood 2, pH 6.35', fontsize=20)
plt.xlabel('Initial NO2 (mM)', fontsize=20)
plt.ylabel('NO3 Consumption Rate (mM/hr)', fontsize=20)
plt.legend(title='Initial NO3 (mM)', loc='upper left', fontsize=15, title_fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no2_vs_no3_rate_0.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO2 Initial', y='NO3 Slope', hue='NO3 Initial', data=df_eh, palette='viridis')
plt.title('ELG_10 (Elder loam grassland)', fontsize=20)
plt.xlabel('Initial NO2 (mM)', fontsize=20)
plt.ylabel('NO3 Consumption Rate (mM/hr)', fontsize=20)
plt.legend(title='Initial NO3 (mM)', loc='upper left', fontsize=15, title_fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no2_vs_no3_rate_1.png')

# Scatter plots of NO3 Initial vs. NO2 Early Slope and NO2 Initial vs. NO2 Early Slope for A~D and E~H rows seperately. Exclude rows where NO2 Initial is zero.
# Filter the DataFrame for A~D rows and exclude rows where NO2 Initial is zero
df_ad_no2 = df[df['Row'].str.startswith(('A', 'B', 'C', 'D')) & (df['NO2 Initial'] != 0)]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO3 Initial', y='NO2 Early Slope', hue='NO2 Initial', data=df_ad_no2, palette='viridis')
plt.title('La Bagh Wood 2, pH 6.35', fontsize=20)
plt.xlabel('Initial NO3 (mM)', fontsize=20)
plt.ylabel('NO2 Early Consumption Rate (mM/hr)', fontsize=20)
plt.legend(title='Initial NO2 (mM)', loc='upper left', fontsize=15, title_fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no3_vs_no2_early_rate_0.png')

# Filter the DataFrame for E~H rows and exclude rows where NO2 Initial is zero
df_eh_no2 = df[df['Row'].str.startswith(('E', 'F', 'G', 'H')) & (df['NO2 Initial'] != 0)]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO3 Initial', y='NO2 Early Slope', hue='NO2 Initial', data=df_eh_no2, palette='viridis')
plt.title('ELG_10 (Elder loam grassland)', fontsize=20)
plt.xlabel('Initial NO3 (mM)', fontsize=20)
plt.ylabel('NO2 Early Consumption Rate (mM/hr)', fontsize=20)
plt.legend(title='Initial NO2 (mM)', loc='upper left', fontsize=15, title_fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no3_vs_no2_early_rate_1.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO2 Initial', y='NO2 Early Slope', hue='NO3 Initial', data=df_ad_no2, palette='viridis')
plt.title('La Bagh Wood 2, pH 6.35', fontsize=20)
plt.xlabel('Initial NO2 (mM)', fontsize=20)
plt.ylabel('NO2 Early Consumption Rate (mM/hr)', fontsize=20)
plt.legend(title='Initial NO3 (mM)', loc='upper left', fontsize=15, title_fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no2_vs_no2_early_rate_0.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO2 Initial', y='NO2 Early Slope', hue='NO3 Initial', data=df_eh_no2, palette='viridis')
plt.title('ELG_10 (Elder loam grassland)', fontsize=20)
plt.xlabel('Initial NO2 (mM)', fontsize=20)
plt.ylabel('NO2 Early Consumption Rate (mM/hr)', fontsize=20)
plt.legend(title='Initial NO3 (mM)', loc='upper left', fontsize=15, title_fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no2_vs_no2_early_rate_1.png')

# Scatter plots of NO3 Initial vs. NO2 Late Slope and NO2 Initial vs. NO2 Late Slope for A~D and E~H rows seperately.
# Filter the DataFrame for A~D rows and exclude rows where NO2 Initial is zero
df_ad_no2 = df[df['Row'].str.startswith(('A', 'B', 'C', 'D')) & (df['NO2 Initial'] != 0)]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO3 Initial', y='NO2 Late Slope', hue='NO2 Initial', data=df_ad_no2, palette='viridis')
plt.title('La Bagh Wood 2, pH 6.35', fontsize=20)
plt.xlabel('Initial NO3 (mM)', fontsize=20)
plt.ylabel('NO2 Late Consumption Rate (mM/hr)', fontsize=20)
plt.legend(title='Initial NO2 (mM)', loc='upper left', fontsize=15, title_fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no3_vs_no2_late_rate_0.png')

# Filter the DataFrame for E~H rows and exclude rows where NO2 Initial is zero
df_eh_no2 = df[df['Row'].str.startswith(('E', 'F', 'G', 'H')) & (df['NO2 Initial'] != 0)]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO3 Initial', y='NO2 Late Slope', hue='NO2 Initial', data=df_eh_no2, palette='viridis')
plt.title('ELG_10 (Elder loam grassland)', fontsize=20)
plt.xlabel('Initial NO3 (mM)', fontsize=20)
plt.ylabel('NO2 Late Consumption Rate (mM/hr)', fontsize=20)
plt.legend(title='Initial NO2 (mM)', loc='upper left', fontsize=15, title_fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no3_vs_no2_late_rate_1.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO2 Initial', y='NO2 Late Slope', hue='NO3 Initial', data=df_ad_no2, palette='viridis')
plt.title('La Bagh Wood 2, pH 6.35', fontsize=20)
plt.xlabel('Initial NO2 (mM)', fontsize=20)
plt.ylabel('NO2 Late Consumption Rate (mM/hr)', fontsize=20)
plt.legend(title='Initial NO3 (mM)', loc='upper left', fontsize=15, title_fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no2_vs_no2_late_rate_0.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO2 Initial', y='NO2 Late Slope', hue='NO3 Initial', data=df_eh_no2, palette='viridis')
plt.title('ELG_10 (Elder loam grassland)', fontsize=20)
plt.xlabel('Initial NO2 (mM)', fontsize=20)
plt.ylabel('NO2 Late Consumption Rate (mM/hr)', fontsize=20)
plt.legend(title='Initial NO3 (mM)', loc='upper left', fontsize=15, title_fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no2_vs_no2_late_rate_1.png')