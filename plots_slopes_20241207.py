import pandas as pd

# Import the DataFrame from the CSV file
csv_file_path = '20241103/plots/consumption_linear_fit/linear_regression_details.csv'
df = pd.read_csv(csv_file_path)

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
sns.scatterplot(x='NO3 Initial', y='NO3 Slope', hue='NO2 Initial', data=df_ad, palette=['blue', 'red', 'violet', 'yellow'])
plt.xlabel('Initial NO3 (mM)', fontsize=20)
plt.ylabel('NO3 Consumption Rate (mM/hr)', fontsize=20)
legend = plt.legend(title='I(0) (mM)', loc='upper left', fontsize=15, title_fontsize=15)
legend.get_title().set_color('red')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no3_vs_no3_rate_0.png')

# Scatter plots of NO3 Initial vs. NO3 Slope for E~H rows and exclude rows where NO3 Initial is zero
# Filter the DataFrame for E~H rows
df_eh = df[df['Row'].str.startswith(('E', 'F', 'G', 'H')) & (df['NO3 Initial'] != 0)]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO3 Initial', y='NO3 Slope', hue='NO2 Initial', data=df_eh, palette=['blue', 'red', 'violet', 'yellow'])
plt.xlabel('Initial NO3 (mM)', fontsize=20)
plt.ylabel('NO3 Consumption Rate (mM/hr)', fontsize=20)
legend = plt.legend(title='I(0) (mM)', loc='upper left', fontsize=15, title_fontsize=15)
legend.get_title().set_color('red')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no3_vs_no3_rate_1.png')

# Scatter plots of NO2 Initial vs. NO3 Slope for A~D and E~H rows separately. Exclude rows where NO3 Initial is zero
plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO2 Initial', y='NO3 Slope', hue='NO3 Initial', data=df_ad, palette=['blue', 'red', 'violet'])
plt.xlabel('Initial NO2 (mM)', fontsize=20)
plt.ylabel('NO3 Consumption Rate (mM/hr)', fontsize=20)
legend = plt.legend(title='A(0) (mM)', loc='upper left', fontsize=15, title_fontsize=15)
legend.get_title().set_color('blue')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no2_vs_no3_rate_0.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO2 Initial', y='NO3 Slope', hue='NO3 Initial', data=df_eh, palette=['blue', 'red', 'violet'])
plt.xlabel('Initial NO2 (mM)', fontsize=20)
plt.ylabel('NO3 Consumption Rate (mM/hr)', fontsize=20)
legend = plt.legend(title='A(0) (mM)', loc='upper left', fontsize=15, title_fontsize=15)
legend.get_title().set_color('blue')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no2_vs_no3_rate_1.png')

# Scatter plots of NO3 Initial vs. NO2 Early Slope and NO2 Initial vs. NO2 Early Slope for A~D and E~H rows separately. Exclude rows where NO2 Initial is zero.
# Filter the DataFrame for A~D rows and exclude rows where NO2 Initial is zero
df_ad_no2 = df[df['Row'].str.startswith(('A', 'B', 'C', 'D')) & (df['NO2 Initial'] != 0)]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO3 Initial', y='NO2 Early Slope', hue='NO2 Initial', data=df_ad_no2, palette=['blue', 'red', 'violet'])
plt.xlabel('Initial NO3 (mM)', fontsize=20)
plt.ylabel('NO2 Early Consumption Rate (mM/hr)', fontsize=20)
legend = plt.legend(title='I(0) (mM)', loc='upper left', fontsize=15, title_fontsize=15)
legend.get_title().set_color('red')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no3_vs_no2_early_rate_0.png')

# Filter the DataFrame for E~H rows and exclude rows where NO2 Initial is zero
df_eh_no2 = df[df['Row'].str.startswith(('E', 'F', 'G', 'H')) & (df['NO2 Initial'] != 0)]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO3 Initial', y='NO2 Early Slope', hue='NO2 Initial', data=df_eh_no2, palette=['blue', 'red', 'violet'])
plt.xlabel('Initial NO3 (mM)', fontsize=20)
plt.ylabel('NO2 Early Consumption Rate (mM/hr)', fontsize=20)
legend = plt.legend(title='I(0) (mM)', loc='upper left', fontsize=15, title_fontsize=15)
legend.get_title().set_color('red')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no3_vs_no2_early_rate_1.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO2 Initial', y='NO2 Early Slope', hue='NO3 Initial', data=df_ad_no2, palette=['blue', 'red', 'violet', 'yellow'])
plt.xlabel('Initial NO2 (mM)', fontsize=20)
plt.ylabel('NO2 Early Consumption Rate (mM/hr)', fontsize=20)
legend = plt.legend(title='A(0) (mM)', loc='upper left', fontsize=15, title_fontsize=15)
legend.get_title().set_color('blue')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no2_vs_no2_early_rate_0.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO2 Initial', y='NO2 Early Slope', hue='NO3 Initial', data=df_eh_no2, palette=['blue', 'red', 'violet', 'yellow'])
plt.xlabel('Initial NO2 (mM)', fontsize=20)
plt.ylabel('NO2 Early Consumption Rate (mM/hr)', fontsize=20)
legend = plt.legend(title='A(0) (mM)', loc='upper left', fontsize=15, title_fontsize=15)
legend.get_title().set_color('blue')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no2_vs_no2_early_rate_1.png')

# Scatter plots of NO3 Initial vs. NO2 Late Slope and NO2 Initial vs. NO2 Late Slope for A~D and E~H rows separately.
# Filter the DataFrame for A~D rows and exclude rows where NO2 Initial is zero
df_ad_no2 = df[df['Row'].str.startswith(('A', 'B', 'C', 'D')) & (df['NO2 Initial'] != 0)]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO3 Initial', y='NO2 Late Slope', hue='NO2 Initial', data=df_ad_no2, palette=['blue', 'red', 'violet'])
plt.xlabel('Initial NO3 (mM)', fontsize=20)
plt.ylabel('NO2 Late Consumption Rate (mM/hr)', fontsize=20)
legend = plt.legend(title='I(0) (mM)', loc='upper left', fontsize=15, title_fontsize=15)
legend.get_title().set_color('red')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no3_vs_no2_late_rate_0.png')

# Filter the DataFrame for E~H rows and exclude rows where NO2 Initial is zero
df_eh_no2 = df[df['Row'].str.startswith(('E', 'F', 'G', 'H')) & (df['NO2 Initial'] != 0)]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO3 Initial', y='NO2 Late Slope', hue='NO2 Initial', data=df_eh_no2, palette=['blue', 'red', 'violet'])
plt.xlabel('Initial NO3 (mM)', fontsize=20)
plt.ylabel('NO2 Late Consumption Rate (mM/hr)', fontsize=20)
legend = plt.legend(title='I(0) (mM)', loc='upper left', fontsize=15, title_fontsize=15)
legend.get_title().set_color('red')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no3_vs_no2_late_rate_1.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO2 Initial', y='NO2 Late Slope', hue='NO3 Initial', data=df_ad_no2, palette=['blue', 'red', 'violet', 'yellow'])
plt.xlabel('Initial NO2 (mM)', fontsize=20)
plt.ylabel('NO2 Late Consumption Rate (mM/hr)', fontsize=20)
legend = plt.legend(title='A(0) (mM)', loc='upper left', fontsize=15, title_fontsize=15)
legend.get_title().set_color('blue')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no2_vs_no2_late_rate_0.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NO2 Initial', y='NO2 Late Slope', hue='NO3 Initial', data=df_eh_no2, palette=['blue', 'red', 'violet', 'yellow'])
plt.xlabel('Initial NO2 (mM)', fontsize=20)
plt.ylabel('NO2 Late Consumption Rate (mM/hr)', fontsize=20)
legend = plt.legend(title='A(0) (mM)', loc='upper left', fontsize=15, title_fontsize=15)
legend.get_title().set_color('blue')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no2_vs_no2_late_rate_1.png')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('20241103/plots/no2_vs_no2_late_rate_1.png')
