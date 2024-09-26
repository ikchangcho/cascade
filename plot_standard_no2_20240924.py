import numpy as np
import matplotlib.pyplot as plt

# 1. Load the CSV file into a numpy array
data = np.loadtxt('data/20240914_Ik_NO2_standard_540_average.csv', delimiter=',')

# 2. Reshape the array into 8 rows and 9 columns
reshaped_data = data.reshape(8, 9)

# 3. Define x values
x_values = np.array([2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0])

# 4. Check for rows where all three values are NaN, and remove those rows
valid_rows = ~np.all(np.isnan(reshaped_data[:, :3]), axis=1)

# Filter out invalid rows from reshaped_data and x_values
reshaped_data = reshaped_data[valid_rows]
x_values = x_values[valid_rows]

# 5. Calculate the mean and standard deviation for the first three columns (replicates), ignoring NaN values
mean_values = np.nanmean(reshaped_data[:, :3], axis=1)
std_dev = np.nanstd(reshaped_data[:, :3], axis=1)

# 6. Plot the mean with error bars (standard deviation), with smaller dots
plt.figure()
plt.errorbar(x_values, mean_values, yerr=std_dev, fmt='o', capsize=5, markersize=5)

# 7. Fit a quadratic function (2nd degree polynomial) using the remaining valid data
coefficients = np.polyfit(x_values, mean_values, 2)
quadratic_fit = np.poly1d(coefficients)

# Generate points for the fit curve
x_fit = np.linspace(min(x_values), max(x_values), 100)
y_fit = quadratic_fit(x_fit)
equation_text = f'y = {coefficients[0]:.4f}x² + {coefficients[1]:.4f}x + {coefficients[2]:.4f}'

# Plot the fitted quadratic curve
plt.plot(x_fit, y_fit, label=equation_text, linestyle='--')

# # Dynamically adjust the position of the equation text to fit on the plot
# x_text_pos = np.mean(x_values)
# y_text_pos = np.max(mean_values) + (np.max(mean_values) - np.min(mean_values)) * 0.1
# plt.text(x_text_pos, y_text_pos, equation_text, fontsize=10, color='black')

# Labels and title
plt.xlabel('[$NO_2$] (mM)')
plt.ylabel('Absorbance')
plt.title('$NO_2$ Standard Curve (540 nm)')
plt.legend()

# Display the plot
plt.show()