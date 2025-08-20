import matplotlib.pyplot as plt
# Data for second plot
time_2 = [0, 0.966666667, 1.965277778, 3.882638889, 5.840972222, 6.8125, 8.814583333]
water_pink = [75.62154696, 72.14654696, 68.69654696, 61.29654696, 55.77154696, 52.09654696, 45.62154696]
water_white = [75.84654696, 72.57154696, 69.47154696, 63.04654696, 58.27154696, 55.22154696, 49.64654696]

# Plotting second figure
plt.figure(figsize=(8, 6))
plt.plot(time_2, water_pink, label='Water in Soil 1', marker='o')
plt.plot(time_2, water_white, label='Water in Soil 2', marker='o')

# Labels and title for second figure
plt.xlabel('Time (days)', fontsize=15)
plt.ylabel('Water Content (%WHC)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.title("La Bagh Soil Water Content under 26.2°C, 87%", fontsize=15, fontweight='bold')
plt.legend(fontsize=15)
plt.grid(True)

# Show second plot
plt.tight_layout()
plt.show()

# Data
time = [0, 0.966666667, 1.965277778, 3.882638889, 5.840972222, 6.8125, 8.814583333]
pink_mass = [75, 73.61, 72.23, 69.27, 67.06, 65.59, 63]
white_mass = [75.09, 73.78, 72.54, 69.97, 68.06, 66.84, 64.61]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(time, pink_mass, label='Soil 1', marker='o')
plt.plot(time, white_mass, label='Soil 2', marker='o')

# Labels and title
font_size = 15
plt.xlabel('Time (days)', fontsize = font_size)
plt.ylabel('Mass (g)', fontsize = font_size)
plt.tick_params(axis='both', which='major', labelsize = font_size)
plt.title("La Bagh Soil Drying under 26.2°C, 87%", fontsize=font_size, fontweight='bold')
plt.legend(fontsize = font_size)
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()