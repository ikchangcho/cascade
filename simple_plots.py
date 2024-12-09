import numpy as np

import matplotlib.pyplot as plt

# Define the range for A
A = np.linspace(0, 100, 500)

# Define the functions
y1 = A / (1 + A)
y2 = A / (10 + A)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(A, y1, label='y = A / (1 + A)', color='r')
plt.plot(A, y2, label='y = A / (10 + A)', color='b')

# Add horizontal dotted lines at y=1 and y=0.5
plt.axhline(y=1, color='k', linestyle='--', linewidth=0.7)
# plt.axhline(y=0.5, color='k', linestyle='--', linewidth=0.7)

# # Add vertical dotted lines at A=1 and A=10, cut at y=0.5
# plt.plot([1, 1], [0, 0.5], color='r', linestyle='--', linewidth=0.7)
# plt.plot([10, 10], [0, 0.5], color='b', linestyle='--', linewidth=0.7)

# # Add text annotations for A=1 and A=10
# plt.text(1, -0.1, '1', color='r', ha='center')
# plt.text(10, -0.1, '10', color='b', ha='center')

# Add labels and title
plt.xlabel('A', fontsize=15)
plt.ylabel('y', fontsize=15)
#plt.title('Plots of y = A / (1 + A) and y = A * (10 + A)')
plt.legend(fontsize=15)
#plt.grid(True)

# Show the plot
plt.show()