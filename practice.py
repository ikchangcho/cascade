import sys
import os
import pandas as pd
import numpy as np
import glob
import pickle
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from datetime import datetime
import importlib
import griess as gr
import bmgdata as bd
import denitfit as dn

# Load the pickle file
monocultures = pickle.load(open( "monocultures.pkl", "rb" ))

# Display the type of the loaded data
print(f'Type of data: {type(monocultures)}')

# Access the first object in the first sublist
first_experiment = monocultures[0][0]

if hasattr(first_experiment, 'A0'):
	print(first_experiment.A0)
else:
	print("Attribute 'A0' does not exist in first_experiment")
print(first_experiment.A)
print(first_experiment.t)


# # Get a list of all attributes and methods of the object
# attributes = dir(first_experiment)
# print("Attributes and methods of the first experiment object:")
# for attribute in attributes:
#     print(attribute)

# # Get detailed information about the object
# print("\nDetailed information about the first experiment object:")
# help(first_experiment)

# # Define the range for A
# A = np.linspace(0, 100, 500)

# # Define the functions
# y1 = A / (1 + A)
# y2 = A / (10 + A)

# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(A, y1, label='y = A / (1 + A)', color='r')
# plt.plot(A, y2, label='y = A / (10 + A)', color='b')

# # Add horizontal dotted lines at y=1 and y=0.5
# plt.axhline(y=1, color='k', linestyle='--', linewidth=0.7)
# # plt.axhline(y=0.5, color='k', linestyle='--', linewidth=0.7)

# # # Add vertical dotted lines at A=1 and A=10, cut at y=0.5
# # plt.plot([1, 1], [0, 0.5], color='r', linestyle='--', linewidth=0.7)
# # plt.plot([10, 10], [0, 0.5], color='b', linestyle='--', linewidth=0.7)

# # # Add text annotations for A=1 and A=10
# # plt.text(1, -0.1, '1', color='r', ha='center')
# # plt.text(10, -0.1, '10', color='b', ha='center')

# # Add labels and title
# plt.xlabel('A', fontsize=15)
# plt.ylabel('y', fontsize=15)
# #plt.title('Plots of y = A / (1 + A) and y = A * (10 + A)')
# plt.legend(fontsize=15)
# #plt.grid(True)

# # Show the plot
# plt.show()