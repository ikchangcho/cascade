import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
x = ['Group A', 'Group B', 'Group C']
means = [10, 15, 12]
stds = [1.2, 2.5, 0.8]

# Create bar plot with error bars
plt.bar(x, means, yerr=stds, capsize=5, color='skyblue', label='Mean ± SD')
plt.ylabel('Value')
plt.legend()
plt.show()