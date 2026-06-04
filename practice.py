import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 4])
ax.set_xticks([3, 1, 2])
ax.set_xticklabels(['zero', 'one', 'two'])
plt.show()