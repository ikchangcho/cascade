import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
y = [1, 0.5, 0.1, 0.04, 0.0]
zero_indices = np.where(np.array(y) < 0.05)
print(zero_indices[0])
