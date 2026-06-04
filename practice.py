import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
df = pd.DataFrame({'height': [150, 160, 170, 180, 190], 'weight': [50, 60, 70, 80, 90]})
height = df['height'].astype(float).values
print(type(height))