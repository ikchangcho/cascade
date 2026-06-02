import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
df = pd.read_csv(f'concentrations/4.2.batch1_no3_cons.csv', index_col=0)
print(df.loc['A04', 'Sample_type'] == )