import numpy as np
import pandas as pd

df = pd.read_csv(f'concentrations/4.2.batch1_no3_conc.csv', index_col=0)
print(df.loc['A01'].iloc[-4])