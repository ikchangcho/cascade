import sys
sys.path.append('/Users/ik/Pycharm/cascade')
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

meta_fn = '/Users/ik/Pycharm/cascade/data_20240914/sample_metadata.csv'
data_540_fn = 'data_20240914/20240914_Ik_NO2_standard_540.CSV'
data_900_fn = 'data_20240914/20240914_Ik_NO2_standard_900.CSV'

meta_df = pd.read_csv(meta_fn, index_col=0).dropna(how='all')
print(meta_df)