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

filepath = '20250114'
meta_data = pd.read_csv(f'{filepath}/sample_metadata.csv', index_col=0).dropna(how='all')
print(10 % 3)
print(10 // 3)