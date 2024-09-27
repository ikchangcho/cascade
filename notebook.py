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

meta_fn = 'data/standards_metadata.csv'
data_540_fn = 'data/20240914_Ik_NO2_standard_540.CSV'
data_900_fn = 'data/20240914_Ik_NO2_standard_900.CSV'

bd.read_abs_wellscan(data_540_fn)
data = gr.read_griess(meta_fn, data_540_fn=data_540_fn, data_900_fn=data_900_fn)
fn = glob.glob(f"/Users/ik/Pycharm/cascade/data/*_Ik_NO2NO3_time0_900*")[0]
print(bd.read_abs_wellscan(fn))