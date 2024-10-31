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

datetime_array = [
    datetime(2024, 10, 1, 13, 1),
    datetime(2024, 10, 2, 5, 0),
    datetime(2024, 10, 2, 16, 12),
    datetime(2024, 10, 3, 11, 11),
    datetime(2024, 10, 3, 11, 11),
    datetime(2024, 10, 3, 11, 11),
    datetime(2024, 10, 3, 11, 11),
    datetime(2024, 10, 3, 11, 11)]

times = [0]
for i in range(1, len(datetime_array)):
    time_diff = datetime_array[i] - datetime_array[0]  # Subtract previous from current
    times.append(time_diff.total_seconds() / 3600)

print(times)

