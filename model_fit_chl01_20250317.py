import sys
sys.path.append('./_functions')
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters
from scipy.interpolate import interp1d
import json
import pickle
import copy
from model1 import *

# Load your data
no2 = pd.read_csv(f'concentrations/no2_chl1_evap.csv', index_col=0)
no3 = pd.read_csv(f'concentrations/no3_chl1_evap.csv', index_col=0)
#nh4 = pd.read_csv(f'concentrations/nh4_chl1_evap.csv', index_col=0)

