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
import lmfit
from model1 import *


t_eval = np.linspace(0, 60, 100)
A0 = 2.0
I0 = 1.0
X0 = 0.1

params = Parameters()
#params.add('eps',  value=0.1, min=1e-3, max=1)
params.add('K_A',  value=0.1, min=1e-4, max=10)
params.add('K_I',  value=0.1, min=1e-4, max=10)
params.add('r_A',  value=1.0, min=1e-4, max=1e3)
params.add('r_I',  value=1.0, min=1e-4, max=1e3)
params.add('gamA', value=1.0, min=1e-4, max=1e3)
params.add('gamI', value=1.0, min=1e-4, max=1e3)
params.add('X0',   value=0.1, min=1e-3, max=1e3, vary=False)

A_sim, I_sim = simulate(params, (t_eval[0], t_eval[-1]), [A0, I0, X0], t_eval)

