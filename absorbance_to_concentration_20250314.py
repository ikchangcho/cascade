import sys
sys.path.append('functions')
print("Python path: ", sys.executable)
import os
#print("Current working directory:", os.getcwd())
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
import ammonia as am

def no2_no3_abs_to_conc(date, chl):
    std_meta_fn = glob.glob(f'raw_data/{date}_standards_metadata.csv')[0]
    std_no2_540_fn = glob.glob(f"raw_data/{date}_Ik_STD_NO2_540.CSV")[0]
    std_no2_900_fn = glob.glob(f"raw_data/{date}_Ik_STD_NO2_900.CSV")[0]
    std_no2no3_540_fn = glob.glob(f"raw_data/{date}_Ik_STD_NO2NO3_540.CSV")[0]
    std_no2no3_900_fn = glob.glob(f"raw_data/{date}_Ik_STD_NO2NO3_900.CSV")[0]

    meta_fn = glob.glob(f'raw_data/{date}_samples_metadata.csv')[0]
    no2_540_fns = sorted(glob.glob(f'raw_data/{date}_Ik_NO2_{chl}*540*'))
    no2_900_fns = sorted(glob.glob(f'raw_data/{date}_Ik_NO2_{chl}*900*'))
    no2no3_540_fns = sorted(glob.glob(f'raw_data/{date}_Ik_NO2NO3_{chl}*540*'))
    no2no3_900_fns = sorted(glob.glob(f'raw_data/{date}_Ik_NO2NO3_{chl}*900*'))

    # fitted parameters
    [[no2_blank, no2no3_blank], g_fit, v_fit, no3_fit] = gr.fit_griess(meta_fn = std_meta_fn, no2_540_fn=std_no2_540_fn, no2_900_fn=std_no2_900_fn, no2no3_540_fn = std_no2no3_540_fn, no2no3_900_fn = std_no2no3_900_fn)

    # create times series dataframe
    no2_conc_dic = {}
    no3_conc_dic = {}
    col_num = 0
    for no2_540_fn, no2_900_fn, no2no3_540_fn, no2no3_900_fn in zip(no2_540_fns, no2_900_fns, no2no3_540_fns, no2no3_900_fns):
        df = gr.get_concentration(no2_blank=no2_blank, no2no3_blank=no2no3_blank, g_fit=g_fit, v_fit=no3_fit,
                            meta_fn=meta_fn, no2_fn=None, no2_540_fn=no2_540_fn, no2_900_fn=no2_900_fn,
                            no2no3_fn=None, no2no3_540_fn=no2no3_540_fn, no2no3_900_fn=no2no3_900_fn)
        no2_conc_dic[col_num] = df['NO2_mM'].copy()
        no3_conc_dic[col_num] = df['NO3_mM'].copy()
        col_num += 1
    
    no2_conc = pd.DataFrame(no2_conc_dic)
    no3_conc = pd.DataFrame(no3_conc_dic)

    return no2_conc, no3_conc

def nh4_abs_to_conc(date, chl):
    std_am_meta_fn = f"data/{date}_standards_metadata.csv"
    std_am_absorb_fn = glob.glob(f"data/{date}_NH4_STD_{chl}_650.CSV")[0]
    std_am_900_fn = glob.glob(f"data/{date}_NH4_STD_{chl}_900.CSV")[0]

    meta_fn = f"data/{date}_samples_metadata.csv"
    nh4_650_fns = sorted(glob.glob(f"data/{date}_NH4_{chl}*650*"))
    nh4_900_fns = sorted(glob.glob(f"data/{date}_NH4_{chl}*900*"))

    fit_list = am.fit_ammonia(meta_fn = std_am_meta_fn, wavelength="650", data_absorb_fn = std_am_absorb_fn, data_900_fn = std_am_900_fn)
    [[ammonia_blank], fit, b_fit] = fit_list 

    # create times series dataframe
    nh4_dic = {}
    col_num = 0
    for nh4_650_fn, nh4_900_fn in zip(nh4_650_fns, nh4_900_fns):
        df = am.get_concentration(ammonia_blank = ammonia_blank, fit = b_fit, meta_fn = meta_fn, wavelength = "650", data_absorb_fn=nh4_650_fn, data_900_fn=nh4_900_fn)
        nh4_dic[col_num] = df['Ammonia_mM'].copy()
        col_num += 1

    nh4_conc = pd.DataFrame(nh4_dic)

    return nh4_conc

###################################Input Parameters###################################
# Convert absorbance to concentration and save as dataframes
[no2_chl0_conc, no3_chl0_conc] = no2_no3_abs_to_conc("20250114", "chl-")
[no2_chl1_conc, no3_chl1_conc] = no2_no3_abs_to_conc("20250113", "chl+")
nh4_chl0_conc = pd.concat([nh4_abs_to_conc("20250305", "chl-"), nh4_abs_to_conc("20250308", "chl-")], axis=1)
nh4_chl1_conc = nh4_abs_to_conc("20250304", "chl+")

# Time arrays
chl0_datetime_array = [
    datetime(2025, 1, 1, 14, 30),       # T0
    datetime(2025, 1, 1, 15, 35),       # T1
    datetime(2025, 1, 1, 16, 55),       # T2
    datetime(2025, 1, 1, 20, 00),       # T3
    datetime(2025, 1, 1, 22, 58),       # T4
    datetime(2025, 1, 2, 8, 9),       # T5
    datetime(2025, 1, 2, 13, 10),       # T6
    datetime(2025, 1, 2, 17, 56),       # T7
    datetime(2025, 1, 2, 22, 57),       # T8
    datetime(2025, 1, 3, 8, 5),       # T9
    datetime(2025, 1, 3, 13, 3),       # T10
    datetime(2025, 1, 3, 18, 11),       # T11
    datetime(2025, 1, 3, 23, 45)]       # T12       
chl1_datetime_array = [
    datetime(2024, 12, 24, 23, 30),       # T0
    datetime(2024, 12, 25, 8, 0),       # T1
    datetime(2024, 12, 25, 17, 38),       # T2
    datetime(2024, 12, 26, 17, 43),       # T3
    datetime(2024, 12, 27, 17, 50),       # T4
    datetime(2024, 12, 28, 18, 10),       # T5
    datetime(2024, 12, 29, 17, 30),       # T6
    datetime(2024, 12, 30, 17, 16),       # T7
    datetime(2024, 12, 31, 17, 30)]       # T8

# Rows for evaporation correction
no2_rows = ['H10', 'H11', 'H12']
no3_rows = ['H07', 'H08', 'H09']

#######################################################################################

# Concentration
chl0_times = [0]
for i in range(1, len(chl0_datetime_array)):
    time_diff = chl0_datetime_array[i] - chl0_datetime_array[0]
    chl0_times.append(time_diff.total_seconds() / 3600)

no2_chl0_conc.columns = chl0_times
no3_chl0_conc.columns = chl0_times
nh4_chl0_conc.columns = chl0_times

chl1_times = [0]
for i in range(1, len(chl1_datetime_array)):
    time_diff = chl1_datetime_array[i] - chl1_datetime_array[0]
    chl1_times.append(time_diff.total_seconds() / 3600)

no2_chl1_conc.columns = chl1_times
no3_chl1_conc.columns = chl1_times
nh4_chl1_conc.columns = chl1_times

no2_chl0_conc.to_csv("concentration/no2_chl0_conc.csv")
no3_chl0_conc.to_csv("concentration/no3_chl0_conc.csv")
nh4_chl0_conc.to_csv("concentration/nh4_chl0_conc.csv")
no2_chl1_conc.to_csv("concentration/no2_chl1_conc.csv")
no3_chl1_conc.to_csv("concentration/no3_chl1_conc.csv")
nh4_chl1_conc.to_csv("concentration/nh4_chl1_conc.csv")


# Evaporation Correction
norm_no2_chl0 = no2_chl0_conc.loc[no2_rows].mean(axis=0) / no2_chl0_conc.loc[no2_rows].mean(axis=0).iloc[0]
norm_no2_chl1 = no2_chl1_conc.loc[no2_rows].mean(axis=0) / no2_chl1_conc.loc[no2_rows].mean(axis=0).iloc[0]
norm_no3_chl0 = no3_chl0_conc.loc[no3_rows].mean(axis=0) / no3_chl0_conc.loc[no3_rows].mean(axis=0).iloc[0]
norm_no3_chl1 = no3_chl1_conc.loc[no3_rows].mean(axis=0) / no3_chl1_conc.loc[no3_rows].mean(axis=0).iloc[0]
norm_nh4_chl0 = (norm_no2_chl0 + norm_no3_chl0) / 2
norm_nh4_chl1 = (norm_no2_chl1 + norm_no3_chl1) / 2

no2_chl0_evap = no2_chl0_conc.div(norm_no2_chl0.values, axis=1)
no2_chl1_evap = no2_chl1_conc.div(norm_no2_chl1.values, axis=1)
no3_chl0_evap = no3_chl0_conc.div(norm_no3_chl0.values, axis=1)
no3_chl1_evap = no3_chl1_conc.div(norm_no3_chl1.values, axis=1)
nh4_chl0_evap = nh4_chl0_conc.div(norm_nh4_chl0.values, axis=1)
nh4_chl1_evap = nh4_chl1_conc.div(norm_nh4_chl1.values, axis=1)

no2_chl0_evap.to_csv("concentration/no2_chl0_evap.csv")
no2_chl1_evap.to_csv("concentration/no2_chl1_evap.csv")
no3_chl0_evap.to_csv("concentration/no3_chl0_evap.csv")
no3_chl1_evap.to_csv("concentration/no3_chl1_evap.csv")
nh4_chl0_evap.to_csv("concentration/nh4_chl0_evap.csv")
nh4_chl1_evap.to_csv("concentration/nh4_chl1_evap.csv")

# NO3 and NO2 consumptions
no3_chl0_cons = no3_chl0_evap.iloc[:, 0].values.reshape(-1, 1) - no3_chl0_evap
no2_chl0_cons = no2_chl0_evap.iloc[:, 0].values.reshape(-1, 1) - no2_chl0_evap + no3_chl0_cons
no3_chl1_cons = no3_chl1_evap.iloc[:, 0].values.reshape(-1, 1) - no3_chl1_evap
no2_chl1_cons = no2_chl1_evap.iloc[:, 0].values.reshape(-1, 1) - no2_chl1_evap + no3_chl1_cons

no3_chl0_cons.to_csv("concentration/no3_chl0_cons.csv")
no2_chl0_cons.to_csv("concentration/no2_chl0_cons.csv")
no3_chl1_cons.to_csv("concentration/no3_chl1_cons.csv")
no2_chl1_cons.to_csv("concentration/no2_chl1_cons.csv")