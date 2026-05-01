import numpy as np
import pandas as pd

input_dir = 'concentrations'
def load_csv(ids, wells):
    data_dict = {}
    for id in ids:
        data_dict[id] = {}
        no3_conc_df = pd.read_csv(f'{input_dir}/{id}_no3_conc.csv', index_col=0)
        no2_conc_df = pd.read_csv(f'{input_dir}/{id}_no2_conc.csv', index_col=0)
        no3_cons_df = pd.read_csv(f'{input_dir}/{id}_no3_cons.csv', index_col=0)
        no2_cons_df = pd.read_csv(f'{input_dir}/{id}_no2_cons.csv', index_col=0)
        for well in wells:
            well_data_df = pd.concat([no3_conc_df.loc[[well]], 
                                     no2_conc_df.loc[[well]], 
                                     no3_cons_df.loc[[well]], 
                                     no2_cons_df.loc[[well]]], axis=0)
            well_data_df.index = ['no3_conc', 'no2_conc', 'no3_cons', 'no2_cons']
            data_dict[id][well] = well_data_df
    
    return data_dict

ids = ['4.2.batch1', '4.2.batch2', '4.2.batch3']
wells = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
data_dict = load_csv(ids, wells)
example = data_dict['4.2.batch1']['A01']
print(example)
print(example.index)
print(example.columns)
