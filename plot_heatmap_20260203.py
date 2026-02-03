import pandas as pd

exp_num = 4.2
ids = ['batch1', 'batch2', 'batch3', 'batch4', 'batch5']

for id in ids[0:1]:
    fitting_results = pd.read_csv(f"fitting_results/{exp_num}.{id}_linear_regression_results.csv", index_col=0)
    meta_data = pd.read_csv(f"absorbances/{exp_num}.{id}_samples_metadata.csv", index_col=0)
    # get mean values with standard deviation for rows whose values in meta_data are the same
    combined = fitting_results.join(meta_data)
    mean_and_std = combined.groupby(['Nitrite_input', 'Nitrate_input', 'Chloramphenicol']).agg(['mean', 'std'])
    print(mean_and_std.loc['A01'][('NO2_slope', 'mean')])