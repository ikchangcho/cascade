import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to create pivot table
def create_pivot_table(df, chl, column_name):
    return df.query(f'Chloramphenicol == {chl}').pivot(
        index='Nitrite_input', 
        columns='Nitrate_input', 
        values=column_name
    ).sort_index(ascending=False).sort_index(axis=1, ascending=False)

exp_num = 4.2
ids = ['batch1', 'batch2', 'batch3', 'batch4', 'batch5']

for id in ids[0:5]:
    fitting_results = pd.read_csv(f"fitting_results/{exp_num}.{id}_linear_regression_results.csv", index_col=0)
    meta_df = pd.read_csv(f"absorbances/{exp_num}.{id}_samples_metadata.csv", index_col=0)
    fitting_results = fitting_results.join(meta_df)
    fitting_results = fitting_results.drop(['A01', 'A02', 'A03', 'E01', 'E02', 'E03'], errors='ignore')
    
    # Group by and calculate mean/std
    groupby_cols = ['Nitrite_input', 'Nitrate_input', 'Chloramphenicol']
    drop_cols = ['Sample_type', 'NO2 Intercept', 'NO3 Intercept']
    
    mean_df = fitting_results.drop(drop_cols, axis=1).groupby(groupby_cols, as_index=False).mean()
    var_df = fitting_results.drop(drop_cols, axis=1).groupby(groupby_cols, as_index=False).var()
    mean_df.to_csv(f"fitting_results/{exp_num}.{id}_regression_results_mean.csv", index=False)
    var_df.to_csv(f"fitting_results/{exp_num}.{id}_regression_results_var.csv", index=False)
    
    # no3_rate_mean_chl0 = -create_pivot_table(mean_df, 0, 'NO3 Slope') * 24  # convert to per day
    # no3_rate_var_chl0 = create_pivot_table(var_df, 0, 'NO3 Slope') * (24 ** 2)  # convert to per day squared
    # no3_rate_std_chl0 = no3_rate_var_chl0.pow(0.5)
    # no3_rate_annot_chl0 = no3_rate_mean_chl0.round(2).astype(str) + "\n±" + no3_rate_std_chl0.round(2).astype(str)

    # no3_rate_mean_chl1 = -create_pivot_table(mean_df, 1, 'NO3 Slope') * 24  # convert to per day
    # no3_rate_var_chl1 = create_pivot_table(var_df, 1, 'NO3 Slope') * (24 ** 2)  # convert to per day squared
    # no3_rate_std_chl1 = no3_rate_var_chl1.pow(0.5)
    # no3_rate_annot_chl1 = no3_rate_mean_chl1.round(2).astype(str) + "\n±" + no3_rate_std_chl1.round(2).astype(str)

    # no2_rate_mean_chl0 = -create_pivot_table(mean_df, 0, 'NO2 Slope') * 24  + no3_rate_mean_chl0
    # no2_rate_var_chl0 = create_pivot_table(var_df, 0, 'NO2 Slope') * (24 ** 2) + no3_rate_var_chl0
    # no2_rate_std_chl0 = no2_rate_var_chl0.pow(0.5)
    # no2_rate_annot_chl0 = no2_rate_mean_chl0.round(2).astype(str) + "\n±" + no2_rate_std_chl0.round(2).astype(str)
    
    # no2_rate_mean_chl1 = -create_pivot_table(mean_df, 1, 'NO2 Slope') * 24 + no3_rate_mean_chl1
    # no2_rate_var_chl1 = create_pivot_table(var_df, 1, 'NO2 Slope') * (24 ** 2) + no3_rate_var_chl1
    # no2_rate_std_chl1 = no2_rate_var_chl1.pow(0.5)
    # no2_rate_annot_chl1 = no2_rate_mean_chl1.round(2).astype(str) + "\n±" + no2_rate_std_chl1.round(2).astype(str)

    # fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # heatmap_configs = [
    #     (no3_rate_mean_chl1, no3_rate_annot_chl1, 'NO3 Rate (CHL+)', axes[0, 1]),
    #     (no3_rate_mean_chl0, no3_rate_annot_chl0, 'NO3 Rate (CHL-)', axes[1, 1]),
    #     (no2_rate_mean_chl1, no2_rate_annot_chl1, 'NO2 Rate (CHL+)', axes[0, 0]),
    #     (no2_rate_mean_chl0, no2_rate_annot_chl0, 'NO2 Rate (CHL-)', axes[1, 0])
    # ]
    
    # for data, annot, title, ax in heatmap_configs:
    #     sns.heatmap(data, annot=annot, fmt='', cmap='binary', ax=ax, 
    #                 cbar_kws={'label': 'Rate (per day)'})
    #     ax.set_title(title)
    #     ax.set_xlabel('Nitrate Input (mM)')
    #     ax.set_ylabel('Nitrite Input (mM)')
    
    # plt.suptitle(f'{id} Nitrite and Nitrate Initial Reduction Rates', fontsize=16)
    # plt.savefig(f'plots/{exp_num}.{id}_heatmap_initial_rates.png', dpi=300, bbox_inches='tight')
    # print(f'Saved plots/{exp_num}.{id}_heatmap_initial_rates.png')
    # plt.close()
