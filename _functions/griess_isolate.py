# Karna Gowda
# 0.0.1

import bmgdata as bd
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

def plot_griess_fit_mixed(meta_fn,no2_fn=None,no2_540_fn=None,no2_900_fn=None,no2no3_fn=None,no2no3_540_fn=None,no2no3_900_fn=None):  #KC added 08/06/2021 Ik edited 02/17/2025
    meta = pd.read_csv(meta_fn,index_col=0).dropna(how='all')
    meta = meta[['NO2','NO3']] #keep only the NO2/NO3 concentration columns
    no2 = read_griess(meta_fn,data_fn = no2_fn,data_540_fn=no2_540_fn,data_900_fn=no2_900_fn)
    no2no3 = read_griess(meta_fn,data_fn = no2no3_fn,data_540_fn=no2no3_540_fn,data_900_fn=no2no3_900_fn)
    
    #remove nan values
    nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    meta = meta.drop(nan_idx)
    no2 = no2.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)
    
    #subtract blank values
    blank_idx = meta.index[(meta['NO2']==0) & (meta['NO3']==0)].tolist()
    ###print(blank_idx)
    no2_blank = no2.loc[blank_idx].median()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2 = no2 - no2_blank
    no2no3 = no2no3 - no2no3_blank

    #identify pure no2 and pure no3 samples
    no2_std_idx = meta.index[(meta['NO2']>0)].tolist() #& (meta['NO3']==0)].tolist()
    no3_std_idx = meta.index[(meta['NO2']==0) & (meta['NO3']>0)].tolist()

    #no2 standard curve, griess measurement
    y = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)
    plt.scatter(x,y,label = 'Griess measurement')
    x_vals = x
   
    #Fit a quadratic model to the nitrite concentration
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    g_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    ###print(x_vals)
    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num = 50)
    plt.plot(x_vals_smooth, reg.intercept_[0] + x_vals_smooth*reg.coef_[0][0] + x_vals_smooth*x_vals_smooth*reg.coef_[0][1], label =  'griess fit')
    plt.xlabel('NO2 [mM]')
    plt.ylabel('Griess measurement')
    plt.title('Griess standard curve, y='+str(round(reg.intercept_[0],2))+ '+'+str(round(reg.coef_[0][0],2))+'x+'+str(round(reg.coef_[0][1],2))+'x^2')
    plt.legend()
    #plt.show()
   
    #no2 and no3 standard curves, vcl3 measurement
    y1 = no2no3.loc[no2_std_idx].values
    y2 = no2no3.loc[no3_std_idx].values
    x1 = meta.loc[no2_std_idx].values
    x2 = meta.loc[no3_std_idx].values
    #plt.plot(x1,y1,label = 'vcl3 no2 measurement')
    #plt.plot(x2,y2,label = 'vcl3 no3 measurement')
    #x1_vals = x1
    #x2_vals = x2

    
    #Fit a quadratic model to the sum of nitrate and nitrite concentrations
    x = np.append(np.sum(x1,axis=1),np.sum(x2,axis=1)).reshape(-1,1)
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    y = np.append(y1,y2).reshape(-1,1)
    reg = LinearRegression().fit(x, y)
    v_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    #plt.plot(x1_vals, reg.intercept_[0] + x_vals*reg.coef_[0][0] + x_vals*x_vals*reg.coef_[0][1], label = 'vcl3 fit')
    #plt.show()
    #fit = [[no2_blank,no2no3_blank], g_fit, v_fit]
    return

def fit_griess_mixed(meta_fn,no2_fn=None,no2_540_fn=None,no2_900_fn=None,no2no3_fn=None,no2no3_540_fn=None,no2no3_900_fn=None):  # KC added 08/06/2021 Ik edited 02/17/2025
    meta = pd.read_csv(meta_fn,index_col=0).dropna(how='all')
    meta = meta[['NO2','NO3']] #keep only the NO2/NO3 concentration columns
    no2 = read_griess(meta_fn,data_fn = no2_fn,data_540_fn=no2_540_fn,data_900_fn=no2_900_fn)
    no2no3 = read_griess(meta_fn,data_fn = no2no3_fn,data_540_fn=no2no3_540_fn,data_900_fn=no2no3_900_fn)
    
    #remove nan values
    nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    meta = meta.drop(nan_idx)
    no2 = no2.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)
    
    #subtract blank values
    blank_idx = meta.index[(meta['NO2']==0) & (meta['NO3']==0)].tolist()
    no2_blank = no2.loc[blank_idx].median()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2 = no2 - no2_blank
    no2no3 = no2no3 - no2no3_blank

    #identify pure no2 and pure no3 samples
    no2_std_idx = meta.index[(meta['NO2']>0)].tolist() #& (meta['NO3']==1.75)].tolist()
    no3_std_idx = meta.index[(meta['NO2']==0) & (meta['NO3']>0)].tolist()
    #no3_2_std_idx =  meta.index[(meta['NO2']==0) & (meta['NO3']==1.75)].tolist()
    #no2 standard curve, griess measurement
    y = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)
    #plt.scatter(x,y,label = 'griess measurement')
    #x_vals = x
   
    #Fit a quadratic model to the nitrite concentration
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    g_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    ###print(x_vals)
    #plt.plot(x_vals, reg.intercept_[0] + x_vals*reg.coef_[0][0] + x_vals*x_vals*reg.coef_[0][1], label =  'griess fit')
    
   
    #no2 and no3 standard curves, vcl3 measurement
    ###print(no2no3.loc[no3_2_std_idx].values)
    y1 = no2no3.loc[no2_std_idx].values #- np.mean(no2no3.loc[no3_2_std_idx].values)
    y2 = no2no3.loc[no3_std_idx].values
    ###print('Warning, hardcoded below')
    x1 = meta.loc[no2_std_idx].values#.T#[0] #- 1.75/2.0 #hard coded, careful of this
    x2 = meta.loc[no3_std_idx].values
    #x1 = x1.T[0]
    ###print(y1)
    ###print(x1)
    
    #Fit a quadratic model to the sum of nitrate and nitrite concentrations
    x = np.append(np.sum(x1,axis=1),np.sum(x2,axis=1)).reshape(-1,1)
    ##print(x1)
    ##print(x2)
    #error
    x_vals1 = np.sum(x1,axis=1)
    ##print(x_vals1)
    x_vals2 = np.sum(x2,axis=1)
    ##print(x_vals2)
    x_vals = x
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    y = np.append(y1,y2).reshape(-1,1)
    #plt.scatter(x_vals,y,label = 'VCl3 measurement')
    reg = LinearRegression().fit(x, y)
    v_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num = 50)
    fit = [[no2_blank,no2no3_blank], g_fit, v_fit]
    return fit

def find_outliers(data):  # Modified by Kiseok Lee 230622
    val_above = 1.5
    # print("returns true for elements more than " + str(val_above) + " interquartile ranges above the upper quartile")
    q75 = np.quantile(data, 0.75)
    q25 = np.quantile(data, 0.25)
    is_outlier = data > q75 + (q75 - q25) * val_above
    df_outliers = is_outlier  # dataframe
    # Need to make sure that it has less than 3 Trues for each row
    # Find the rows that have more than 3 True values.
    # Then, change it to have True in positions of the 2 of the lowest values
    df_outliers[df_outliers.sum(axis=1) >= 3] = data[df_outliers.sum(axis=1) >= 3].rank(axis=1) > 2
    return df_outliers


# 230622 What if I got outliers per well? ->
# def find_outliers_per_well(data):
#     # Apply find_outliers_per_well function to every row
#     outliers = data.apply(find_outliers, axis=1)
#     return outliers


def remove_bubbles(data_in, data_540, data_900):
    # find wells with bubbles in the no2no3 measurements

    bub_in = find_outliers(data_in['900'])
    bub_900 = find_outliers(data_900)

    # replace values at 540 nm
    data_out = data_in.copy()
    for idx in bub_in.index[bub_in == True].tolist():
        replacement = data_540.loc[idx]
        data_out['540'].loc[idx] = np.median(
            replacement[~bub_900.loc[idx]])  # replace with median of values that don't have bubbles
    return data_out


def read_griess(meta_fn, data_fn=None, data_540_fn=None, data_900_fn=None):
    '''
    Bubbles correction and take average on 540 nm absorbance data
    :param meta_fn:
    :param data_fn:
    :param data_540_fn:
    :param data_900_fn:
    :return: N x 2 dataframe
    '''
    # returns absorbance data at 540 nm
    # corrects for bubbles if well scan measurements are provided
    # there are three valid cases that this function works for:
    # 1) A file name is provided for data_fn. Usually this is for importing no2 data.
    # 2) File names are provided for data_fn and associated well scan files data_540_fn and data_900_fn. Usually this is for importing old no2no3 measurements using both endpoint and well scan data.
    # 3) File names are provided for only well scan files data_540_fn and data_900_fn. This is the prefered measurement data for future experiments.

    meta = pd.read_csv(meta_fn, index_col=0).dropna(how='all')  # import metadata

    if (data_540_fn != None) & (data_900_fn != None):
        data_540 = bd.read_abs_wellscan(data_540_fn)  # import well scan data (540 nm)
        data_540 = data_540[data_540.index.isin(meta.index)]  # remove indices with no values in the metadata
        data_900 = bd.read_abs_wellscan(data_900_fn)  # import well scan data (900 nm)
        data_900 = data_900[data_900.index.isin(meta.index)]  # remove indices with no values in the metadata

    if data_fn != None:  # case 1 or 2
        data_out = bd.read_abs_endpoint(data_fn)  # import data
        data_out = data_out[data_out.index.isin(meta.index)]  # remove indices with no values in the metadata
        # correct for bubbles
        if data_540_fn != None:  # case 2
            data_out = remove_bubbles(data_out, data_540, data_900)  # correct for bubbles in measurement.
        data_out = data_out['540']  # pick out 540 nm measurement
    elif (data_540_fn != None) & (data_900_fn != None):  # case 3
        bub_900 = find_outliers(data_900)
        for row in data_540.index:
            if sum(np.isnan(bub_900.loc[row])) < 4:
                data_540.loc[row, bub_900.loc[row]] = np.nan  # Set bubbles to NaN
            else:
                data_540.loc[row] = data_540.loc[row] - data_900.loc[row]
        data_out = data_540.median(axis=1)
        data_out = data_out.rename("540")

    return data_out


def plot_griess_fit(meta_fn, no2_fn=None, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None, no2no3_540_fn=None,
                    no2no3_900_fn=None):  # KC added 08/06/2021
    meta = pd.read_csv(meta_fn, index_col=0).dropna(how='all')
    meta = meta[['NO2', 'NO3']]  # keep only the NO2/NO3 concentration columns
    # no2 = read_griess(meta_fn,no2_fn)
    no2 = read_griess(meta_fn, data_fn=no2_fn, data_540_fn=no2_540_fn, data_900_fn=no2_900_fn)
    no2no3 = read_griess(meta_fn, data_fn=no2no3_fn, data_540_fn=no2no3_540_fn, data_900_fn=no2no3_900_fn)
    # print(no2)
    # print(no2no3)

    # remove nan values
    # (this is the old code) nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    # new (1/31/22 Kiseok)
    nan_idx = no2[no2.apply(np.isnan)].index.union(no2no3[no2no3.apply(np.isnan)].index)
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    no2 = no2.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)

    # subtract blank values
    blank_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] == 0)].tolist()
    # print(blank_idx)
    no2_blank = no2.loc[blank_idx].median()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2 = no2 - no2_blank
    no2no3 = no2no3 - no2no3_blank

    # identify pure no2 and pure no3 samples
    no2_std_idx = meta.index[meta['NO2'] > 0].tolist()
    no3_std_idx = meta.index[meta['NO3'] > 0].tolist()

    # no2 standard curve, griess measurement
    y = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)
    plt.scatter(x, y, label='Measurement of NO2 standards')
    x_vals = x

    # Fit a quadratic model to the nitrite concentration
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    g_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    # print(x_vals)
    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num=50)
    plt.plot(x_vals_smooth,
             reg.intercept_[0] + x_vals_smooth * reg.coef_[0][0] + x_vals_smooth * x_vals_smooth * reg.coef_[0][1],
             label='quadratic model fit')
    plt.xlabel('NO2 [mM]')
    plt.ylabel('OD measurement (540nm)')
    plt.title('Griess assay standard curve (before VCl treatment: Nitrite), y=' + str(
        round(reg.intercept_[0], 2)) + '+' + str(round(reg.coef_[0][0], 2)) + 'x+' + str(
        round(reg.coef_[0][1], 2)) + 'x^2' + ', Adjusted R^2 =' + str(
        np.round(1 - (1 - reg.score(x, y)) * (len(y) - 1) / (len(y) - x.shape[1] - 1), 3)))
    plt.legend()
    # plt.show()

    # no2 and no3 standard curves, vcl3 measurement
    y1 = no2no3.loc[no2_std_idx].values
    y2 = no2no3.loc[no3_std_idx].values
    x1 = meta.loc[no2_std_idx].values
    x2 = meta.loc[no3_std_idx].values
    # plt.plot(x1,y1,label = 'vcl3 no2 measurement')
    # plt.plot(x2,y2,label = 'vcl3 no3 measurement')
    # x1_vals = x1
    # x2_vals = x2

    # Fit a quadratic model to the sum of nitrate and nitrite concentrations
    x = np.append(np.sum(x1, axis=1), np.sum(x2, axis=1)).reshape(-1, 1)
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    y = np.append(y1, y2).reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    v_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    # plt.plot(x1_vals, reg.intercept_[0] + x_vals*reg.coef_[0][0] + x_vals*x_vals*reg.coef_[0][1], label = 'vcl3 fit')
    # plt.show()
    # fit = [[no2_blank,no2no3_blank], g_fit, v_fit]
    return


def plot_vcl_fit(meta_fn, no2_fn=None, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None, no2no3_540_fn=None,
                 no2no3_900_fn=None):  # KC added 08/06/2021
    meta = pd.read_csv(meta_fn, index_col=0).dropna(how='all')
    meta = meta[['NO2', 'NO3']]  # keep only the NO2/NO3 concentration columns
    # no2 = read_griess(meta_fn,no2_fn) # this does not correct for bouble in no2 data
    no2no3 = read_griess(meta_fn, data_540_fn=no2no3_540_fn, data_900_fn=no2no3_900_fn)

    # remove nan values
    # old code: nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    # new (1/31/22 Kiseok)
    nan_idx = no2no3[no2no3.apply(np.isnan)].index
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)

    # subtract blank values
    blank_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] == 0)].tolist()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2no3 = no2no3 - no2no3_blank

    # identify pure no2 and pure no3 samples
    # no2_std_idx = meta.index[(meta['NO2']>0) & (meta['NO3']==0)].tolist()
    # no3_std_idx = meta.index[(meta['NO2']==0) & (meta['NO3']>0)].tolist()

    # identify samples with no2 or no3
    no2_std_idx = meta.index[(meta['NO2'] > 0)].tolist()
    no3_std_idx = meta.index[(meta['NO3'] > 0)].tolist()

    # no2 and no3 standard curves, vcl3 measurement
    y1 = no2no3.loc[no2_std_idx].values
    y2 = no2no3.loc[no3_std_idx].values
    x1 = meta.loc[no2_std_idx].values
    x2 = meta.loc[no3_std_idx].values

    # Fit a quadratic model to the sum of nitrate and nitrite concentrations
    x = np.append(np.sum(x1, axis=1), np.sum(x2, axis=1)).reshape(-1, 1)
    # print(x1)
    # print(x2)
    # error
    x_vals1 = np.sum(x1, axis=1)
    # print(x_vals1)
    x_vals2 = np.sum(x2, axis=1)
    # print(x_vals2)
    x_vals = x
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    y = np.append(y1, y2).reshape(-1, 1)
    # plt.scatter(x_vals,y,label = 'VCl3 measurement')
    plt.scatter(x_vals1, y1, label='Measurement NO2 standards')
    plt.scatter(x_vals2, y2, label='Measurement of reduced NO3 standards')
    reg = LinearRegression().fit(x, y)
    v_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num=50)
    plt.plot(x_vals_smooth,
             (reg.intercept_[0] + x_vals_smooth * reg.coef_[0][0] + x_vals_smooth * x_vals_smooth * reg.coef_[0][1]),
             label='quadratic model fit')

    plt.legend()
    plt.xlabel('NO2 + NO3 [mM]')
    plt.ylabel('OD measurement (540nm)')
    # plt.title('VCl3 standard curve')
    plt.title('Griess assay standard curve (after VCl treatment: Nitrate reduction), y=' + str(
        round(reg.intercept_[0], 2)) + '+' + str(round(reg.coef_[0][0], 2)) + 'x+' + str(
        round(reg.coef_[0][1], 2)) + 'x^2' + ', Adjusted R^2 =' + str(
        np.round(1 - (1 - reg.score(x, y)) * (len(y) - 1) / (len(y) - x.shape[1] - 1), 3)))

    ## Let's get the p-value (Source: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression)
    params = np.append(reg.intercept_, reg.coef_)
    predictions = reg.predict(x)
    newx = pd.DataFrame({"Constant": np.ones(len(x))}).join(pd.DataFrame(x))
    # print(newx)

    MSE = (sum((y - predictions) ** 2)) / (len(newx) - len(newx.columns))
    # print(MSE)
    var_b = MSE * (np.linalg.inv(np.dot(newx.T, newx)).diagonal())
    # print("var_b",var_b)
    sd_b = np.sqrt(var_b)
    # print("sd_b",sd_b)
    ts_b = params / sd_b
    # print("ts_b",ts_b)

    # print("len(newx[0])",len(newx[0]))
    # print("len(newx)-len(newx[0])",len(newx)-len(newx[0]))
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (y.shape[0] - x.shape[1]))) for i in ts_b]
    # print("p-values",p_values)

    sd_b = np.round(sd_b, 4)
    ts_b = np.round(ts_b, 4)
    p_values = np.round(p_values, 4)
    params = np.round(params, 4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["P-Values"] = [params, sd_b, ts_b,
                                                                                             p_values]

    print(
        "\n\n (From sklearn LinearRegression) Manual calculation of p-value. Here we use sklearn LinearRegression package")
    print(myDF3)

    # Getting summary from statsmodel package
    print("\n\n (From statsmodel package) Coefficients are a bit different from the sklearn LinearRegression package")
    est = sm.OLS(y, x)
    est2 = est.fit()
    print(est2.summary())

    ## Let's get the R squre value
    # compute with sklearn linear_model, although could not find any function to compute adjusted-r-square directly from documentation
    print("\n\n (R square and adjusted R square value)")
    print(reg.score(x, y), 1 - (1 - reg.score(x, y)) * (len(y) - 1) / (len(y) - x.shape[1] - 1))
    print("\n")

    return


def plot_vcl_fit_only_nitrate_standards(meta_fn, no2_fn, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None,
                                        no2no3_540_fn=None, no2no3_900_fn=None):  # KC added 08/06/2021
    meta = pd.read_csv(meta_fn, index_col=0).dropna(how='all')
    meta = meta[['NO2', 'NO3']]  # keep only the NO2/NO3 concentration columns
    # no2 = read_griess(meta_fn,no2_fn) # this does not correct for bouble in no2 data
    no2no3 = read_griess(meta_fn, data_540_fn=no2no3_540_fn, data_900_fn=no2no3_900_fn)

    # remove nan values
    # old code: nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    # new (1/31/22 Kiseok)
    nan_idx = no2no3[no2no3.apply(np.isnan)].index
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)

    # subtract blank values
    blank_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] == 0)].tolist()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2no3 = no2no3 - no2no3_blank

    # identify pure no2 and pure no3 samples
    # no2_std_idx = meta.index[(meta['NO2']>0) & (meta['NO3']==0)].tolist()
    no3_std_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] > 0)].tolist()

    # no2 and no3 standard curves, vcl3 measurement
    #     y1 = no2no3.loc[no2_std_idx].values
    y2 = no2no3.loc[no3_std_idx].values
    #     x1 = meta.loc[no2_std_idx].values
    x2 = meta.loc[no3_std_idx].values

    # Fit a quadratic model to the sum of nitrate and nitrite concentrations
    x = np.sum(x2, axis=1).reshape(-1, 1)
    # error
    #     x_vals1 = np.sum(x1,axis=1)
    x_vals2 = np.sum(x2, axis=1)
    # print(x_vals2)
    x_vals = x
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    # print(y2)
    y = y2.reshape(-1, 1)
    # print(x)
    # print(y)
    # plt.scatter(x_vals,y,label = 'VCl3 measurement')
    plt.scatter(x_vals2, y2, label='Measurement of reduced NO3 standards')
    reg = LinearRegression().fit(x, y)

    v_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num=50)
    plt.plot(x_vals_smooth,
             (reg.intercept_[0] + x_vals_smooth * reg.coef_[0][0] + x_vals_smooth * x_vals_smooth * reg.coef_[0][1]),
             label='quadratic model fit')

    plt.legend()
    plt.xlabel('NO3 [mM]')
    plt.ylabel('OD measurement (540nm)')
    # plt.title('VCl3 standard curve')
    plt.title('Griess assay standard curve (after VCl treatment: only reduced nitrate), y=' + str(
        round(reg.intercept_[0], 2)) + '+' + str(round(reg.coef_[0][0], 2)) + 'x+' + str(
        round(reg.coef_[0][1], 2)) + 'x^2' + ', Adjusted R^2 =' + str(
        np.round(1 - (1 - reg.score(x, y)) * (len(y) - 1) / (len(y) - x.shape[1] - 1), 3)))

    ## Let's get the p-value (Source: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression)
    params = np.append(reg.intercept_, reg.coef_)
    predictions = reg.predict(x)
    newx = pd.DataFrame({"Constant": np.ones(len(x))}).join(pd.DataFrame(x))
    # print(newx)

    MSE = (sum((y - predictions) ** 2)) / (len(newx) - len(newx.columns))
    # print(MSE)
    var_b = MSE * (np.linalg.inv(np.dot(newx.T, newx)).diagonal())
    # print("var_b",var_b)
    sd_b = np.sqrt(var_b)
    # print("sd_b",sd_b)
    ts_b = params / sd_b
    # print("ts_b",ts_b)

    # print("len(newx[0])",len(newx[0]))
    # print("len(newx)-len(newx[0])",len(newx)-len(newx[0]))
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (y.shape[0] - x.shape[1]))) for i in ts_b]
    # print("p-values",p_values)

    sd_b = np.round(sd_b, 4)
    ts_b = np.round(ts_b, 4)
    p_values = np.round(p_values, 4)
    params = np.round(params, 4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["P-Values"] = [params, sd_b, ts_b,
                                                                                             p_values]

    print(
        "\n\n (From sklearn LinearRegression) Manual calculation of p-value. Here we use sklearn LinearRegression package")
    print(myDF3)

    # Getting summary from statsmodel package
    print("\n\n (From statsmodel package) Coefficients are a bit different from the sklearn LinearRegression package")
    est = sm.OLS(y, x)
    est2 = est.fit()
    print(est2.summary())

    ## Let's get the R squre value
    # compute with sklearn linear_model, although could not find any function to compute adjusted-r-square directly from documentation
    print("\n\n (R square and adjusted R square value)")
    print(reg.score(x, y), 1 - (1 - reg.score(x, y)) * (len(y) - 1) / (len(y) - x.shape[1] - 1))
    print("\n")

    return


# only use nitrate standards
def plot_no3_fit(meta_fn, no2_fn=None, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None, no2no3_540_fn=None,
                 no2no3_900_fn=None):  # Kiseok added 10/06/2021
    meta = pd.read_csv(meta_fn, index_col=0).dropna(how='all')
    meta = meta[['NO2', 'NO3']]  # keep only the NO2/NO3 concentration columns
    # no2 = read_griess(meta_fn,no2_fn) # this does not correct for bouble in no2 data
    no2no3 = read_griess(meta_fn, data_540_fn=no2no3_540_fn, data_900_fn=no2no3_900_fn)

    # remove nan values
    # old code: nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    # new (1/31/22 Kiseok)
    nan_idx = no2no3[no2no3.apply(np.isnan)].index
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)

    # subtract blank values
    blank_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] == 0)].tolist()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2no3 = no2no3 - no2no3_blank

    # identify pure no2 and pure no3 samples
    # no2_std_idx = meta.index[(meta['NO2']>0) & (meta['NO3']==0)].tolist()
    no3_std_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] > 0)].tolist()

    # no3 standard curve, griess measurement
    y = no2no3.loc[no3_std_idx].values.reshape(-1, 1)
    x = meta['NO3'].loc[no3_std_idx].values.reshape(-1, 1)
    plt.scatter(x, y, label='Measurement of NO3 standards')
    x_vals = x

    # Fit a quadratic model to the nitrite concentration
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    no3_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    # print(x_vals)
    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num=50)
    plt.plot(x_vals_smooth,
             reg.intercept_[0] + x_vals_smooth * reg.coef_[0][0] + x_vals_smooth * x_vals_smooth * reg.coef_[0][1],
             label='quadratic model fit')
    plt.xlabel('NO3 [mM]')
    plt.ylabel('OD measurement (540nm)')
    plt.title('Griess assay standard curve (after VCl treatment, fitted using only NO3 standards), y=' + str(
        round(reg.intercept_[0], 2)) + '+' + str(round(reg.coef_[0][0], 2)) + 'x+' + str(
        round(reg.coef_[0][1], 2)) + 'x^2' + ', Adjusted R^2 =' + str(
        np.round(1 - (1 - reg.score(x, y)) * (len(y) - 1) / (len(y) - x.shape[1] - 1), 3)))
    plt.legend()

    ## Let's get the p-value (Source: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression)
    params = np.append(reg.intercept_, reg.coef_)
    predictions = reg.predict(x)
    newx = pd.DataFrame({"Constant": np.ones(len(x))}).join(pd.DataFrame(x))
    # print(newx)

    MSE = (sum((y - predictions) ** 2)) / (len(newx) - len(newx.columns))
    # print(MSE)
    var_b = MSE * (np.linalg.inv(np.dot(newx.T, newx)).diagonal())
    # print("var_b",var_b)
    sd_b = np.sqrt(var_b)
    # print("sd_b",sd_b)
    ts_b = params / sd_b
    # print("ts_b",ts_b)

    # print("len(newx[0])",len(newx[0]))
    # print("len(newx)-len(newx[0])",len(newx)-len(newx[0]))
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (y.shape[0] - x.shape[1]))) for i in ts_b]
    # print("p-values",p_values)

    sd_b = np.round(sd_b, 4)
    ts_b = np.round(ts_b, 4)
    p_values = np.round(p_values, 4)
    params = np.round(params, 4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["P-Values"] = [params, sd_b, ts_b,
                                                                                             p_values]

    print(
        "\n\n (From sklearn LinearRegression) Manual calculation of p-value. Here we use sklearn LinearRegression package")
    print(myDF3)

    # Getting summary from statsmodel package
    print("\n\n (From statsmodel package) Coefficients are a bit different from the sklearn LinearRegression package")
    est = sm.OLS(y, x)
    est2 = est.fit()
    print(est2.summary())

    ## Let's get the R squre value
    # compute with sklearn linear_model, although could not find any function to compute adjusted-r-square directly from documentation
    print("\n\n (R square and adjusted R square value)")
    print(reg.score(x, y), 1 - (1 - reg.score(x, y)) * (len(y) - 1) / (len(y) - x.shape[1] - 1))
    print("\n")

    return


def plot_mixed_standard_predict(meta_fn, no2_fn, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None, no2no3_540_fn=None,
                                no2no3_900_fn=None):  # KC added 08/06/2021
    meta = pd.read_csv(meta_fn, index_col=0).dropna(how='all')
    meta = meta[['NO2', 'NO3']]  # keep only the NO2/NO3 concentration columns
    no2 = read_griess(meta_fn, data_fn=no2_fn, data_540_fn=no2_540_fn, data_900_fn=no2_900_fn)
    no2no3 = read_griess(meta_fn, data_fn=no2no3_fn, data_540_fn=no2no3_540_fn, data_900_fn=no2no3_900_fn)

    # remove nan values
    # old code: nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    # new (1/31/22 Kiseok)
    nan_idx = no2[no2.apply(np.isnan)].index.union(no2no3[no2no3.apply(np.isnan)].index)
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    no2 = no2.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)

    # subtract blank values
    blank_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] == 0)].tolist()
    no2_blank = no2.loc[blank_idx].median()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2 = no2 - no2_blank
    no2no3 = no2no3 - no2no3_blank

    # identify pure no2 and pure no3 samples
    no2_std_idx = meta.index[(meta['NO2'] > 0) & (meta['NO3'] == 0)].tolist()
    no3_std_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] > 0)].tolist()

    # identify mixed no2 and no3 samples
    no2no3_std_idx = meta.index[(meta['NO2'] > 0) & (meta['NO3'] > 0)].tolist()

    # no2 standard curve, griess measurement
    y = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)
    # plt.scatter(x,y,label = 'griess measurement')
    # x_vals = x

    # Fit a quadratic model to the nitrite concentration
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    g_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    # print(x_vals)
    # plt.plot(x_vals, reg.intercept_[0] + x_vals*reg.coef_[0][0] + x_vals*x_vals*reg.coef_[0][1], label =  'griess fit')

    # no2 in mixed conditions prediction
    x_inf = meta['NO2'].loc[no2no3_std_idx].values.reshape(-1, 1)
    y_measured = no2.loc[no2no3_std_idx].values.reshape(-1, 1)
    x_pred = ((-g_fit[1] + np.sqrt(g_fit[1] ** 2 - 4 * (g_fit[0] - y_measured) * g_fit[2])) / 2 / g_fit[2])
    plt.scatter(x_inf, x_pred, label='NO2')

    # no2 and no3 standard curves, vcl3 measurement
    y1 = no2no3.loc[no2_std_idx].values
    y2 = no2no3.loc[no3_std_idx].values
    x1 = meta.loc[no2_std_idx].values
    x2 = meta.loc[no3_std_idx].values

    x_inf_NO3 = meta['NO3'].loc[no2no3_std_idx].values.reshape(-1, 1)
    y_measured_NO2NO3 = no2no3.loc[no2no3_std_idx].values
    # print(x_inf_NO2NO3)
    # print(y_measured_NO2NO3)

    # Fit a quadratic model to the sum of nitrate and nitrite concentrations
    x = np.append(np.sum(x1, axis=1), np.sum(x2, axis=1)).reshape(-1, 1)
    # print(x1)
    # print(x2)
    # error
    x_vals1 = np.sum(x1, axis=1)
    # print(x_vals1)
    x_vals2 = np.sum(x2, axis=1)
    # print(x_vals2)
    x_vals = x
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    y = np.append(y1, y2).reshape(-1, 1)
    # plt.scatter(x_vals,y,label = 'VCl3 measurement')
    # plt.scatter(x_vals1,y1,label = 'VCl3 measurement NO2')
    # plt.scatter(x_vals2,y2,label = 'VCl3 measurement reduced NO3')
    reg = LinearRegression().fit(x, y)
    v_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    x_pred_NO3 = ((-v_fit[1] + np.sqrt(v_fit[1] ** 2 - 4 * (v_fit[0] - y_measured_NO2NO3) * v_fit[2])) / 2 / v_fit[
        2]) - x_pred.T

    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num=50)
    # plt.plot(x_vals_smooth, (reg.intercept_[0] + x_vals_smooth*reg.coef_[0][0] + x_vals_smooth*x_vals_smooth*reg.coef_[0][1]), label = 'VCl3 fit')
    plt.plot(x_vals_smooth, x_vals_smooth, 'k--', alpha=0.5)  # perfect prediction
    # print(y_measured_NO2NO3)
    # print(x_pred)
    # print(x_inf_NO3)
    # print(x_pred_NO3)
    plt.scatter(x_inf_NO3, x_pred_NO3, label='NO3', alpha=0.3)
    plt.xlim([0, 1.25])
    plt.legend()
    plt.xlabel('Known NO2- or NO3- concentration of standards [mM]')
    plt.ylabel('Predicted NO2- or NO3- concentration [mM]')
    # plt.title('VCl3 standard curve')
    plt.title('Nitrite, nitrate mixed standard prediction')
    # plt.show()
    # fit = [[no2_blank,no2no3_blank], g_fit, v_fit]
    return


def plot_mixed_standard_predict_with_no3_fit(meta_fn, no2_fn, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None,
                                             no2no3_540_fn=None, no2no3_900_fn=None):  # KC added 08/06/2021
    meta = pd.read_csv(meta_fn, index_col=0).dropna(how='all')
    meta = meta[['NO2', 'NO3']]  # keep only the NO2/NO3 concentration columns
    no2 = read_griess(meta_fn, data_fn=no2_fn, data_540_fn=no2_540_fn, data_900_fn=no2_900_fn)
    no2no3 = read_griess(meta_fn, data_fn=no2no3_fn, data_540_fn=no2no3_540_fn, data_900_fn=no2no3_900_fn)

    # remove nan values
    # old code: nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    # new (1/31/22 Kiseok)
    nan_idx = no2[no2.apply(np.isnan)].index.union(no2no3[no2no3.apply(np.isnan)].index)
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    no2 = no2.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)

    # subtract blank values
    blank_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] == 0)].tolist()
    no2_blank = no2.loc[blank_idx].median()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2 = no2 - no2_blank
    no2no3 = no2no3 - no2no3_blank

    # identify pure no2 and pure no3 samples
    no2_std_idx = meta.index[(meta['NO2'] > 0) & (meta['NO3'] == 0)].tolist()
    no3_std_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] > 0)].tolist()

    # identify mixed no2 and no3 samples
    no2no3_std_idx = meta.index[(meta['NO2'] > 0) & (meta['NO3'] > 0)].tolist()

    # no2 standard curve, griess measurement
    y = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)
    # plt.scatter(x,y,label = 'griess measurement')
    # x_vals = x

    # Fit a quadratic model to the nitrite concentration
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    g_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    # print(x_vals)
    # plt.plot(x_vals, reg.intercept_[0] + x_vals*reg.coef_[0][0] + x_vals*x_vals*reg.coef_[0][1], label =  'griess fit')

    # no2 in mixed conditions prediction
    x_inf = meta['NO2'].loc[no2no3_std_idx].values.reshape(-1, 1)
    y_measured = no2.loc[no2no3_std_idx].values.reshape(-1, 1)
    x_pred = ((-g_fit[1] + np.sqrt(g_fit[1] ** 2 - 4 * (g_fit[0] - y_measured) * g_fit[2])) / 2 / g_fit[2])
    plt.scatter(x_inf, x_pred, label='NO2')

    # no2 and no3 standard curves, vcl3 measurement
    y1 = no2no3.loc[no2_std_idx].values
    y2 = no2no3.loc[no3_std_idx].values
    x1 = meta.loc[no2_std_idx].values
    x2 = meta.loc[no3_std_idx].values

    x_inf_NO3 = meta['NO3'].loc[no2no3_std_idx].values.reshape(-1, 1)
    y_measured_NO2NO3 = no2no3.loc[no2no3_std_idx].values
    # print(x_inf_NO2NO3)
    # print(y_measured_NO2NO3)

    # Fit a quadratic model to the sum of nitrate and nitrite concentrations
    y = no2no3.loc[no3_std_idx].values.reshape(-1, 1)
    x = meta['NO3'].loc[no3_std_idx].values.reshape(-1, 1)
    x_vals = x

    # Fit a quadratic model to the nitrite concentration
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    no3_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    x_pred_NO3 = ((-no3_fit[1] + np.sqrt(no3_fit[1] ** 2 - 4 * (no3_fit[0] - y_measured_NO2NO3) * no3_fit[2])) / 2 /
                  no3_fit[2]) - x_pred.T

    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num=50)
    # plt.plot(x_vals_smooth, (reg.intercept_[0] + x_vals_smooth*reg.coef_[0][0] + x_vals_smooth*x_vals_smooth*reg.coef_[0][1]), label = 'VCl3 fit')
    plt.plot(x_vals_smooth, x_vals_smooth, 'k--', alpha=0.5)  # perfect prediction
    # print(y_measured_NO2NO3)
    # print(x_pred)
    # print(x_inf_NO3)
    # print(x_pred_NO3)
    plt.scatter(x_inf_NO3, x_pred_NO3, label='NO3', alpha=0.3)
    plt.xlim([0, 1.25])
    plt.legend()
    plt.xlabel('Known NO2- or NO3- concentration of standards [mM]')
    plt.ylabel('Predicted NO2- or NO3- concentration [mM]')
    # plt.title('VCl3 standard curve')
    plt.title('Nitrite, nitrate mixed standard prediction using NO3 fit')
    # plt.show()
    return


def plot_nitrite_standard_predict_no2(meta_fn, no2_fn, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None,
                                      no2no3_540_fn=None, no2no3_900_fn=None):  # KS added 11/4/21
    meta = pd.read_csv(meta_fn, index_col=0).dropna(how='all')
    meta = meta[['NO2', 'NO3']]  # keep only the NO2/NO3 concentration columns
    no2 = read_griess(meta_fn, data_fn=no2_fn, data_540_fn=no2_540_fn, data_900_fn=no2_900_fn)
    no2no3 = read_griess(meta_fn, data_fn=no2no3_fn, data_540_fn=no2no3_540_fn, data_900_fn=no2no3_900_fn)

    # remove nan values
    # old code: nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    # new (1/31/22 Kiseok)
    nan_idx = no2[no2.apply(np.isnan)].index.union(no2no3[no2no3.apply(np.isnan)].index)
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    no2 = no2.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)

    # subtract blank values
    blank_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] == 0)].tolist()
    no2_blank = no2.loc[blank_idx].median()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2 = no2 - no2_blank
    no2no3 = no2no3 - no2no3_blank

    # identify pure no2 and pure no3 samples
    no2_std_idx = meta.index[(meta['NO2'] > 0) & (meta['NO3'] == 0)].tolist()
    no3_std_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] > 0)].tolist()

    # merge those two list
    pure_std_idx = no2_std_idx + no3_std_idx

    # no2 standard curve, griess measurement
    y = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)
    # plt.scatter(x,y,label = 'griess measurement')
    # x_vals = x

    # Fit a quadratic model to the nitrite concentration
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    g_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]

    # no2 in pure conditions prediction
    x_inf = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)
    y_measured = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x_pred = ((-g_fit[1] + np.sqrt(g_fit[1] ** 2 - 4 * (g_fit[0] - y_measured) * g_fit[2])) / 2 / g_fit[2])
    plt.scatter(x_inf, x_pred, label='NO2')

    x_vals_smooth = np.linspace(0, max(x_inf), num=50)
    plt.plot(x_vals_smooth, x_vals_smooth, 'k--', alpha=0.5)  # perfect prediction
    # plt.xlim([0,1.25])
    plt.legend()
    plt.xlabel('Known NO2- or NO3- concentration of standards [mM]')
    plt.ylabel('Predicted NO2- or NO3- concentration [mM]')
    # plt.title('VCl3 standard curve')
    plt.title('Nitrite pure standard prediction with same standards')
    # plt.show()
    # fit = [[no2_blank,no2no3_blank], g_fit, v_fit]
    return


def plot_nitrite_standard_predict_no2no3(meta_fn, no2_fn, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None,
                                         no2no3_540_fn=None, no2no3_900_fn=None):  # KS added 11/4/21
    meta = pd.read_csv(meta_fn, index_col=0).dropna(how='all')
    meta = meta[['NO2', 'NO3']]  # keep only the NO2/NO3 concentration columns
    no2 = read_griess(meta_fn, data_fn=no2_fn, data_540_fn=no2_540_fn, data_900_fn=no2_900_fn)
    no2no3 = read_griess(meta_fn, data_fn=no2no3_fn, data_540_fn=no2no3_540_fn, data_900_fn=no2no3_900_fn)

    # remove nan values
    # old code: nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    # new (1/31/22 Kiseok)
    nan_idx = no2[no2.apply(np.isnan)].index.union(no2no3[no2no3.apply(np.isnan)].index)
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    no2 = no2.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)

    # subtract blank values
    blank_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] == 0)].tolist()
    no2_blank = no2.loc[blank_idx].median()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2 = no2 - no2_blank
    no2no3 = no2no3 - no2no3_blank

    # identify pure no2 and pure no3 samples
    no2_std_idx = meta.index[(meta['NO2'] > 0) & (meta['NO3'] == 0)].tolist()
    no3_std_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] > 0)].tolist()

    # merge those two list
    pure_std_idx = no2_std_idx + no3_std_idx

    # no2 standard curve, griess measurement
    y = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)
    # plt.scatter(x,y,label = 'griess measurement')
    # x_vals = x

    # Fit a quadratic model to the nitrite concentration
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    g_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]

    # no2 in pure conditions prediction
    x_inf = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)
    y_measured = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x_pred = ((-g_fit[1] + np.sqrt(g_fit[1] ** 2 - 4 * (g_fit[0] - y_measured) * g_fit[2])) / 2 / g_fit[2])
    plt.scatter(x_inf, x_pred, label='NO2', alpha=0.9)

    # no2 and no3 standard curves, vcl3 measurement
    y1 = no2no3.loc[no2_std_idx].values
    y2 = no2no3.loc[no3_std_idx].values
    y = np.append(y1, y2).reshape(-1, 1)
    x1 = meta.loc[no2_std_idx].values
    x2 = meta.loc[no3_std_idx].values
    x = np.append(np.sum(x1, axis=1), np.sum(x2, axis=1)).reshape(-1, 1)

    # Fit a quadratic model to the sum of nitrate and nitrite concentrations
    x_vals1 = np.sum(x1, axis=1)
    # print(x_vals1)
    x_vals2 = np.sum(x2, axis=1)
    # print(x_vals2)
    x_vals = x
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables

    # plt.scatter(x_vals,y,label = 'VCl3 measurement')
    # plt.scatter(x_vals1,y1,label = 'VCl3 measurement NO2')
    # plt.scatter(x_vals2,y2,label = 'VCl3 measurement reduced NO3')
    reg = LinearRegression().fit(x, y)
    v_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]

    # plot NO2NO3 prediction
    x_inf_NO2 = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)
    y_measured_NO2NO3 = no2no3.loc[no2_std_idx].values
    # print(x_inf_NO2NO3)
    # print(y_measured_NO2NO3)
    x_pred_NO2NO3 = (
                (-v_fit[1] + np.sqrt(v_fit[1] ** 2 - 4 * (v_fit[0] - y_measured_NO2NO3) * v_fit[2])) / 2 / v_fit[2])
    plt.scatter(x_inf_NO2, x_pred_NO2NO3, label='NO2NO3', alpha=0.9)

    ## plot NO3 prediction
    x_pred_NO3 = ((-v_fit[1] + np.sqrt(v_fit[1] ** 2 - 4 * (v_fit[0] - y_measured_NO2NO3) * v_fit[2])) / 2 / v_fit[
        2]) - x_pred.T
    x_pred_NO3[x_pred_NO3 < 0] = 0.0
    plt.scatter(x_inf_NO2, x_pred_NO3, label='NO3', alpha=0.9)

    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num=50)
    plt.plot(x_vals_smooth, x_vals_smooth, 'k--', alpha=0.5)  # perfect prediction

    plt.legend()
    plt.xlabel('Known NO2- input concentration of standards [mM]')
    plt.ylabel('Predicted NO2-, NO3- concentration [mM]')
    plt.title('Nitrite, nitrate concentration prediction for pure NO2 standards')
    return


def plot_nitrate_standard_predict_no2no3(meta_fn, no2_fn, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None,
                                         no2no3_540_fn=None, no2no3_900_fn=None):  # KS added 11/4/21
    meta = pd.read_csv(meta_fn, index_col=0).dropna(how='all')
    meta = meta[['NO2', 'NO3']]  # keep only the NO2/NO3 concentration columns
    no2 = read_griess(meta_fn, data_fn=no2_fn, data_540_fn=no2_540_fn, data_900_fn=no2_900_fn)
    no2no3 = read_griess(meta_fn, data_fn=no2no3_fn, data_540_fn=no2no3_540_fn, data_900_fn=no2no3_900_fn)

    # remove nan values
    # old code: nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    # new (1/31/22 Kiseok)
    nan_idx = no2[no2.apply(np.isnan)].index.union(no2no3[no2no3.apply(np.isnan)].index)
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    no2 = no2.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)

    # subtract blank values
    blank_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] == 0)].tolist()
    no2_blank = no2.loc[blank_idx].median()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2 = no2 - no2_blank
    no2no3 = no2no3 - no2no3_blank

    # identify pure no2 and pure no3 samples
    no2_std_idx = meta.index[(meta['NO2'] > 0) & (meta['NO3'] == 0)].tolist()
    no3_std_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] > 0)].tolist()

    # merge those two list
    pure_std_idx = no2_std_idx + no3_std_idx

    # no2 standard curve, griess measurement
    y = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)
    # plt.scatter(x,y,label = 'griess measurement')
    # x_vals = x

    # Fit a quadratic model to the nitrite concentration
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    g_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]

    # no2 in pure conditions prediction
    x_inf_NO3 = meta['NO3'].loc[no3_std_idx].values.reshape(-1, 1)
    y_measured = no2.loc[no3_std_idx].values.reshape(-1, 1)
    x_pred = ((-g_fit[1] + np.sqrt(g_fit[1] ** 2 - 4 * (g_fit[0] - y_measured) * g_fit[2])) / 2 / g_fit[2])

    # no2 and no3 standard curves, vcl3 measurement
    y1 = no2no3.loc[no2_std_idx].values
    y2 = no2no3.loc[no3_std_idx].values
    y = np.append(y1, y2).reshape(-1, 1)
    x1 = meta.loc[no2_std_idx].values
    x2 = meta.loc[no3_std_idx].values
    x = np.append(np.sum(x1, axis=1), np.sum(x2, axis=1)).reshape(-1, 1)

    # Fit a quadratic model to the sum of nitrate and nitrite concentrations
    x_vals1 = np.sum(x1, axis=1)
    # print(x_vals1)
    x_vals2 = np.sum(x2, axis=1)
    # print(x_vals2)
    x_vals = x
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables

    # plt.scatter(x_vals,y,label = 'VCl3 measurement')
    # plt.scatter(x_vals1,y1,label = 'VCl3 measurement NO2')
    # plt.scatter(x_vals2,y2,label = 'VCl3 measurement reduced NO3')
    reg = LinearRegression().fit(x, y)
    v_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]

    # plot NO2NO3 prediction
    x_inf_NO3 = meta['NO3'].loc[no3_std_idx].values.reshape(-1, 1)
    y_measured_NO2NO3 = no2no3.loc[no3_std_idx].values
    # print(x_inf_NO2NO3)
    # print(y_measured_NO2NO3)
    x_pred_NO2NO3 = (
                (-v_fit[1] + np.sqrt(v_fit[1] ** 2 - 4 * (v_fit[0] - y_measured_NO2NO3) * v_fit[2])) / 2 / v_fit[2])
    plt.scatter(x_inf_NO3, x_pred_NO2NO3, label='NO2NO3', alpha=0.9)

    ## plot NO3 prediction
    x_pred_NO3 = ((-v_fit[1] + np.sqrt(v_fit[1] ** 2 - 4 * (v_fit[0] - y_measured_NO2NO3) * v_fit[2])) / 2 / v_fit[
        2]) - x_pred.T
    x_pred_NO3[x_pred_NO3 < 0] = 0.0
    plt.scatter(x_inf_NO3, x_pred_NO3, label='NO3', alpha=0.9)

    ## plot NO2 prediction
    plt.scatter(x_inf_NO3, x_pred, label='NO2', alpha=0.9)

    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num=50)
    plt.plot(x_vals_smooth, x_vals_smooth, 'k--', alpha=0.5)  # perfect prediction

    plt.legend()
    plt.xlabel('Known NO3- input concentration of standards [mM]')
    plt.ylabel('Predicted NO2-, NO3- concentration [mM]')
    plt.title('Nitrite, nitrate concentration prediction for pure NO3 standards')
    return


def fit_griess(meta_fn, no2_fn=None, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None, no2no3_540_fn=None,
               no2no3_900_fn=None, id=None, out_dir=None):
    """
    Fits a quadratic model to the nitrite and nitrate concentrations using the provided metadata and data files.

    Parameters:
    meta_fn (str): Path to the metadata file.
    no2_fn (str, optional): Path to the NO2 data file.
    no2_540_fn (str, optional): Path to the 540 nm well scan data file for NO2.
    no2_900_fn (str, optional): Path to the 900 nm well scan data file for NO2.
    no2no3_fn (str, optional): Path to the NO2NO3 data file.
    no2no3_540_fn (str, optional): Path to the 540 nm well scan data file for NO2NO3.
    no2no3_900_fn (str, optional): Path to the 900 nm well scan data file for NO2NO3.

    Returns:
    list: A list containing the fit parameters for NO2, NO2NO3, and NO3.
    """
    
    meta = pd.read_csv(meta_fn, index_col=0).dropna(how='all')
    meta = meta[['NO2', 'NO3']]  # keep only the NO2/NO3 concentration columns
    # no2 = read_griess(meta_fn,no2_fn)
    no2 = read_griess(meta_fn, data_fn=no2_fn, data_540_fn=no2_540_fn, data_900_fn=no2_900_fn)
    no2no3 = read_griess(meta_fn, data_fn=no2no3_fn, data_540_fn=no2no3_540_fn, data_900_fn=no2no3_900_fn)

    # remove nan values
    # old code: nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    # new (1/31/22 Kiseok)
    nan_idx = no2[no2.apply(np.isnan)].index.union(no2no3[no2no3.apply(np.isnan)].index)
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    no2 = no2.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)

    # subtract blank values 
    blank_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] == 0)].tolist()
    no2_blank = no2.loc[blank_idx].median()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2 = no2 - no2_blank
    no2no3 = no2no3 - no2no3_blank

    # identify pure no2 and pure no3 samples
    # no2_std_idx = meta.index[(meta['NO2']>0) & (meta['NO3']==0)].tolist()
    # no3_std_idx = meta.index[(meta['NO2']==0) & (meta['NO3']>0)].tolist()

    # identify samples with no2 or no3
    no2_std_idx = meta.index[(meta['NO2'] >= 0)].tolist()
    no3_std_idx = meta.index[(meta['NO3'] >= 0)].tolist()

    # no2 standard curve, griess measurement
    y = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)

    # Fit a quadratic model to the nitrite concentration
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    g_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]

    if out_dir is not None:
        # Save the plot of y vs x with the quadratic line given by g_fit
        plt.scatter(x[:, 0], y, label='Measured values')
        x_vals_smooth = np.linspace(min(x[:, 0]), max(x[:, 0]), num=50)
        plt.plot(x_vals_smooth, g_fit[0] + g_fit[1] * x_vals_smooth + g_fit[2] * x_vals_smooth**2, label='Quadratic fit', color='red')
        plt.xlabel('[NO2] (mM)')
        plt.ylabel('540 nm Absorbance')
        plt.title('NO2 Standard Curve')
        plt.legend()
        plt.grid()

        plt.savefig(f"{out_dir}_standard_no2.png")
        print(f"NO2 standard curve plot saved to {out_dir}_standard_no2.png")
        plt.close()

    ## Let's get the p-value (Source: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression)
    params = np.append(reg.intercept_, reg.coef_)
    predictions = reg.predict(x)
    newx = pd.DataFrame({"Constant": np.ones(len(x))}).join(pd.DataFrame(x))
    # print(newx)

    MSE = (sum((y - predictions) ** 2)) / (len(newx) - len(newx.columns))
    # print(MSE)
    var_b = MSE * (np.linalg.inv(np.dot(newx.T, newx)).diagonal())
    # print("var_b",var_b)
    sd_b = np.sqrt(var_b)
    # print("sd_b",sd_b)
    ts_b = params / sd_b
    # print("ts_b",ts_b)

    # print("len(newx[0])",len(newx[0]))
    # print("len(newx)-len(newx[0])",len(newx)-len(newx[0]))
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (y.shape[0] - x.shape[1]))) for i in ts_b]
    # print("p-values",p_values)

    sd_b = np.round(sd_b, 4)
    ts_b = np.round(ts_b, 4)
    p_values = np.round(p_values, 4)
    params = np.round(params, 4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["P-Values"] = [params, sd_b, ts_b,
                                                                                             p_values]

    print("\n\n (From sklearn LinearRegression) Manual calculation of p-value. Here we use sklearn LinearRegression package")
    print(myDF3)

    # Getting summary from statsmodel package
    print("\n\n (From statsmodel package) Coefficients are a bit different from the sklearn LinearRegression package")
    est = sm.OLS(y, x)
    est2 = est.fit()
    print(est2.summary())

    ## Let's get the R squre value
    # compute with sklearn linear_model, although could not find any function to compute adjusted-r-square directly from documentation
    print("\n\n (R square and adjusted R square value)")
    print(reg.score(x, y), 1 - (1 - reg.score(x, y)) * (len(y) - 1) / (len(y) - x.shape[1] - 1))
    print("\n")

    # (2) VCl3+Griess calibration curve, fitted from pure NO2 standards only (NO3=0).
    # Using pure NO2 standards ensures v_fit and g_fit agree when NO3=0 (VCl3 has nothing
    # to reduce), so the subtraction NO3 = NO2NO3_mM - NO2_mM correctly returns 0 when
    # the sample contains no nitrate. Including NO3 standards (where VCl3 efficiency < 1)
    # would shift the slope of v_fit relative to g_fit and introduce a systematic positive
    # floor in all NO3 estimates.
    no2_pure_idx = meta.index[(meta['NO2'] > 0) & (meta['NO3'] == 0)].tolist()
    y = no2no3.loc[no2_pure_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_pure_idx].values.reshape(-1, 1)
    x = np.append(x, x ** 2, axis=1)
    reg = LinearRegression().fit(x, y)
    v_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]

    if out_dir is not None:
        # Save the plot of y vs x with the quadratic line given by g_fit
        plt.scatter(x[:, 0], y, label='Measured values')
        x_vals_smooth = np.linspace(min(x[:, 0]), max(x[:, 0]), num=50)
        plt.plot(x_vals_smooth, v_fit[0] + v_fit[1] * x_vals_smooth + v_fit[2] * x_vals_smooth**2, label='Quadratic fit', color='red')
        plt.xlabel('[NO2] + [NO3] (mM)')
        plt.ylabel('540 nm Absorbance')
        plt.title('NO2 + NO3 Standard Curve')
        plt.legend()
        plt.grid()

        plt.savefig(f"{out_dir}_standard_no2no3.png")
        print(f"NO2 + NO3 standard curve plot saved to {out_dir}_standard_no2no3.png")
        plt.close()

    # (3) no3 standard curves (after vcl3 measurement only fitting with nitrate standards)

    # identify pure no2 and pure no3 samples
    # no2_std_idx = meta.index[(meta['NO2']>0) & (meta['NO3']==0)].tolist()
    no3_std_idx = meta.index[(meta['NO2'] == 0) & (meta['NO3'] > 0)].tolist()

    # no2 standard curve, griess measurement
    y = no2no3.loc[no3_std_idx].values.reshape(-1, 1)
    x = meta['NO3'].loc[no3_std_idx].values.reshape(-1, 1)

    # Fit a quadratic model to the nitrite concentration
    x = np.append(x, x ** 2, axis=1)  # make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    no3_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]

    fit = [[no2_blank, no2no3_blank], g_fit, v_fit, no3_fit]

    return fit


def invert_griess(no2, fit, no2no3=None):
    # Returns inferred concentrations
    NO2 = ((-fit[1][1] + np.sqrt(fit[1][1] ** 2 - 4 * (fit[1][0] - no2) * fit[1][2])) / 2 / fit[1][2]).rename(
        "NO2");  # solve quadratic formula
    NO2[NO2 < 0] = 0.0
    if isinstance(no2no3, pd.Series):
        NO3 = (((-fit[2][1] + np.sqrt(fit[2][1] ** 2 - 4 * (fit[2][0] - no2no3) * fit[2][2])) / 2 / fit[2][
            2]) - NO2).rename("NO3");  # solve quadratic formula
        NO3[NO3 < 0] = 0.0
    else:
        NO3 = NO2.copy().rename("NO3")
        NO3[NO3 != 0] = 0.0

    #data_out = pd.DataFrame()
    #data_out = data_out.append([NO2, NO3]).transpose()
    data_out = pd.concat([NO2, NO3], axis=1) # Ik added 02/17/2025
    return data_out

# Use g_fit (NO2 standard equation) for calculating NO2 concentration and v_fit (NO2NO3 standard equation) for calculating NO2NO3 concentration
def get_concentration_xlsx(excel_output, no2_blank, no2no3_blank, g_fit, v_fit, meta_fn, no2_fn=None, no2_540_fn=None,
                           no2_900_fn=None, no2no3_fn=None, no2no3_540_fn=None, no2no3_900_fn=None):
    # Check for wellscan outlier removal
    print("Check wellscan for outlier removal in NO2")
    df_check_no2 = bd.read_abs_wellscan(no2_900_fn)
    bd.plot_heatmap_wellscan(df_check_no2, title="NO2")
    df_check_no2_540 = bd.read_abs_wellscan(no2_540_fn)  # For 540nm
    bd.check_540_heatmap_wellscan(df_check_no2_540, title="NO2")  # For 540nm

    print("-----------------------")
    print("Check wellscan for outlier removal in NO2NO3")
    df_check_no2no3 = bd.read_abs_wellscan(no2no3_900_fn)
    bd.plot_heatmap_wellscan(df_check_no2no3, title="NO2NO3")
    df_check_no2no3_540 = bd.read_abs_wellscan(no2no3_540_fn)  # For 540nm
    bd.check_540_heatmap_wellscan(df_check_no2no3_540, title="NO2NO3")  # For 540nm

    # read NO2
    print(meta_fn)
    print("Before subtracting blank value")
    NO2_OD540_plus_blank = read_griess(meta_fn, data_fn=None, data_540_fn=no2_540_fn, data_900_fn=no2_900_fn).rename(
        "NO2_OD540")
    print(NO2_OD540_plus_blank)

    # read NO2NO3
    NO2NO3_OD540_plus_blank = read_griess(meta_fn, data_fn=None, data_540_fn=no2no3_540_fn,
                                          data_900_fn=no2no3_900_fn).rename("NO2NO3_OD540")
    print(NO2NO3_OD540_plus_blank)

    # Get NO2
    # subtract blanks
    print("Subtracted blank value")
    NO2_OD540 = NO2_OD540_plus_blank - no2_blank
    # print(NO2_OD540)
    NO2NO3_OD540 = NO2NO3_OD540_plus_blank - no2no3_blank
    # print(NO2NO3_OD540)

    NO2_OD540[NO2_OD540 < 0] = 0.0
    NO2NO3_OD540[NO2NO3_OD540 < 0] = 0.0
    # print(NO2_OD540)
    # print(NO2NO3_OD540)

    # Returns inferred concentrations
    print("Calculated NO2, NO3 concentrations")
    NO2_mM = ((-g_fit[1] + np.sqrt(g_fit[1] ** 2 - 4 * (g_fit[0] - NO2_OD540) * g_fit[2])) / 2 / g_fit[2]).rename(
        "NO2_mM")  ## solve quadratic formula
    NO2_mM[NO2_mM < 0] = 0.0  # make it zero if it is less than zero
    NO2NO3_mM = ((-v_fit[1] + np.sqrt(v_fit[1] ** 2 - 4 * (v_fit[0] - NO2NO3_OD540) * v_fit[2])) / 2 / v_fit[2]).rename(
        "NO2NO3_mM")  ## solve quadratic formula
    NO2NO3_mM[NO2NO3_mM < 0] = 0.0  # make it zero if it is less than zero
    NO3_mM = (NO2NO3_mM - NO2_mM).rename("NO3_mM")  ## solve quadratic formula
    NO3_mM[NO3_mM < 0] = 0.0  # make it zero if it is less than zero

    print(NO2_mM)
    print(NO2NO3_mM)
    print(NO3_mM)

    # combine it into a dataframe
    df_read = pd.DataFrame()
    df_read = pd.concat([df_read, pd.DataFrame([NO2_OD540, NO2NO3_OD540, NO2_mM, NO2NO3_mM, NO3_mM]).T])

    df_meta = pd.read_csv(meta_fn, index_col=0).dropna()
    df_out = pd.concat([df_meta, df_read], axis=1)
    df_out.to_excel(excel_output)
    return


def get_concentration(no2_blank, no2no3_blank, g_fit, v_fit, meta_fn, no2_fn=None, no2_540_fn=None,
                           no2_900_fn=None, no2no3_fn=None, no2no3_540_fn=None, no2no3_900_fn=None, extract_factor = 2.5): # Edited by Ik
    # read NO2
    # print(meta_fn)
    # print("Before subtracting blank value")
    NO2_OD540_plus_blank = read_griess(meta_fn, data_fn=None, data_540_fn=no2_540_fn, data_900_fn=no2_900_fn).rename(
        "NO2_OD540")
    # print(NO2_OD540_plus_blank)

    # read NO2NO3
    NO2NO3_OD540_plus_blank = read_griess(meta_fn, data_fn=None, data_540_fn=no2no3_540_fn,
                                          data_900_fn=no2no3_900_fn).rename("NO2NO3_OD540")
    # print(NO2NO3_OD540_plus_blank)

    # Get NO2
    # subtract blanks
    # print("Subtracted blank value")
    NO2_OD540 = NO2_OD540_plus_blank - no2_blank
    # print(NO2_OD540)
    NO2NO3_OD540 = NO2NO3_OD540_plus_blank - no2no3_blank
    # print(NO2NO3_OD540)

    NO2_OD540[NO2_OD540 < 0] = 0.0
    NO2NO3_OD540[NO2NO3_OD540 < 0] = 0.0
    # print(NO2_OD540)
    # print(NO2NO3_OD540)

    # Returns inferred concentrations
    # print("Calculated NO2, NO3 concentrations")
    NO2_mM = ((-g_fit[1] + np.sqrt(g_fit[1] ** 2 - 4 * (g_fit[0] - NO2_OD540) * g_fit[2])) / 2 / g_fit[2]).rename(
        "NO2_mM")  ## solve quadratic formula
    NO2_mM[NO2_mM < 0] = 0.0  # make it zero if it is less than zero
    NO2NO3_mM = ((-v_fit[1] + np.sqrt(v_fit[1] ** 2 - 4 * (v_fit[0] - NO2NO3_OD540) * v_fit[2])) / 2 / v_fit[2]).rename(
        "NO2NO3_mM")  ## solve quadratic formula
    NO2NO3_mM[NO2NO3_mM < 0] = 0.0  # make it zero if it is less than zero
    NO3_mM = (NO2NO3_mM - NO2_mM).rename("NO3_mM")  ## solve quadratic formula
    NO3_mM[NO3_mM < 0] = 0.0  # make it zero if it is less than zero

    NO2_mM = NO2_mM * extract_factor
    NO2NO3_mM = NO2NO3_mM * extract_factor
    NO3_mM = NO3_mM * extract_factor

    # print(NO2_mM)
    # print(NO2NO3_mM)
    # print(NO3_mM)

    # combine it into a dataframe
    df_read = pd.DataFrame()
    df_read = pd.concat([df_read, pd.DataFrame([NO2_OD540, NO2NO3_OD540, NO2_mM, NO2NO3_mM, NO3_mM]).T])
    df_meta = pd.read_csv(meta_fn, index_col=0).dropna(how='all')
    df_out = pd.concat([df_meta, df_read], axis=1)
    #df_out.to_csv(output_fn)
    return df_out


def load_plate_timeseries(meta_fn, od_fn, no2_fns, no2no3_540_fns, no2no3_900_fns, fit,
                          pidx):  # written by KG, added by KC 08/16/2021
    plate_meta = pd.read_csv(meta_fn, index_col=0).dropna()  # import metadata and drop rows with any empty elements
    plate_od600 = bd.read_abs_endpoint(od_fn)  # read in endpoint OD measurements
    plate_od600 = plate_od600['600']  # use only 600 nm

    # reads in files and infers concentrations from absorbances using standard curve parameters ("fit")
    # assumes filenames are in the order t1, t2, t3, ...
    plate_no2 = pd.DataFrame()
    plate_no3 = pd.DataFrame()
    for i in range(0, len(no2_fns)):
        # print(i)
        no2 = read_griess(meta_fn, data_fn=no2_fns[i])
        if no2no3_540_fns == None:
            no2no3 = None
        else:
            no2no3 = read_griess(meta_fn, data_540_fn=no2no3_540_fns[i], data_900_fn=no2no3_900_fns[i])
        data = invert_griess(no2, fit, no2no3)
        plate_no2 = plate_no2.append(data["NO2"].rename("t" + str(i + 1)))
        plate_no3 = plate_no3.append(data["NO3"].rename("t" + str(i + 1)))

    # adds a prefix to the column names to indicate what plate the sample belongs to
    plate_meta = ((plate_meta.transpose()).add_prefix("p" + str(pidx) + "_")).transpose()
    plate_od600 = ((plate_od600.transpose()).add_prefix("p" + str(pidx) + "_")).transpose()
    plate_no2 = plate_no2.add_prefix("p" + str(pidx) + "_")
    plate_no3 = plate_no3.add_prefix("p" + str(pidx) + "_")

    # transposes data frame so columns are time points and rows are wells
    plate_no2 = plate_no2.transpose()
    plate_no3 = plate_no3.transpose()

    return [plate_meta, plate_od600, plate_no2, plate_no3]
