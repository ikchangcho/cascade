#Karna Gowda
#0.0.1

import bmgdata as bd
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

def find_outliers(data):
    #returns true for elements more than 1.5 interquartile ranges above the upper quartile.
    
    q75 = np.quantile(data,0.75)
    q25 = np.quantile(data,0.25)
    is_outlier = data > q75 + (q75-q25)*1.5
    return is_outlier

def remove_bubbles(data_in,data_540,data_900):
    #find wells with bubbles in the no2no3 measurements
    
    bub_in  = find_outliers(data_in['900'])
    bub_900 = find_outliers(data_900)

    #replace values at 540 nm
    data_out = data_in.copy()
    for idx in bub_in.index[bub_in == True].tolist():
        replacement = data_540.loc[idx]
        data_out['540'].loc[idx] = np.median(replacement[~bub_900.loc[idx]]) #replace with median of values that don't have bubbles
    return data_out

def read_griess(meta_fn,data_fn=None,data_540_fn=None,data_900_fn=None):
    #returns absorbance data at 540 nm
    #corrects for bubbles if well scan measurements are provided
    #there are three valid cases that this function works for:
    #1) A file name is provided for data_fn. Usually this is for importing no2 data.
    #2) File names are provided for data_fn and associated well scan files data_540_fn and data_900_fn. Usually this is for importing old no2no3 measurements using both endpoint and well scan data.
    #3) File names are provided for only well scan files data_540_fn and data_900_fn. This is the prefered measurement data for future experiments.
    
    meta       = pd.read_csv(meta_fn,index_col=0).dropna(how='all')  #import metadata
    
    if (data_540_fn != None) & (data_900_fn != None): 
            data_540 = bd.read_abs_wellscan(data_540_fn) #import well scan data (540 nm)
            data_540 = data_540[data_540.index.isin(meta.index)] #remove indices with no values in the metadata
            data_900 = bd.read_abs_wellscan(data_900_fn) #import well scan data (900 nm)
            data_900 = data_900[data_900.index.isin(meta.index)] #remove indices with no values in the metadata
            
    if data_fn != None: #case 1 or 2
        data_out    = bd.read_abs_endpoint(data_fn) #import data
        data_out    = data_out[data_out.index.isin(meta.index)] #remove indices with no values in the metadata
        #correct for bubbles
        if data_540_fn != None: #case 2
            data_out = remove_bubbles(data_out,data_540,data_900) #correct for bubbles in measurement.
        data_out = data_out['540'] #pick out 540 nm measurement
    elif (data_540_fn != None) & (data_900_fn != None): #case 3
        bub_900 = find_outliers(data_900)
        for row in data_540.index:
            if sum(np.isnan(bub_900.loc[row])) < 4:
                data_540.loc[row][bub_900.loc[row]] = np.NaN #Set bubbles to NaN
            else:
                data_540.loc[row] = data_540.loc[row] - data_900.loc[row]
        data_out = data_540.median(axis=1)
        data_out = data_out.rename("540")
        
    return data_out

def plot_griess_fit(meta_fn, no2_fn, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None,no2no3_540_fn=None,no2no3_900_fn=None):  #KC added 08/06/2021
    meta = pd.read_csv(meta_fn,index_col=0).dropna(how='all')
    meta = meta[['NO2','NO3']] #keep only the NO2/NO3 concentration columns
    # no2 = read_griess(meta_fn,no2_fn)
    no2 = read_griess(meta_fn, data_fn=no2_fn, data_540_fn= no2_540_fn, data_900_fn= no2_900_fn)
    no2no3 = read_griess(meta_fn,data_fn = no2no3_fn,data_540_fn=no2no3_540_fn,data_900_fn=no2no3_900_fn)
    
    #remove nan values
    nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    meta = meta.drop(nan_idx)
    no2 = no2.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)
    
    #subtract blank values
    blank_idx = meta.index[(meta['NO2']==0) & (meta['NO3']==0)].tolist()
    #print(blank_idx)
    no2_blank = no2.loc[blank_idx].median()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2 = no2 - no2_blank
    no2no3 = no2no3 - no2no3_blank

    #identify pure no2 and pure no3 samples
    no2_std_idx = meta.index[meta['NO2']>0].tolist()
    no3_std_idx = meta.index[meta['NO3']>0].tolist()

    #no2 standard curve, griess measurement
    y = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)
    plt.scatter(x,y,label = 'Measurement of NO2 standards')
    x_vals = x
   
    #Fit a quadratic model to the nitrite concentration
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    g_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    #print(x_vals)
    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num = 50)
    plt.plot(x_vals_smooth, reg.intercept_[0] + x_vals_smooth*reg.coef_[0][0] + x_vals_smooth*x_vals_smooth*reg.coef_[0][1], label =  'quadratic model fit')
    plt.xlabel('NO2 [mM]')
    plt.ylabel('OD measurement (540nm)')
    plt.title('Griess assay standard curve (before VCl treatment: Nitrite), y='+str(round(reg.intercept_[0],2))+ '+'+str(round(reg.coef_[0][0],2))+'x+'+str(round(reg.coef_[0][1],2))+'x^2'+', Adjusted R^2 =' +str(np.round(1 - (1-reg.score(x, y))*(len(y)-1)/(len(y)-x.shape[1]-1),3)))
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

def plot_vcl_fit(meta_fn, no2_fn, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None, no2no3_540_fn=None, no2no3_900_fn=None):  #KC added 08/06/2021
    meta = pd.read_csv(meta_fn,index_col=0).dropna(how='all')
    meta = meta[['NO2','NO3']] #keep only the NO2/NO3 concentration columns
    # no2 = read_griess(meta_fn,no2_fn) # this does not correct for bouble in no2 data
    no2no3 = read_griess(meta_fn, data_540_fn=no2no3_540_fn, data_900_fn=no2no3_900_fn)

    #remove nan values
    nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    meta = meta.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)
    
    #subtract blank values
    blank_idx = meta.index[(meta['NO2']==0) & (meta['NO3']==0)].tolist()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2no3 = no2no3 - no2no3_blank

    # identify pure no2 and pure no3 samples
    #no2_std_idx = meta.index[(meta['NO2']>0) & (meta['NO3']==0)].tolist()
    #no3_std_idx = meta.index[(meta['NO2']==0) & (meta['NO3']>0)].tolist()
    
    # identify samples with no2 or no3
    no2_std_idx = meta.index[(meta['NO2']>0)].tolist()
    no3_std_idx = meta.index[(meta['NO3']>0)].tolist()

    #no2 and no3 standard curves, vcl3 measurement
    y1 = no2no3.loc[no2_std_idx].values
    y2 = no2no3.loc[no3_std_idx].values
    x1 = meta.loc[no2_std_idx].values
    x2 = meta.loc[no3_std_idx].values

    #Fit a quadratic model to the sum of nitrate and nitrite concentrations
    x = np.append(np.sum(x1,axis=1),np.sum(x2,axis=1)).reshape(-1,1)
    #print(x1)
    #print(x2)
    #error
    x_vals1 = np.sum(x1,axis=1)
    #print(x_vals1)
    x_vals2 = np.sum(x2,axis=1)
    #print(x_vals2)
    x_vals = x
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    y = np.append(y1,y2).reshape(-1,1)
    #plt.scatter(x_vals,y,label = 'VCl3 measurement')
    plt.scatter(x_vals1,y1,label = 'Measurement NO2 standards')
    plt.scatter(x_vals2,y2,label = 'Measurement of reduced NO3 standards')
    reg = LinearRegression().fit(x, y)
    v_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num = 50)
    plt.plot(x_vals_smooth, (reg.intercept_[0] + x_vals_smooth*reg.coef_[0][0] + x_vals_smooth*x_vals_smooth*reg.coef_[0][1]), label = 'quadratic model fit')
    
    plt.legend()
    plt.xlabel('NO2 + NO3 [mM]')
    plt.ylabel('OD measurement (540nm)')
    #plt.title('VCl3 standard curve')
    plt.title('Griess assay standard curve (after VCl treatment: Nitrate reduction), y='+str(round(reg.intercept_[0],2))+ '+'+str(round(reg.coef_[0][0],2))+'x+'+str(round(reg.coef_[0][1],2))+'x^2'+', Adjusted R^2 =' +str(np.round(1 - (1-reg.score(x, y))*(len(y)-1)/(len(y)-x.shape[1]-1),3)))
    
    ## Let's get the p-value (Source: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression)
    params = np.append(reg.intercept_,reg.coef_)
    predictions = reg.predict(x)
    newx = pd.DataFrame({"Constant":np.ones(len(x))}).join(pd.DataFrame(x))
    #print(newx)
                                                        
    MSE = (sum((y-predictions)**2))/(len(newx)-len(newx.columns))
    #print(MSE)
    var_b = MSE*(np.linalg.inv(np.dot(newx.T,newx)).diagonal())
    #print("var_b",var_b)
    sd_b = np.sqrt(var_b)
    #print("sd_b",sd_b)
    ts_b = params/ sd_b
    #print("ts_b",ts_b)
    
    #print("len(newx[0])",len(newx[0]))
    #print("len(newx)-len(newx[0])",len(newx)-len(newx[0]))
    p_values =[2*(1-stats.t.cdf(np.abs(i), (y.shape[0] - x.shape[1]))) for i in ts_b]   
    #print("p-values",p_values)

    sd_b = np.round(sd_b,4)
    ts_b = np.round(ts_b,4)
    p_values = np.round(p_values,4)
    params = np.round(params,4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["P-Values"] = [params,sd_b,ts_b,p_values]
    
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
    print(reg.score(x, y), 1 - (1-reg.score(x, y))*(len(y)-1)/(len(y)-x.shape[1]-1))
    print("\n")
    
    return

def plot_mixed_standard_predict(meta_fn, no2_fn, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None, no2no3_540_fn=None, no2no3_900_fn=None):  #KC added 08/06/2021
    meta = pd.read_csv(meta_fn,index_col=0).dropna(how='all')
    meta = meta[['NO2','NO3']] #keep only the NO2/NO3 concentration columns
    no2 = read_griess(meta_fn, data_fn=no2_fn, data_540_fn= no2_540_fn, data_900_fn= no2_900_fn)
    no2no3 = read_griess(meta_fn, data_fn = no2no3_fn, data_540_fn=no2no3_540_fn, data_900_fn=no2no3_900_fn)
    
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
    no2_std_idx = meta.index[(meta['NO2']>0) & (meta['NO3']==0)].tolist()
    no3_std_idx = meta.index[(meta['NO2']==0) & (meta['NO3']>0)].tolist()
    
    #identify mixed no2 and no3 samples
    no2no3_std_idx = meta.index[(meta['NO2']>0) & (meta['NO3']>0)].tolist()
    
    #no2 standard curve, griess measurement
    y = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)
    #plt.scatter(x,y,label = 'griess measurement')
    #x_vals = x
   
    #Fit a quadratic model to the nitrite concentration
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    g_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    #print(x_vals)
    #plt.plot(x_vals, reg.intercept_[0] + x_vals*reg.coef_[0][0] + x_vals*x_vals*reg.coef_[0][1], label =  'griess fit')
    
    #no2 in mixed conditions prediction
    x_inf = meta['NO2'].loc[no2no3_std_idx].values.reshape(-1, 1)
    y_measured = no2.loc[no2no3_std_idx].values.reshape(-1, 1)
    x_pred = ((-g_fit[1] + np.sqrt(g_fit[1]**2 - 4*(g_fit[0]-y_measured)*g_fit[2]))/2/g_fit[2])
    plt.scatter(x_inf, x_pred, label = 'NO2')
    
    #no2 and no3 standard curves, vcl3 measurement
    y1 = no2no3.loc[no2_std_idx].values
    y2 = no2no3.loc[no3_std_idx].values
    x1 = meta.loc[no2_std_idx].values
    x2 = meta.loc[no3_std_idx].values

    x_inf_NO3 = meta['NO3'].loc[no2no3_std_idx].values.reshape(-1, 1)
    y_measured_NO2NO3 = no2no3.loc[no2no3_std_idx].values
    #print(x_inf_NO2NO3)
    #print(y_measured_NO2NO3)
    
    #Fit a quadratic model to the sum of nitrate and nitrite concentrations
    x = np.append(np.sum(x1,axis=1),np.sum(x2,axis=1)).reshape(-1,1)
    #print(x1)
    #print(x2)
    #error
    x_vals1 = np.sum(x1,axis=1)
    #print(x_vals1)
    x_vals2 = np.sum(x2,axis=1)
    #print(x_vals2)
    x_vals = x
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    y = np.append(y1,y2).reshape(-1,1)
    #plt.scatter(x_vals,y,label = 'VCl3 measurement')
    #plt.scatter(x_vals1,y1,label = 'VCl3 measurement NO2')
    #plt.scatter(x_vals2,y2,label = 'VCl3 measurement reduced NO3')
    reg = LinearRegression().fit(x, y)
    v_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    x_pred_NO3 = ((-v_fit[1] + np.sqrt(v_fit[1]**2 - 4*(v_fit[0]-y_measured_NO2NO3)*v_fit[2]))/2/v_fit[2]) - x_pred.T
    
    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num = 50)
    #plt.plot(x_vals_smooth, (reg.intercept_[0] + x_vals_smooth*reg.coef_[0][0] + x_vals_smooth*x_vals_smooth*reg.coef_[0][1]), label = 'VCl3 fit')
    plt.plot(x_vals_smooth, x_vals_smooth, 'k--', alpha = 0.5) #perfect prediction
    #print(y_measured_NO2NO3)
    #print(x_pred)
    #print(x_inf_NO3)
    #print(x_pred_NO3)
    plt.scatter(x_inf_NO3, x_pred_NO3,label = 'NO3', alpha = 0.3) 
    plt.xlim([0,1.25])
    plt.legend()
    plt.xlabel('Known NO2- or NO3- concentration of standards [mM]')
    plt.ylabel('Predicted NO2- or NO3- concentration [mM]')
    #plt.title('VCl3 standard curve')
    plt.title('Nitrite, nitrate mixed standard prediction')
    #plt.show()
    #fit = [[no2_blank,no2no3_blank], g_fit, v_fit]
    return

def fit_griess(meta_fn,no2_fn, no2_540_fn=None, no2_900_fn=None, no2no3_fn=None,no2no3_540_fn=None,no2no3_900_fn=None): 
    meta = pd.read_csv(meta_fn,index_col=0).dropna(how='all')
    meta = meta[['NO2','NO3']] #keep only the NO2/NO3 concentration columns
    #no2 = read_griess(meta_fn,no2_fn)
    no2 = read_griess(meta_fn, data_fn=no2_fn, data_540_fn= no2_540_fn, data_900_fn= no2_900_fn)
    no2no3 = read_griess(meta_fn, data_fn = no2no3_fn, data_540_fn=no2no3_540_fn, data_900_fn=no2no3_900_fn)
    
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
    #no2_std_idx = meta.index[(meta['NO2']>0) & (meta['NO3']==0)].tolist()
    #no3_std_idx = meta.index[(meta['NO2']==0) & (meta['NO3']>0)].tolist()

    # identify samples with no2 or no3
    no2_std_idx = meta.index[(meta['NO2']>0)].tolist()
    no3_std_idx = meta.index[(meta['NO3']>0)].tolist()
    
    #no2 standard curve, griess measurement
    y = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)

    #Fit a quadratic model to the nitrite concentration
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    g_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    
    ## Let's get the p-value (Source: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression)
    params = np.append(reg.intercept_,reg.coef_)
    predictions = reg.predict(x)
    newx = pd.DataFrame({"Constant":np.ones(len(x))}).join(pd.DataFrame(x))
    #print(newx)
                                                        
    MSE = (sum((y-predictions)**2))/(len(newx)-len(newx.columns))
    #print(MSE)
    var_b = MSE*(np.linalg.inv(np.dot(newx.T,newx)).diagonal())
    #print("var_b",var_b)
    sd_b = np.sqrt(var_b)
    #print("sd_b",sd_b)
    ts_b = params/ sd_b
    #print("ts_b",ts_b)
    
    #print("len(newx[0])",len(newx[0]))
    #print("len(newx)-len(newx[0])",len(newx)-len(newx[0]))
    p_values =[2*(1-stats.t.cdf(np.abs(i), (y.shape[0] - x.shape[1]))) for i in ts_b]   
    #print("p-values",p_values)

    sd_b = np.round(sd_b,4)
    ts_b = np.round(ts_b,4)
    p_values = np.round(p_values,4)
    params = np.round(params,4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["P-Values"] = [params,sd_b,ts_b,p_values]
    
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
    print(reg.score(x, y), 1 - (1-reg.score(x, y))*(len(y)-1)/(len(y)-x.shape[1]-1))
    print("\n")
    
    #no2 and no3 standard curves, vcl3 measurement
    y1 = no2no3.loc[no2_std_idx].values
    y2 = no2no3.loc[no3_std_idx].values
    x1 = meta.loc[no2_std_idx].values
    x2 = meta.loc[no3_std_idx].values

    #Fit a quadratic model to the sum of nitrate and nitrite concentrations
    x = np.append(np.sum(x1,axis=1),np.sum(x2,axis=1)).reshape(-1,1)
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    y = np.append(y1,y2).reshape(-1,1)
    reg = LinearRegression().fit(x, y)
    v_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]

    fit = [[no2_blank,no2no3_blank], g_fit, v_fit]
    
    return fit

def invert_griess(no2,fit,no2no3=None):
    #Returns inferred concentrations
    NO2 = ((-fit[1][1] + np.sqrt(fit[1][1]**2 - 4*(fit[1][0]-no2)*fit[1][2]))/2/fit[1][2]).rename("NO2"); #solve quadratic formula
    NO2[NO2<0] = 0.0
    if isinstance(no2no3,pd.Series):
        NO3 = (((-fit[2][1] + np.sqrt(fit[2][1]**2 - 4*(fit[2][0]-no2no3)*fit[2][2]))/2/fit[2][2]) - NO2).rename("NO3"); #solve quadratic formula
        NO3[NO3<0] = 0.0
    else:
        NO3 = NO2.copy().rename("NO3")
        NO3[NO3!=0] = 0.0

    data_out = pd.DataFrame()
    data_out = data_out.append([NO2,NO3]).transpose()
    return data_out

def load_plate_timeseries(meta_fn,od_fn,no2_fns,no2no3_540_fns,no2no3_900_fns,fit,pidx): #written by KG, added by KC 08/16/2021
    plate_meta = pd.read_csv(meta_fn,index_col=0).dropna()  #import metadata and drop rows with any empty elements
    plate_od600 = bd.read_abs_endpoint(od_fn) #read in endpoint OD measurements
    plate_od600 = plate_od600['600'] #use only 600 nm

    #reads in files and infers concentrations from absorbances using standard curve parameters ("fit")
    #assumes filenames are in the order t1, t2, t3, ...
    plate_no2 = pd.DataFrame()
    plate_no3 = pd.DataFrame()
    for i in range(0,len(no2_fns)):
        #print(i)
        no2 = read_griess(meta_fn,data_fn=no2_fns[i])
        if no2no3_540_fns == None:
            no2no3 = None
        else:
            no2no3 = read_griess(meta_fn,data_540_fn=no2no3_540_fns[i],data_900_fn=no2no3_900_fns[i])
        data = invert_griess(no2,fit,no2no3)
        plate_no2 = plate_no2.append(data["NO2"].rename("t"+str(i+1)))
        plate_no3 = plate_no3.append(data["NO3"].rename("t"+str(i+1)))
    
    #adds a prefix to the column names to indicate what plate the sample belongs to
    plate_meta  = ((plate_meta.transpose()).add_prefix("p" + str(pidx) + "_")).transpose()
    plate_od600 = ((plate_od600.transpose()).add_prefix("p" + str(pidx) + "_")).transpose()
    plate_no2 = plate_no2.add_prefix("p" + str(pidx) + "_")
    plate_no3 = plate_no3.add_prefix("p" + str(pidx) + "_")
    
    #transposes data frame so columns are time points and rows are wells
    plate_no2 = plate_no2.transpose()
    plate_no3 = plate_no3.transpose()
    
    return [plate_meta,plate_od600,plate_no2,plate_no3]
