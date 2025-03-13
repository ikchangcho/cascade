#Adapted from Karna Gowda by Kiseok Lee
#0.0.1
# Changes
# (1) Name: griess.py -> ammonia.py
# (2) 540nm -> 650nm

import bmgdata as bd
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm



#def find_outliers(data):
    #returns true for elements more than 1.5 interquartile ranges above the upper quartile.
    
#    q75 = np.quantile(data,0.75)
#    q25 = np.quantile(data,0.25)
#    is_outlier = data > q75 + (q75-q25)*1.5
#    return is_outlier

def find_outliers(data): # Modified by Kiseok Lee 230622
    val_above = 1 # it was originally 1.5 (I alleviated the threshold because there were too many outliers)
    print("returns true for elements more than "+str(val_above)+" interquartile ranges above the upper quartile")
    q75 = np.quantile(data,0.75)
    q25 = np.quantile(data,0.25)
    is_outlier = data > q75 + (q75-q25)*val_above
    df_outliers = is_outlier # dataframe
    # Need to make sure that it has less than 3 Trues for each row
    # Find the rows that have more than 3 True values.
    # Then, change it to have True in positions of the 2 of the lowest values 
    df_outliers[df_outliers.sum(axis=1) >= 3] = data[df_outliers.sum(axis=1) >= 3].rank(axis=1) > 2
    return df_outliers


def remove_bubbles(data_in, wavelength, data_absorb,data_900): # input absorbance wavelength value (in quotes)
    #find wells with bubbles in the ammonia measurements
    bub_in  = find_outliers(data_in['900'])
    bub_900 = find_outliers(data_900)

    #replace values at 650 nm (or other absorbance value)
    data_out = data_in.copy()
    for idx in bub_in.index[bub_in == True].tolist():
        replacement = data_absorb.loc[idx]
        data_out[wavelength].loc[idx] = np.median(replacement[~bub_900.loc[idx]]) #replace with median of values that don't have bubbles
    return data_out

def read_ammonia(meta_fn, wavelength, data_fn=None, data_absorb_fn=None, data_900_fn=None):
    #returns absorbance data at 650 nm (Walkley-Black assay)
    #corrects for bubbles if well scan measurements are provided
    #there are three valid cases that this function works for:
    #1) A file name is provided for data_fn (220nm - 1000nm). Usually this is for importing ammonia data.
    #2) File names are provided for data_fn (220nm - 1000nm) and associated well scan files data_absorb_fn and data_900_fn.
    #3) File names are provided for only well scan files data_absorb_fn and data_900_fn. This is the prefered measurement data for future experiments.
    
    meta = pd.read_csv(meta_fn,index_col=0).dropna(how='all')  #import metadata
    
    if (data_absorb_fn != None) & (data_900_fn != None): 
            data_absorb = bd.read_abs_wellscan(data_absorb_fn) #import well scan data (650 nm)
            data_absorb = data_absorb[data_absorb.index.isin(meta.index)] #remove indices with no values in the metadata
            data_900 = bd.read_abs_wellscan(data_900_fn) #import well scan data (900 nm)
            data_900 = data_900[data_900.index.isin(meta.index)] #remove indices with no values in the metadata
            
    if data_fn != None: #case 1 or 2
        data_out = bd.read_abs_endpoint(data_fn) #import data
        data_out = data_out[data_out.index.isin(meta.index)] #remove indices with no values in the metadata
        #correct for bubbles
        if data_absorb_fn != None: #case 2
            data_out = remove_bubbles(data_out, wavelength, data_absorb, data_900) #correct for bubbles in measurement.
        data_out = data_out[wavelength] #pick out 650 nm measurement
    elif (data_absorb_fn != None) & (data_900_fn != None): #case 3
        bub_900 = find_outliers(data_900)
        for row in data_absorb.index:
            if sum(np.isnan(bub_900.loc[row])) < 4:
                data_absorb.loc[row][bub_900.loc[row]] = np.NaN #Set bubbles to NaN
            else:
                data_absorb.loc[row] = data_absorb.loc[row] - data_900.loc[row]
        data_out = data_absorb.median(axis=1)
        data_out = data_out.rename(wavelength)
        
    return data_out

def plot_ammonia_fit(meta_fn, wavelength, data_fn): 
    meta = pd.read_csv(meta_fn,index_col=0).dropna(how='all')
    ammonia = read_ammonia(meta_fn, wavelength, data_fn)
    
    #remove nan values
    # old code: nan_idx = meta['Ammonia'].index[meta['Ammonia'].apply(np.isnan)]
    # new (1/31/22 Kiseok)
    nan_idx = ammonia[ammonia.apply(np.isnan)].index
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    ammonia = ammonia.drop(nan_idx)

    #subtract blank values
    blank_idx = meta.index[(meta['Ammonia']==0)].tolist()
    #print(blank_idx)
    ammonia_blank = ammonia.loc[blank_idx].median()
    ammonia = ammonia - ammonia_blank
   
    #ammonia standard curve, ammonia measurement
    y = ammonia.values.reshape(-1, 1)
    x = meta['Ammonia'].values.reshape(-1, 1)
    plt.scatter(x,y,label = 'Measurement')
    x_vals = x
   
    #Fit a quadratic model to the ammonia concentration
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    #print(x_vals)
    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num = 50)
    plt.plot(x_vals_smooth, reg.intercept_[0] + x_vals_smooth*reg.coef_[0][0] + x_vals_smooth*x_vals_smooth*reg.coef_[0][1], label =  'ammonia fit')
    plt.xlabel('Ammonia concentration [N mg/L, mM]')
    plt.ylabel('OD measurement (650nm)')
    plt.title('Ammonia (Salicylate-Hypochlorite assay) standard curve without bubble removal, y='+str(round(reg.intercept_[0],2))+ '+'+str(round(reg.coef_[0][0],2))+'x+'+str(round(reg.coef_[0][1],2))+'x^2'+', Adjusted R^2 =' +str(np.round(1 - (1-reg.score(x, y))*(len(y)-1)/(len(y)-x.shape[1]-1),3)))
    plt.legend()
    #plt.show()
    return

def plot_ammonia_fit_rm_bubble(meta_fn, wavelength, data_fn=None, data_absorb_fn=None, data_900_fn=None):  
    meta = pd.read_csv(meta_fn,index_col=0).dropna(how='all')
    ammonia = read_ammonia(meta_fn, wavelength, data_fn, data_absorb_fn, data_900_fn)
    
    #remove nan values
    # old code: nan_idx = meta['Ammonia'].index[meta['Ammonia'].apply(np.isnan)]
    # new (1/31/22 Kiseok)
    nan_idx = ammonia[ammonia.apply(np.isnan)].index
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    ammonia = ammonia.drop(nan_idx)
    print(ammonia)
    
    #subtract blank values
    blank_idx = meta.index[(meta['Ammonia']==0)].tolist()
    #print(blank_idx)
    ammonia_blank = ammonia.loc[blank_idx].median()
    ammonia = ammonia - ammonia_blank
   
    #ammonia standard curve, ammonia measurement
    y = ammonia.values.reshape(-1, 1)
    x = meta['Ammonia'].values.reshape(-1, 1)
    plt.scatter(x,y,label = 'Measurement')
    x_vals = x
   
    #Fit a quadratic model to the ammonia concentration
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    #print(x_vals)
    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num = 50)
    plt.plot(x_vals_smooth, reg.intercept_[0] + x_vals_smooth*reg.coef_[0][0] + x_vals_smooth*x_vals_smooth*reg.coef_[0][1], label =  'ammonia fit')
    plt.xlabel('Ammonia concentration (mM)')
    plt.ylabel('OD measurement (650nm)')
    plt.title('Ammonia (Salicylate-Hypochlorite assay) after removing well quadrant w/ bubble, y='+str(round(reg.intercept_[0],2))+ '+'+str(round(reg.coef_[0][0],2))+'x+'+str(round(reg.coef_[0][1],2))+'x^2'+', Adjusted R^2 =' +str(np.round(1 - (1-reg.score(x, y))*(len(y)-1)/(len(y)-x.shape[1]-1),3)))
    plt.legend()
    #plt.show()
    return

def fit_ammonia(meta_fn, wavelength, data_fn=None, data_absorb_fn=None, data_900_fn=None): 
    # fitting without bubble removal
    meta = pd.read_csv(meta_fn,index_col=0).dropna(how='all')
    #ammonia = read_ammonia(meta_fn, wavelength, data_fn)
    ammonia = read_ammonia(meta_fn, wavelength, data_fn, data_absorb_fn, data_900_fn)
    
    #remove nan values
    # old code: nan_idx = meta['Ammonia'].index[meta['Ammonia'].apply(np.isnan)]
    # new (1/31/22 Kiseok)
    nan_idx = ammonia[ammonia.apply(np.isnan)].index
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    ammonia = ammonia.drop(nan_idx)
    print(ammonia)

    #subtract blank values
    blank_idx = meta.index[(meta['Ammonia']==0)].tolist()
    #print(blank_idx)
    ammonia_blank = ammonia.loc[blank_idx].median()
    ammonia = ammonia - ammonia_blank
   
    #ammonia standard curve, ammonia measurement
    y = ammonia.values.reshape(-1, 1)
    x = meta['Ammonia'].values.reshape(-1, 1)
    x_vals = x
   
    #Fit a quadratic model to the ammonia concentration
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]

    
    # fitting with bubble removal with well scan
    #print(data_fn)
    data_fn = None
    #print(data_fn)
    ammonia = read_ammonia(meta_fn, wavelength, data_fn, data_absorb_fn, data_900_fn)
    
    #remove nan values
    nan_idx = meta['Ammonia'].index[meta['Ammonia'].apply(np.isnan)]
    meta = meta.drop(nan_idx)
    ammonia = ammonia.drop(nan_idx)
    
    #subtract blank values
    blank_idx = meta.index[(meta['Ammonia']==0)].tolist()
    #print(blank_idx)
    ammonia_blank = ammonia.loc[blank_idx].median()
    ammonia = ammonia - ammonia_blank
   
    #ammonia standard curve, ammonia measurement
    y = ammonia.values.reshape(-1, 1)
    x = meta['Ammonia'].values.reshape(-1, 1)
    x_vals = x
   
    #Fit a quadratic model to the ammonia concentration
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    b_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    
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
    
    
    fit_list = [[ammonia_blank], fit, b_fit]

    return fit_list


def fit_ammonia_without_bubble_removal(meta_fn, wavelength, data_fn=None, data_absorb_fn=None, data_900_fn=None): 
    # fitting without bubble removal
    meta = pd.read_csv(meta_fn,index_col=0).dropna(how='all')
    ammonia = read_ammonia(meta_fn, wavelength, data_fn)
    #ammonia = read_ammonia(meta_fn, wavelength, data_fn, data_absorb_fn, data_900_fn)
    
    #remove nan values
    # old code: nan_idx = meta['Ammonia'].index[meta['Ammonia'].apply(np.isnan)]
    # new (1/31/22 Kiseok)
    nan_idx = ammonia[ammonia.apply(np.isnan)].index
    print("These wells had NaN values: ", nan_idx.tolist())
    meta = meta.drop(nan_idx)
    ammonia = ammonia.drop(nan_idx)

    #subtract blank values
    blank_idx = meta.index[(meta['Ammonia']==0)].tolist()
    #print(blank_idx)
    ammonia_blank = ammonia.loc[blank_idx].median()
    ammonia = ammonia - ammonia_blank
   
    #ammonia standard curve, ammonia measurement
    y = ammonia.values.reshape(-1, 1)
    x = meta['Ammonia'].values.reshape(-1, 1)
    x_vals = x
   
    #Fit a quadratic model to the ammonia concentration
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
   
    ## Let's get the p-value (Source: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression)
    print("\n This fitting is done without bubble removal of well scan data")
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
    
    fit_list = [[ammonia_blank], fit]

    return fit_list


def get_concentration_xlsx(excel_output, ammonia_blank, fit, meta_fn, wavelength, data_fn=None, data_absorb_fn=None, data_900_fn=None):
    # Check for wellscan outlier removal
    print("Check wellscan for outlier removal in Ammonia")
    df_check_am = bd.read_abs_wellscan(data_900_fn)
    bd.plot_heatmap_wellscan(df_check_am, title = "Ammonia")
    df_check_am_650 = bd.read_abs_wellscan(data_absorb_fn)  # For 650nm
    bd.check_650_heatmap_wellscan(df_check_am_650, title="Ammonia") # For 540nm
    
    # read ammonia
    print(meta_fn)
    print("Currently using this fit: ",fit)
    print("Before subtracting blank value")
    Ammonia_OD650_plus_blank = read_ammonia(meta_fn, wavelength, data_fn, data_absorb_fn, data_900_fn).rename("Ammonia_OD650")
    print(Ammonia_OD650_plus_blank)

    # subtract blanks
    print("Subtracted blank value")
    Ammonia_OD650 = Ammonia_OD650_plus_blank - ammonia_blank
    print(Ammonia_OD650)
    # turn negative values to 0
    Ammonia_OD650[Ammonia_OD650<0] = 0.0

    #Returns inferred concentrations
    print("Use the fit to infer Ammonia mM")
    Ammonia_mM = ((-fit[1] + np.sqrt(fit[1]**2 - 4*(fit[0]- Ammonia_OD650 )*fit[2]))/2/fit[2]).rename("Ammonia_mM") ## solve quadratic formula
    Ammonia_mM[Ammonia_mM<0] = 0.0 # make it zero if it is negative
    print(Ammonia_mM)

    # combine it into a dataframe
    df_read = pd.DataFrame()
    df_read = pd.concat([Ammonia_OD650, Ammonia_mM], axis=1)

    df_meta = pd.read_csv(meta_fn,index_col=0).dropna()
    print(df_meta)
    df_out = pd.concat([df_meta, df_read], axis=1)
    df_out.to_excel(excel_output) 
    return

def get_concentration(ammonia_blank, fit, meta_fn, wavelength, data_fn=None, data_absorb_fn=None, data_900_fn=None):
    # Check for wellscan outlier removal
    print("Check wellscan for outlier removal in Ammonia")
    df_check_am = bd.read_abs_wellscan(data_900_fn)
    bd.plot_heatmap_wellscan(df_check_am, title = "Ammonia")
    df_check_am_650 = bd.read_abs_wellscan(data_absorb_fn)  # For 650nm
    bd.check_650_heatmap_wellscan(df_check_am_650, title="Ammonia") # For 540nm
    
    # read ammonia
    print(meta_fn)
    print("Currently using this fit: ",fit)
    print("Before subtracting blank value")
    Ammonia_OD650_plus_blank = read_ammonia(meta_fn, wavelength, data_fn, data_absorb_fn, data_900_fn).rename("Ammonia_OD650")
    print(Ammonia_OD650_plus_blank)

    # subtract blanks
    print("Subtracted blank value")
    Ammonia_OD650 = Ammonia_OD650_plus_blank - ammonia_blank
    print(Ammonia_OD650)
    # turn negative values to 0
    Ammonia_OD650[Ammonia_OD650<0] = 0.0

    #Returns inferred concentrations
    print("Use the fit to infer Ammonia mM")
    Ammonia_mM = ((-fit[1] + np.sqrt(fit[1]**2 - 4*(fit[0]- Ammonia_OD650 )*fit[2]))/2/fit[2]).rename("Ammonia_mM") ## solve quadratic formula
    Ammonia_mM[Ammonia_mM<0] = 0.0 # make it zero if it is negative
    print(Ammonia_mM)

    # combine it into a dataframe
    df_read = pd.DataFrame()
    df_read = pd.concat([Ammonia_OD650, Ammonia_mM], axis=1)

    df_meta = pd.read_csv(meta_fn,index_col=0).dropna()
    print(df_meta)
    df_out = pd.concat([df_meta, df_read], axis=1)
    return df_out