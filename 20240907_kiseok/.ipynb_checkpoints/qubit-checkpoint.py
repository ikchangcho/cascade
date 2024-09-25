#Karna Gowda
#0.0.1

import bmgdata as bd
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


def plot_and_return_qubit_fit(meta_fn, std_qubit_fn):
    
    alist = [line.rstrip() for line in open(std_qubit_fn)]
    alist.remove("")

    l_well =[]
    l_value = []

    for i in alist:
        a1 = i.split(":")[0].rstrip()
        #print(a1)
        l_well.append(a1)
        a2 = i.split(":")[1].replace(" ", "").replace("-","")
        #print(a2)
        l_value.append(a2)

    df_read = pd.DataFrame({'Fluorometer': l_value}, index = l_well)
    df_read['Fluorometer'].replace("",np.nan, inplace = True)
    df_read = df_read.dropna()
    
    # import standard metadata
    meta = pd.read_csv(meta_fn,index_col=0,  encoding='latin-1').dropna(how='all')
    #meta = meta['DNA'] #keep only the column values
    fluoro = df_read.dropna(how='all')
    fluoro['Fluorometer'] = pd.to_numeric(fluoro['Fluorometer']) # convert string to numeric

    # remove NaN values
    nan_index = meta['DNA'].index[meta['DNA'].apply(np.isnan)].values.tolist()
    meta = meta.drop(nan_index)
    fluoro = fluoro.drop(nan_index)
    
    # subtract blank value to all well fluorometer reads!
    #blank_idx = meta.index[meta['DNA']==0].tolist()
    #blank_idx

    #fluoro_blank = fluoro.loc[blank_idx].median().values
    #fluoro_blank_median = fluoro_blank[0]
    #fluoro_blank_median

    #fluoro_blank_subtracted = fluoro - fluoro_blank_median
    #fluoro_blank_subtracted[fluoro_blank_subtracted<0] = 0.0
    
    # Plot standard curve
    std_idx = meta.index.tolist()
    y = fluoro.loc[std_idx].values.reshape(-1, 1)
    x = meta['DNA'].loc[std_idx].values.reshape(-1, 1)
    plt.scatter(x,y,label = 'Measurement of Fluorometer standards')
    x_vals = x
    len(x)

    #Fit a quadratic model to the nitrite concentration
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    q_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]
    #print(x_vals)
    x_vals_smooth = np.linspace(min(x_vals), max(x_vals), num = 50)
    plt.plot(x_vals_smooth, reg.intercept_[0] + x_vals_smooth*reg.coef_[0][0] + x_vals_smooth*x_vals_smooth*reg.coef_[0][1], label =  'quadratic model fit')
    plt.xlabel('DNA [ng/ul]')
    plt.ylabel('Fluorometer read value')
    plt.title('Qubit assay standard curve (2ul standards), y='+str(round(reg.intercept_[0],2))+ '+'+str(round(reg.coef_[0][0],2))+'x+'+str(round(reg.coef_[0][1],2))+'x^2'+', Adjusted R^2 =' +str(np.round(1 - (1-reg.score(x, y))*(len(y)-1)/(len(y)-x.shape[1]-1),3)))
    plt.legend()
    #plt.show()
    
    return q_fit

def get_concentration_xlsx(excel_output, q_fit, sample_qubit_fn, sample_meta_fn, library_size = 400):
    
    alist = [line.rstrip() for line in open(sample_qubit_fn)]
    alist.remove("")

    l_well =[]
    l_value = []

    for i in alist:
        a1 = i.split(":")[0].rstrip()
        #print(a1)
        l_well.append(a1)
        a2 = i.split(":")[1].replace(" ", "").replace("-","")
        #print(a2)
        l_value.append(a2)

    df_read = pd.DataFrame({'Fluorometer': l_value}, index = l_well)
    df_read['Fluorometer'].replace("",np.nan, inplace = True)
    df_read = df_read.dropna()

    fluoro = df_read.dropna(how='all')
    fluoro['Fluorometer'] = pd.to_numeric(fluoro['Fluorometer']) # convert string to numeric
    
    # I don't need to subtract blank because I already computed q_fit (standard curves)
    # subtract blank value to all well fluorometer reads!
    # blank_idx = meta.index[meta['DNA']==0].tolist()
    # blank_idx

    # fluoro_blank = fluoro.loc[blank_idx].median().values
    # fluoro_blank_median = fluoro_blank[0]
    # fluoro_blank_median

    # fluoro_blank_subtracted = fluoro - fluoro_blank_median
    # fluoro_blank_subtracted[fluoro_blank_subtracted<0] = 0.0
    # std_idx = meta.index.tolist()
    
    # use fitted curve to get DNA concentration
    DNA_ngul = (-q_fit[1] + np.sqrt(q_fit[1]**2 - 4*(q_fit[0]- fluoro )*q_fit[2]))/2/q_fit[2] ## solve quadratic formula
    DNA_ngul[DNA_ngul<0] = 0.0
    df_DNA_ngul = DNA_ngul # (12/19/22 Don't remove standard from plate)
    
    # change column name
    df_DNA_ngul.columns = ['DNA_ngul'] 
    
    ## calculate concentration in nM
    df_DNA_nM = df_DNA_ngul * (10**6) / (660 * library_size)
    # change column name
    df_DNA_nM.columns = ['DNA_nM']
    # output
    df_meta = pd.read_csv(sample_meta_fn,index_col=0,  encoding='latin-1').dropna(how='all')
    df_out = pd.concat([df_meta, df_DNA_ngul, df_DNA_nM], axis=1)
    df_out.to_excel(excel_output) 
    
    return
    

