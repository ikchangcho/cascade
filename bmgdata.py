#Karna Gowda
#0.0.1
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import griess as gr

# This function is for finding the row number in the excel that has search_str(string)
def search_rows_str(data, search_str, nth=1):
    ctr = 1
    for i in range(0,len(data)):
        if any(search_str in string for string in data[i]):
            if ctr == nth:
                search_idx = i
                break
            else:
                ctr = ctr + 1;
    return search_idx

# this function finds the "first" row number in the excel that has has well readings.
def find_first_well(data):
    letters = ["A","B","C","D","E","F","G","H"]
    numbers = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    wells = []
    for letter in letters:
        for number in numbers:
            wells.append(letter+number)
            
    start_idx = 0
    for i in range(0,len(data)):
        try:
            if data[i][0][0:3] in wells:
                start_idx = i
                break
        except:
            pass
            
    return start_idx


def read_abs_endpoint(file_name):
    with open(file_name, encoding='latin_1') as csv_file:
        data = list(csv.reader(csv_file, delimiter=','))

    #Find row containing number of wavelengths.
    idx = search_rows_str(data, "No. of Channels / Multichromatics:")
    row = data[idx].copy()
    n_wl = [int(i) for i in row[0].split() if i.isdigit()][0] #extract number of wavelengths

    #Find row(s) containing wavelength range information.
    idx = search_rows_str(data,"nm")
    row = data[idx].copy()
    if "..." in row[0]:
        wl_range = [int(i) for i in row[0].replace('...',' ').replace('nm','').split() if i.isdigit()] #extract wavelength range
        wl = np.linspace(wl_range[0],wl_range[1],n_wl).astype('int')
    else:
        wl = np.zeros(n_wl).astype('int')
        for j in range(0,n_wl):
            idx = search_rows_str(data,"nm",nth=j+1)
            row = data[idx].copy()
            wl[j] = [int(i) for i in row[0].replace('nm','').split() if i.isdigit()][0]

    #Find first data row and fill in data starting at that row.   
    start_idx = find_first_well(data)
    for idx in range(start_idx,len(data)-1):
        row = data[idx].copy()
    
        if idx == start_idx:
            temp_exists = any("T" in string for string in row[-1]) #determine whether there is a temperature field
            row_label = [row[0][0:3]]
        else:
            row_label.append(row[0][0:3])

        row[0] = row[0][5:] #strip out plate index numbering
    
        if temp_exists:
            row[-1] = row[-1][7:]
    
        try:
            if 'data_array' in locals():
                data_array = np.vstack((data_array,np.asarray(row, dtype=np.float64, order='C')))
            else:
                data_array = np.asarray(row, dtype=np.float64, order='C')
        except:
            row_label.pop(-1) #if row contains unparseable data, remove label

    #Turn into data frame.
    if temp_exists:
        df = pd.DataFrame(data_array,columns = np.append(wl,'T'),index = row_label)
    else:
        df = pd.DataFrame(data_array,columns = wl.astype('str'),index = row_label)
    
    return df
    
def read_abs_wellscan(file_name):
    '''
    load well scan data
    :param meta_fn:
    :param data_fn:
    :param data_540_fn:
    :param data_900_fn:
    :return: N x 4 dataframe
    '''
    #print(file_name)
    with open(file_name,encoding='latin_1') as csv_file:
        data = list(csv.reader(csv_file, delimiter=','))

    #Find first data row and fill in data starting at that row.   
    start_idx = find_first_well(data)
    row_label = []
    #print('above the loop')
    for idx in range(start_idx,len(data)-1,5):
        row = data[idx].copy()
        row_label.append(row[0][0:3])
        
        #assumes a 2x2 well scan
        row = data[idx+2].copy()
        row.append(data[idx+3][0])
        row.append(data[idx+3][1])
        #print('above the try')        
        try:
            #print('in the try')
            
            if 'data_array' in locals():
                #print(row)
                row[:] = [x for x in row if x] #removes blanks ('') which show up in some data and cause an error
                #print(row)
                #print('in the if')
                data_array = np.vstack((data_array,np.asarray(row, dtype=np.float64, order='C')))
            else:
                #print('in the else')
                #print(row)
                row[:] = [x for x in row if x] #removes blanks ('') which show up in some data and cause an error
                #print(row)
                data_array = np.asarray(row, dtype=np.float64, order='C')
        except:
            #print('in the except')
            row_label.pop(-1) #if row contains unparseable data, remove label

    #Turn into data frame.
    df = pd.DataFrame(data_array,index = row_label)
    
    return df

# def plot_heatmap_wellscan(df, title="NO2"):
#     # Set up the figure and axes
#     fig, ax = plt.subplots(figsize=(26, 4))
#     # Generate the heatmap using seaborn
#     vmin = np.nanmin(df.values) # df.values.min()
#     vmax = np.nanmax(df.values)
#     heatmap = sns.heatmap(df.transpose(), cmap='RdYlGn_r', annot=True, linewidths=0.5, ax=ax, annot_kws={"rotation": 90},
#                          vmin=vmin, vmax=vmax)
#     ax.set_title('[Before removing bubble, '+title +"] Heatmap of wellscan") # Set the title
#     plt.show() # plot
    
#     # find outliers
#     bub_900 = gr.find_outliers(df)
#     df[bub_900] = np.nan
#     heatmap = sns.heatmap(df.transpose(), cmap='RdYlGn_r', annot=True, linewidths=0.5, ax=ax, annot_kws={"rotation": 90},
#                          vmin=vmin, vmax=vmax)
#     ax.set_title('[After removing bubble, '+title +"] Heatmap of wellscan") # Set the title
#     plt.show() # plot



def plot_heatmap_wellscan(df, title="NO2"):
    # Set up the figure and axes
    fig, axes = plt.subplots(2, 1, figsize=(26, 10))

    # Generate the heatmap using seaborn - Before removing bubble
    vmin = np.nanmin(df.values)
    vmax = np.nanmax(df.values)
    ax1 = axes[0]
    heatmap1 = sns.heatmap(df.transpose(), cmap='RdYlGn_r', annot=True, linewidths=0.5, ax=ax1, annot_kws={"rotation": 90},
                          vmin=vmin, vmax=vmax)
    ax1.set_title('[Before removing 900nm outliers in '+title+"] Heatmap of wellscan") # Set the title

    # Find outliers
    bub_900 = gr.find_outliers(df)
    df[bub_900] = np.nan

    # Generate the heatmap using seaborn - After removing bubble
    ax2 = axes[1]
    heatmap2 = sns.heatmap(df.transpose(), cmap='RdYlGn_r', annot=True, linewidths=0.5, ax=ax2, annot_kws={"rotation": 90},
                          vmin=vmin, vmax=vmax)
    ax2.set_title('[After removing 900nm outliers in '+title+"] Heatmap of wellscan") # Set the title

    # Adjust the layout
    fig.tight_layout()

    # Show the plot
    plt.show()
       
def check_540_heatmap_wellscan(df, title="NO2"):
    # Set up the figure and axes
    fig, axes = plt.subplots(figsize=(26, 4))

    # Generate the heatmap using seaborn - Before removing bubble
    vmin = np.nanmin(df.values)
    vmax = np.nanmax(df.values)
    heatmap = sns.heatmap(df.transpose(), cmap='RdYlGn_r', annot=True, linewidths=0.5, ax=axes, annot_kws={"rotation": 90},
                          vmin=vmin, vmax=vmax)
    axes.set_title('[540nm values in '+title+"] Heatmap of wellscan") # Set the title
    # Show the plot
    plt.show()
    