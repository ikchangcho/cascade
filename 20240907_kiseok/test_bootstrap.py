#below is code to fit denitrification data for a single strain.
import sys
sys.path.append('/Users/kylecrocker/Documents/Research/microbial_ecology/20240907_kiseok')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import denitfit as dn
from lmfit import Parameters, report_fit
import pickle
from os.path import exists
from pprint import pprint
from pandas import read_csv


plt.rc('text', usetex=True)

monocultures = pickle.load(open( "/Users/kylecrocker/Documents/Research/microbial_ecology/enrichment_isolation/data/anaerobic_growth_curves/phenotypes_072021.pkl", "rb" ))

print('Each element of the list monocultures is a list of experiment objects associated with a single strain:')
print(monocultures[0])
#print('')
print('Each experiment object contains data from a given experiment (i.e. set of initial conditions) for the given strain, with multiple replicates:')
for experiment in monocultures[4]:
    pprint(vars(experiment))
    print('')
    print('')



#extract strain IDs and phenotypes from monocultures object
strain_id = []
phenotype = []
name_dict = np.asarray(read_csv('/Users/kylecrocker/Documents/Research/microbial_ecology/enrichment_isolation/phenotypes/strain_relabel_lookup_KC_08042021.csv', header = None))
for i in range(0,len(monocultures)):
    for j in range(len(name_dict[:])):
        if monocultures[i][0].ID == name_dict[j][0]:
            strain_id.append(name_dict[j])
    phenotype.append(monocultures[i][0].phen)
print(strain_id)
#FIT METABOLITE DATA
#if exists('pfit.npz'): #since it takes a while to fit all monoculture data, save it and load subsequently
if False:                                                                                                                                                                                                         
    pfit_test = np.load('pfit_test.npz')['arr_0']
    #gam_ses = np.load('pfit.npz')['arr_1']
    gam_r2s = np.load('pfit_test.npz')['arr_1']
    offsets = np.load('pfit_test.npz')['arr_2']
else:
    #pfit = np.zeros((4,len(monocultures)))                                                                                                                                                                      
    pfit = np.zeros((6,len(monocultures),1)) #need the third dimension for the call to plotDenitFit
    #gam_ses = np.zeros(len(monocultures))
    gam_r2s = np.zeros(len(monocultures))
    offsets = np.zeros(len(monocultures))
    gamA_min = np.zeros(len(monocultures))
    gamA_max = np.zeros(len(monocultures))

    gamI_min = np.zeros(len(monocultures))
    gamI_max = np.zeros(len(monocultures))

    #print(np.shape(pfit))
    for i in range(0,len(monocultures)):
        print(i)
    #for i in range(4): 
        params = Parameters()                                                                                                                                                                                            
        if monocultures[i][0].phen == 'Nar/Nir':
            params.add('rA', value=10.0, min=0, max=30,vary=True,brute_step=5)
            params.add('rI', value=10.0, min=0, max=25,vary=True,brute_step=5)
        elif monocultures[i][0].phen == 'Nar':
            params.add('rA', value=10.0, min=0, max=30,vary=True,brute_step=5)
            params.add('rI', value=0.0, min=0, max=25,vary=False)
        elif monocultures[i][0].phen == 'Nir':
            params.add('rA', value=0.0, min=0, max=30,vary=False)
            params.add('rI', value=10.0, min=0, max=25,vary=True,brute_step=5)
                                                                                                                                                 
        #Fit yield parameters                                                                                                                                                                                          
        gamA,gamI,gam_se,r2,offset = dn.fitYields(monocultures[i]) #fits across different initial conditions and replicates
        #print(gam_se)
        #print(r2)
        #gam_ses[i] = gam_se
        gam_r2s[i] = r2
        offsets[i] = offset
        gamA_min[i] = gam_se[0][0]
        gamA_max[i] = gam_se[0][1]
        gamI_min[i] = gam_se[1][0]
        gamI_max[i] = gam_se[1][1]
        #print(gamA)
        if gamA<0:
            gamA = 0
        if gamI<0:
            gamI = 0
        if gam_se[0,0]<0:
            gamA = 0
        if gam_se[1,0]<0:
            gamI = 0

        #Fit rate parameters                                                                       
        params.add('kA', value=1e-2, min=1e-3, max=1e1,vary=False) #fix affinity parameters as in paper. can also allow these to vary
        params.add('kI', value=1e-2, min=1e-3, max=1e1,vary=False)
        params.add('gamA', value=gamA, min=0, max=0.04,vary=False)
        params.add('gamI', value=gamI, min=0, max=0.04,vary=False)
        result = dn.fitRates(params,monocultures[i])
        pprint(vars(result))
        pprint(vars(result.params['rA']))

        #format fit parameters in array                                                  
        pfit[0,i,0]=result.params['rA'].value
        pfit[1,i,0]=result.params['rI'].value
        pfit[2,i,0]=result.params['kA'].value
        pfit[3,i,0]=result.params['kI'].value
        pfit[4,i,0]=result.params['gamA'].value
        pfit[5,i,0]=result.params['gamI'].value
        #print(pfit[0,i,0])
        #print(dn.fitRatesBootstrap(params,monocultures[i],n=1))
        

    np.savez('pfit_test', pfit, gam_r2s, offsets)
