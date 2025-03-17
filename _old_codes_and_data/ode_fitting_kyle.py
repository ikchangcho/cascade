result = dn.fitRates(params,monocultures[i])

def fitRates(params,experiments,n=1):            
    fitter = Minimizer(residualGlobLMFit, params, fcn_args=(experiments,))
    result_brute = fitter.minimize(method='brute')
    best_result = copy.deepcopy(result_brute)
    for candidate in result_brute.candidates:
        trial = fitter.minimize(method='leastsq', params=candidate.params)
        if trial.chisqr < best_result.chisqr:
            best_result = trial
    #compute 99% CI
    # ci = conf_interval(fitter, best_result,sigmas=[0.99])
    # return best_result, ci
    return best_result
# The Minimizer class from the lmfit library is used to perform optimization on a given objective function. 
# The output of the Minimizer's minimize method is an OptimizeResult object. This object contains several attributes that provide information about the optimization process and its results. Key attributes include:
# params: The optimized parameters.
# chisqr: The chi-square value of the fit.
# redchi: The reduced chi-square value.
# aic: The Akaike Information Criterion.
# bic: The Bayesian Information Criterion.
# success: A boolean indicating if the optimizer exited successfully.
# message: A string message providing information about the exit status.
# nfev: The number of function evaluations.
# residual: The residual array.
# In the context of your fitRates function, the result_brute and trial variables are OptimizeResult objects returned by the minimize method of the Minimizer class.

def residualGlobLMFit(params,experiments,n=1):
    return residualGlob(convertPTableToMat(params,n),experiments,n)

def convertPTableToMat(params,n=1):
    if n == 1:
        p_out = [[params['rA'].value, params['rI'].value, params['kA'].value, params['kI'].value, params['gamA'].value, params['gamI'].value]]
    else:
        p_out = np.zeros((n,6))
        for i in range(0,n):
            p_out[i,0] = params[i]['rA'].value
            p_out[i,1] = params[i]['rI'].value
            p_out[i,2] = params[i]['kA'].value
            p_out[i,3] = params[i]['kI'].value
            p_out[i,4] = params[i]['gamA'].value
            p_out[i,5] = params[i]['gamI'].value
    return p_out

def residualGlob(p,experiments,n=1):
    #Computes the residual for all conditions
    res_out = np.array([])
    for i in range(0,len(experiments)):
        res_out = np.append(res_out,residual(p,experiments[i],n))
    return res_out

def residual(p,experiment,n=1):
    #Compute the residual vector for the A and I variables using the replicate measurements taken in a given condition
    #print(experiment)
    y0 = np.append(experiment.N0, [np.nanmedian(experiment.A[:,0]), np.nanmedian(experiment.I[:,0])])
#     y0 = np.append(experiment.N0, [experiment.A0, experiment.I0])
    yh = denitODE(y0,experiment.t,p,n)
    res = np.ravel([experiment.A-yh[:,-2],experiment.I-yh[:,-1]])
    res = res[~np.isnan(res)] #remove any nan elements
    return res

def denitODE(y0,t,p,n=1):
    sol = odeint(F, y0, t, Dfun=J, args=(p,n), rtol=1e-6)
    return sol
# odeint is a function from the scipy.integrate module in the SciPy library. It is used to integrate ordinary differential equations (ODEs). 
# The function numerically solves a system of first-order ODEs given initial conditions and a time array.

def F(y,t,p,n):
    #ODE RHS
    #takes p as an ndarray where each row contains the parameters for a given strain
    Fout = np.zeros(n+2)
    for i in range(0,n):
        Fout[ i]  = f(p[i],[y[i],y[-2],y[-1]])
        Fout[-2] += g(p[i],[y[i],y[-2],y[-1]])
        Fout[-1] += h(p[i],[y[i],y[-2],y[-1]])
    return Fout

def f(p,y):
    #takes p as a 1darray
    rA     = p[0]
    rI     = p[1]
    kA     = p[2]
    kI     = p[3]
    gamA   = p[4]
    gamI   = p[5]
    return gamA*rA*y[0]*y[1]/(kA + y[1]) + gamI*rI*y[0]*y[2]/(kI + y[2])

def g(p,y):
    #takes p as a 1darray
    rA     = p[0]
    kA     = p[2]
    return -1*rA*y[0]*y[1]/(kA + y[1])

def h(p,y):
    #takes p as a 1darray
    rA     = p[0]
    rI     = p[1]
    kA     = p[2]
    kI     = p[3]
    return rA*y[0]*y[1]/(kA + y[1]) - rI*y[0]*y[2]/(kI + y[2])