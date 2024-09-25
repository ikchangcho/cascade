import numpy as np
import matplotlib.pyplot as plt

def get_max_growth_rate(x,t):  #x'/x
    dxdt = np.gradient(x,t)
    return np.max(dxdt/x)

def get_growth_rate(x,t):  #x'/x
    dxdt = np.gradient(x,t)
    return dxdt/x

def running_avg(x, t, N):  #note that you lose first and last (N-1)/2 points
    if N%2 == 0:
        raise NameError('N should be odd!')
    elif N == 1:  #don't do averaging
        return t, x
    else:
        return t[int((N-1)/2): -int((N-1)/2)], np.convolve(x, np.ones((N,))/N)[(N-1):-(N-1)] #see https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/43200476#43200476







##tests

#get_max_growth_rate test#
'''
t = np.arange(12)
x = np.asarray([1,1.1,1.2,1.4,4,8,16,20,21,22,23,24])


plt.plot(t,x, label = 'x')
plt.plot(t,np.gradient(x,t), label = 'dxdt')
plt.plot(t, np.gradient(x,t)/x, label = 'growth rate')
plt.title('max is ' + str(get_max_growth_rate(x,t)))
plt.legend()
plt.show()
'''


##running avg test
'''
x = np.asarray([1,3,5,7,9, 11,13, 15, 17, 19, 21])
t = np.arange(len(x))
print('x is ' + str(x))
print('t is ' + str(t))

print(running_avg(x, t, 3))
print(running_avg(x, t, 5))
print(running_avg(x, t, 7))
'''
