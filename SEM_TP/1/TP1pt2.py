import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

mu = 1
sigma = 0.02/1.96
N = 1000
n = 1000 #amount of repetitions

def mes():
    return np.random.normal(mu,sigma,N)

def avgmes():
    avg = np.zeros(n)
    for i in range(n):
        avg[i] = np.mean(mes())
    return avg
A = avgmes()

print(np.std(A))

count, bins, patches = plt.hist(mes(),bins=int(0.1*N))
plt.plot(bins, stats.norm.pdf(bins,mu,sigma))
plt.show()

cumulative = np.cumsum(count/N)

j = (np.abs(cumulative - 0.975)).argmin()
i = (np.abs(cumulative - 0.025)).argmin()

lower_bound_x = bins[i]*np.ones(len(cumulative))
upper_bound_x = bins[j]*np.ones(len(cumulative))
lower_bound_y = cumulative[i]*np.ones(len(cumulative))
upper_bound_y = cumulative[j]*np.ones(len(cumulative))


plt.plot(lower_bound_x[:i],cumulative[:i],'--',color='red')
plt.plot(upper_bound_x[:j],cumulative[:j],'--',color='red')
plt.plot(bins[:i],lower_bound_y[:i],'--',color='red')
plt.plot(bins[:j],upper_bound_y[:j],'--',color='red')
plt.plot(bins[1:],cumulative)
plt.show()