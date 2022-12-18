import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# Data processing
duration = np.array([1.925 + i*0.05 for i in range(13)])
N = np.array([19,19,39,48,87,94,104,92,57,44,28,26,13])
ddof = 13

def total_duration(duration,N):
    duration_tot = []
    for i in range(len(N)):
        for j in range(N[i]):
            duration_tot.append(duration[i])
            
    return duration_tot

data_total = total_duration(duration,N)
# print(total_duration(duration,N))


# Mean
data_mean = np.mean(data_total)
#print(data_mean)


#STD
data_STD =  np.std(data_total)
#print(data_STD)


# Density distributions
count, bins, patches = plt.hist(data_total,bins=13,density=True,fill=False,color='blue',label='Data distribution')
plt.plot(bins, stats.norm.pdf(bins,data_mean,data_STD),color='red',label='Normal distribution of\nm = 2.216\nsigma = 0.135')

plt.xlabel('Time slots')
plt.ylabel('Amount of flights')
plt.grid()
plt.title('Distribution comparisons')
plt.legend()
plt.show()

#CDF
cumulative_prob_normal = stats.norm.cdf(bins,data_mean,data_STD)
cumulative_prob = np.cumsum(count/sum(count))

plt.plot(duration,cumulative_prob,'-.',label='Flights CDF')
plt.plot(bins,cumulative_prob_normal,'-.',label='Normal law CDF',color='red')

plt.xlabel('Effective')
plt.ylabel('Cumulative Probability')
plt.title('Comparison of our data the Normal law CDF')
plt.legend()
plt.grid()
plt.show()
print(sum(N))


# Normalized normal law
B = 0.05*670*stats.norm.pdf(duration,data_mean,data_STD)
Bnorm=B*sum(N)/sum(B)
print(Bnorm)


# D2 coeff
D2 = sum((N-Bnorm)**2/B)
print(D2)
print('________')
results = stats.chisquare(N,f_exp=Bnorm,ddof=12)
print(results)
