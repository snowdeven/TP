import numpy as np
import matplotlib.pyplot as plt

N = 1000
repetitions = 5000
p = 0.5

def poll(N, repetitions, p):

    Nb_Ones = np.zeros(repetitions)
    vote_tot = np.zeros((N, repetitions))

    for i in range(repetitions):

        vote_tot[:,i] = np.random.binomial(1, p, N)
        Nb_Ones[i] = np.count_nonzero(vote_tot[:,i] == 1)

    return(vote_tot,Nb_Ones)



count, bins, patches = plt.hist(poll(N,repetitions,p)[1],bins=100,range=(400,600))
plt.close()

cumulative = np.cumsum(count/repetitions)

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



