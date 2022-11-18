import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

nb=1000000


Value=1+np.random.uniform(-0.02,0.02,nb)

plt.hist(Value,bins=250)

print("the mean is",np.mean(Value),"the satndart deviation is",np.std(Value))




Value_n=np.random.normal(1,0.02/1.96,size=nb)



count, bins, patches =plt.hist(Value_n)



plt.show()











plt.plot(bins,stats.norm.pdf(bins,1,0.02/1.96))
plt.hist(Value_n,bins=(250))

plt.show()
