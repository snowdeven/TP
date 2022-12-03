import numpy as np
from numpy import linalg as LA
from scipy import linalg as LA2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mp3d

from settings import *
from function import *

C=1/np.sqrt(SIGMA_X*np.sqrt(2*np.pi)) 
for i in range(N_STEP):
    Psi[i,0]=C*np.exp(-(XD[i]-X_MAX/2)**2/(4*SIGMA_X**2))


V=V(F)


H=Hamiltonian(V,False)

H=-1j*H*DELTA_T


print(H)

expH=LA2.expm(H)

print(expH)


# print(expH[0,0])

P=Propagator(Psi,H,False,V)

P=np.real((P* np.conjugate(P)))

plt.pcolormesh(XT,XD,P)
plt.colorbar()
plt.show()



