import numpy as np
from numpy import linalg as LA
from scipy import linalg as LA2
import matplotlib.pyplot as py
import mpl_toolkits.mplot3d as mp3d
from settings import *


def Hamiltonian(V,TD):
    HK=np.zeros((N_STEP,N_STEP),dtype="complex")
    for i in range(N_STEP):
        for j in range(N_STEP):
            if i == j :
                HK[i,j]=np.pi**2*((N_STEP+1)**2+2)/(3*X_MAX**2*2*M)
            else:
                HK[i,j]=(-1)**(j-i)*2*np.pi**2/(X_MAX**2*np.sin((j-i)*np.pi/(N_STEP+1))**2*2*M)
    if TD==True:
        return HK
    else:
        Vmat=LA.eigvalsh(V)
        return HK +Vmat
    
def Propagator(Psi,H,TD,V):
    expH=LA2.expm(-1j*H*DELTA_T)
    print(expH)
    for i in range(N_TIME-1):
        if TD==True:
            for j in range(N_STEP):
                expV=1
            # LA2.expm(-1j*V[j,i])
            expH=expV @ expH @ expV
            Psi[:,i+1]=np.dot(expH,Psi[:,i])
        else:
            Psi[:,i+1]=np.dot(expH,Psi[:,i])
    return Psi


def V(F):
    V=np.zeros((N_STEP,N_STEP))
    
    for i in range(N_STEP):
        V[i,i]=F(XD[i])
    
    
    return V

def F(x):
    return 0