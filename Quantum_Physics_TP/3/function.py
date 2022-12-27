import numpy as np
from numpy import linalg as LA
from scipy import linalg as LA2
import matplotlib.pyplot as py
import mpl_toolkits.mplot3d as mp3d
from settings import *


#________________________________________________________
# Wave functions with respect to the system

def Psi(POT):
    """Returns the wave function with respect
    to a chosen system and to N_STEP/N_TIME.

    Args:
        POT (int): input system

    Returns:
        2D array: Psi, values of the wave function
    """
    PSI=np.zeros((N_STEP,N_TIME),dtype="complex")
    
    # Box
    if POT == 1 :
        # Exponential prefactor
        C = 1 / np.sqrt(SIGMA_X * np.sqrt(2*np.pi)) 
        
        for i in range(N_STEP):
            PSI[i,0] = C * np.exp(-(XD[i]-X_MAX/2)**2 / (4*SIGMA_X**2))
    
    # Barrier
    elif POT == 2:
        # Exponential prefactor
        C = 1 / np.sqrt(SIGMA_X * np.sqrt(2*np.pi)) 
        
        for i in range(N_STEP):
            PSI[i,0] = C * np.exp(-(XD[i]-10/2)**2/(4*SIGMA_X**2)) * np.exp(-1j*K0*XD[i])
    
    # Accelerator     
    elif POT == 3:
        # Exponential prefactor
        C = np.sqrt(2/X_MAX)
        
        for i in range(N_STEP):
            PSI[i,0] = C * np.sin(P*np.pi*XD[i] / X_MAX)
    
    # Wrong input
    else:
        return 'Please choose a valid system (1-3)'
    
    return PSI

#________________________________________________________
# Hamiltonian with respect to the selected system

def Hamiltonian(V):
    """Function building the selected system hamiltonian

    Args:
        V (float): Potential value in a given point

    Returns:
        2D array: System hamiltonian
    """
    
    HK = np.zeros((N_STEP,N_STEP),dtype="complex")
    
    # Looping over space steps
    for i in range(N_STEP):
        
        for j in range(N_STEP):
            
            if i == j :
                HK[i,j]=np.pi**2*((N_STEP+1)**2+2)/(3*X_MAX**2*2*M)
            else:
                HK[i,j]=(-1)**(j-i)*2*np.pi**2/(X_MAX**2*np.sin((j-i)*np.pi/(N_STEP+1))**2*2*M)
    
    # Checking if system is time-dependant
    if POT == 3:
        return HK
    
    # If time independant
    else:
        # Diagonal potential matrix
        Vmat=np.diag([V(XD[j],POT) for j in range(N_STEP)])
        
        # Hamiltonian
        H=HK +Vmat
        
        print("adjacency matrix is ",np.all(H == HK))
        return H



def Propagator(Psi,H,V):
    """Propagates H with respect to its time
    dependency

    Args:
        Psi (2D array): Wave function matrix to be computed
        H (2D array): System Hamiltonian
        V (float): Potential value in a given point

    Returns:
        2D array: Spacetime computed wave function
    """
    expH=LA2.expm(-1j*H*DELTA_T)
    
    for i in range(N_TIME-1):
        
        if POT == 3:
            
            expV=np.diag([np.exp(-1j*V(XD[j],POT,XT[i])*DELTA_T/2) for j in range(N_STEP)])
            expH=expV @ expH @ expV
            Psi[:,i+1]=np.dot(expH,Psi[:,i])
            
        else:
            
            Psi[:,i+1]=np.dot(expH,Psi[:,i])
            
    return Psi


def V(x,POT,t=0):
    """Potential in a given point

    Args:
        x (float): space coordinate
        POT (_type_): selected system
        t (float, optional): Time dependency. Defaults to 0.

    Returns:
        float: Potential at given space and time point if time dependent
    """
    
    # Box
    if POT == 1:
        return 0
    
    # Barrier
    elif POT == 2:
        x0=10
        x1=11
        
        if x0 < x < x1:
            v=3
            
        else:
            v=0
        return v
    
    # Fermi accelerator
    elif POT == 3:
        
        X_MIN=0.75*X_MAX
        w=2*np.pi/TAU
        born=X_MAX-((X_MAX-X_MIN)/2)*(1-np.cos(w*t))
        
        if x < born :
            return 0
        
        else:
            return 10**15
