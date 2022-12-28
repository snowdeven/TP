


import numpy as np
from numpy import linalg as LA
from scipy import linalg as LA2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mp3d

# Toggle the potential 
# 1: wave packet in a box
# 2: in front of a potential barrier
# 3: Quantum Fermi accelerator
POT=3

# Space discretization
X_MAX=20
N_STEP=250
DELTA_X=X_MAX/N_STEP

# Space axis
XD=[i*DELTA_X for i in range(N_STEP)]

# Particle mass
M=1

# Time discretization
T_MAX=30
N_TIME=250
DELTA_T=T_MAX/N_TIME

# Time axis
XT=[i*DELTA_T for i in range(N_TIME)]

# Initial momentum
K0=10*np.sqrt(6)

# STD of gaussian packet
SIGMA_X=0.5

# Potential period
TAU=15

# Wave packet period
P=2


x = np.linspace(0,20,10000)

plt.plot(x,np.sin(10*x*np.pi/20)**2)
plt.show()
