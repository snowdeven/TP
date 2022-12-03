


import numpy as np
from numpy import linalg as LA
from scipy import linalg as LA2
import matplotlib.pyplot as py
import mpl_toolkits.mplot3d as mp3d


X_MAX=10
M=1
N_STEP=100
DELTA_X=X_MAX/N_STEP
XD=[i*DELTA_X for i in range(N_STEP)]
T_MAX=16
N_TIME=300
DELTA_T=T_MAX/N_TIME
SIGMA_X=0.5
XT=[i*DELTA_T for i in range(N_TIME)]

Psi=np.zeros((N_STEP,N_TIME),dtype="complex")


