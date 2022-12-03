import numpy as np 
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

M=np.array([[737,	1066,	934,	1045,	1123,	777,	822,	843,	880,	790],
[660,	933,	800,	987,	935,	794,	878,	830,	972,	937],
[685,	1013,	994,	964,	970,	758,	906,	754,	910,	695],
[709,	935,	897,	971,	886,	826,	794,	776,	941,	876],
[733,	992,	962,	925,	972,	809,	794,	811,	910,	696],
[690,	963,	858,	910,	862,	789,	850,	764,	1004,	744],
[593,	935,	851,	870,	871,	814,	822,	865,	880,	913],
[715,	876,	862,	920,	957,	787,	794,	809,	941,	752],
[759,	938,	909,	958,	886,	727,	850,	740,	880,	675],
[765,	931,	927,	921,	908,	753,	850,	679,	790,	711],
[755,	852,	897,	905,	900,	765,	794,	782,	849,	714],
[745,	899,	898,	856,	878,	784,	822,	834,	849,	612],
[664,	852,	851,	856,	797,	865,	767,	808,	941,	730],
[629,	841,	901,	882,	930,	757,	822,	720,	849,	795],
[689,	791,	777,	870,	903,	725,	822,	795,	1067,	598],
[651,	847,	842,	930,	852,	789,	767,	775,	790,	761],
[589,	821,	866,	913,	821,	767,	714,	868,	790,	794],
[604,	956,	847,	901,	905,	736,	822,	663,	790,	656],
[633,	892,	772,	794,	821,	727,	767,	836,	849,	772]]
)

n , m =np.shape(M)



M_av=[]



for i in range(m):

    M_av.append(np.mean(M[:,i]))

M_c=np.zeros((n,m))

for i in range(m):
    M_c[:,i]=M[:,i]-M_av[i]


C=np.cov(M_c,rowvar=False)




M_r=np.zeros((m,m))

for i in range(m):
    for j in range(m):
        M_r[i,j]=C[i,j]/(np.sqrt(C[i,i]*C[j,j]))

M_v=LA.eigh(C)[1]



M_y=np.dot(M_c,M_v)






M_z=M_y[:,-3:]





M_Psi = M_v.transpose()
M_Psi=M_Psi[-3:,:]





M_trunc= M_av + np.dot(M_z,M_Psi)




print(abs(M-M_trunc)/M_trunc *100)



fig = plt.figure()
ax = plt.axes(projection='3d')



ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zdata = M_z[:,2]
xdata = M_z[:,0]
ydata = M_z[:,1]
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

plt.show()