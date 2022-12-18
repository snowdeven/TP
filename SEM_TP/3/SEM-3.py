import numpy as np 
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import pandas as pd

import seaborn as sns





#define the path of the csv file of JODecathlon
path=os.path.join(os.path.dirname(__file__)
                , 'resultJODecathlon.csv') 


x=pd.read_csv(path,sep=";",header=0) #read the csv file from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
Sport_labels=list(x.head(0))[2:-1]   #keep the sport labels with removing other name of column not usefull
Names=list(x['Athlete'])             #keep the name of our athlete
M=x.to_numpy()                       #transforme dataframe into an array
M=M[:,2:]                            #remove the 2 first column
M=M[:,:-1]                           #remove the last column

n , m =np.shape(M) #get the shape of our matrix

M_av=[]            #define a empty list for our average
for i in range(m):
    M_av.append(np.mean(M[:,i]))

M_cov=np.zeros((n,m))           #define a array of zero 
for i in range(m):
    M_cov[:,i]=M[:,i]-M_av[i]   #centered reduced matrix
C=np.cov(M_cov,rowvar=False)    #take the covariance matrix of our centered reduced matrix 
                                #to get the covariance matrix 

R=np.corrcoef(C, rowvar=False)  #Compute the correlation coefficients 
                                # to get correlation coefficient matrix

V=LA.eigh(C)[1]   #Compute the eigenvector from the covariance matrix
                    #to get eigen vectors matrix 

Y=np.dot(M_cov,V)   #Compute the matrix product between covariance matrix and eigen vectors matrix
                        # to get the Karhunen-Loève transformed matrix of data

Z=Y[:,-3:]          #remove the two first column of Y matrix to get Z matrix

PHI = V.transpose() #Compute the transpose of eigen vectors matrix 
                        #to get the NxN rotation matrix

PSI=PHI[-3:,:]      #remove the two first line from the NxN rotation matrix
                    #to get the Phi matrix

M_trunc= M_av + np.dot(Z,PSI)   #compute the sum of average matrix and the tensor product bewteen
                                #Z and Psi matrix to get truncated matrix

##############################################################################
#                                                                            #
#                          ###################                               #
#                          #   QUESTION 1    #                               #
#                          #_________________#                               #
#                                                                            #
##############################################################################

Mcompar = (M - M_trunc)/M  #compare the trucated matrix with the original matrix


##############################################################################
#                                                                            #
#                          ###################                               #
#                          #   QUESTION 2    #                               #
#                          #_________________#                               #
#                                                                            #
##############################################################################

loss = sum(LA.eigh(C)[0][:-3])/sum(LA.eigh(C)[0]) #comppute the information lost in our computation

print("the coefficient of information loss is Q = ",round(loss,3),"%")

##############################################################################
#                                                                            #
#                          ###################                               #
#                          #   QUESTION 3    #                               #
#                          #_________________#                               #
#                                                                            #
##############################################################################
"""plot a color graph and a 3D graph for the R matrix
with plt.color and ax.bar3D (and a mesh grid )
to see the correlation between the coefficients
"""
fig, ax = plt.subplots()
plt.rcParams.update({'font.size': 7})
plt.pcolor(R,)
plt.colorbar()

plt.title("2D colorgraph of  the  correlation  coefficients  matrix  R")
ax.set_yticks(np.arange(len(Sport_labels)), labels=Sport_labels)
ax.set_xticks(np.arange(len(Sport_labels)), labels=Sport_labels)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")
plt.show()



fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

X = Y = np.arange(1,11)
x, y = np.meshgrid(X,Y)
x, y = x.ravel(), y.ravel()

top = R.ravel()
bottom = np.zeros_like(top)
width = depth = 1 * top

color = ['r' if height > 0 else 'b' for height in top ]
ax1.bar3d(x, y, bottom, width, depth, top, shade = True,color=color)
ax1.plot([0, 0], [0, 0], [-1, 1], color='white')
ax1.set_yticks(np.arange(len(Sport_labels)), labels=Sport_labels)
ax1.set_xticks(np.arange(len(Sport_labels)), labels=Sport_labels)
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

plt.title("3D graph of  the  correlation  coefficients  matrix  R")
plt.show()

##############################################################################
#                                                                            #
#                          ###################                               #
#                          #   QUESTION 4    #                               #
#                          #_________________#                               #
#                                                                            #
##############################################################################
"""plot a 3D graph of last 3 column of Y
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(19):
    ax.quiver(0, 0, 0, Z[i,0], Z[i,1], Z[i,2],alpha=0.6)
    ax.text3D(Z[i,0], Z[i,1], Z[i,2],Names[i],)
ax.set_xlim([-200, 200])
ax.set_ylim([-200, 200])
ax.set_zlim([-200, 200])
ax.set_xlabel("component 1")
ax.set_ylabel("component 2")
ax.set_zlabel("component 3")
plt.title("3D graph of the athletes")
plt.show()

##############################################################################
#                                                                            #
#                          ###################                               #
#                          #   QUESTION 5    #                               #
#                          #_________________#                               #
#                                                                            #
##############################################################################
"""plot a 3D graph of last 3 rows of PSI
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(10):
    ax.quiver(0, 0, 0,PSI[0,i],PSI[1,i],PSI[2,i],alpha=0.6)
    ax.text3D(PSI[0,i],PSI[1,i],PSI[2,i],Sport_labels[i],)

ax.set_xlabel("component 1")
ax.set_ylabel("component 2")
ax.set_zlabel("component 3")
ax.set_xlim([-0.70, 0.70])
ax.set_ylim([-0.70, 0.70])
ax.set_zlim([-0.70, 0.70])
plt.title("3D graph of the events")
plt.show()
