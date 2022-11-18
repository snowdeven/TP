import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


################### binonial distriution of N trials ###################


largeur = 1000
hauteur = 1000


N=np.random.rand(hauteur,largeur)

for i in range(hauteur):
  
    for j in range(largeur):
        
        if N[i][j] < 0.5:
            N[i,j] = 0
          
        else:
              N[i,j] = 1
           

     
K = [sum(N[i]) for i in range(hauteur)] 


CS=np.cumsum(K)/np.mean(K)

OC , nb_oc=np.unique(sorted(K),return_counts=True)

nb_p_oc=np.cumsum(nb_oc/hauteur)




print("the mean for our sum of",largeur,"trials is equal to ", np.mean(K))
print("the variance for our sum of",largeur,"trials is equal to ", np.var(K))


plt.hist(K,bins=150)
plt.xlim(min(K),max(K))
plt.ylabel("the frequency of each result in our N trials")
plt.xlabel("All the possible result for our N trials ")
plt.grid()
plt.show()

fig, ax = plt.subplots()
    
ax.add_patch(Rectangle((np.mean(OC)-25,0.025,), 100,0.95 ,color="yellow"))
    


    

plt.plot(sorted(OC),sorted(nb_p_oc))
plt.grid()
plt.show()


