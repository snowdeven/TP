import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def GAUSS3000(x,a,b,c):
    return a*np.exp((-(x-b)**2)/(2*c**2))



T_DATA=[1.925,1.975,2.025,2.075,2.125,2.175,2.225,2.275,2.325,2.375,2.425,2.475,2.525]
Y_DATA=[19,19,39,48,87,94,104,92,57,44,28,26,13]
DATA=[]
z=0

for i in Y_DATA:
    for j in range(i):
        
        DATA.append(T_DATA[z])
    z +=1


X=np.linspace(1.925,2.525,100)
th_coef=np.max(Y_DATA),np.mean(DATA),np.std(DATA)
fit_coef, pcov = curve_fit(GAUSS3000, T_DATA, Y_DATA)
uncertainty = np.sqrt(np.diag(pcov))


plt.scatter(T_DATA,Y_DATA,label="Scatter plot")
plt.plot(T_DATA, GAUSS3000(T_DATA, *fit_coef), 'r-',
        label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(fit_coef))

plt.plot(X,GAUSS3000(X,*th_coef),label="Theoritical curve: a=%5.3f, b=%5.3f, c=%5.3f "% tuple(th_coef))
plt.xlabel("Time in hours")
plt.ylabel("Amplitude")
plt.title("Non-linear fit of a gaussian on fly Paris-Alger data")
plt.legend()
plt.grid()
plt.show()

print("The th_coef: Amplitude =%5.3f, Average = %5.3f , Standard deviation =%5.3f"% tuple(th_coef))
print("The fit_coef: Amplitude =%5.3f, Average = %5.3f , Standard deviation =%5.3f"% tuple(fit_coef))
print("Uncertainty on : Amplitude =%5.3f, Average = %5.3f , Standard deviation =%5.3f"% tuple(uncertainty))