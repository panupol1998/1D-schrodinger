'''

The infinite square well in 1 dimension
conditions:
box length = L
potential(V) = 0, 0<=x<=L & inf, otherwise
wave function : psi(x=0) = psi(x=L) = 0

'''

import numpy as np
import matplotlib.pyplot as plt

L = 1.0
N = 100
state = 5
x_points = np.linspace(0,L,N+1)
delta_x = L/N 
H = np.zeros((N-1,N-1))

### define hamiltonian operator in matrix form
for i in range(1,N):
    index = i-1
    H[index,index] = 2.0/(delta_x*delta_x)
    if (index-1>=0):
        H[index,index-1] = -1.0/(delta_x*delta_x)
    if (index+1<N-1):
        H[index,index+1] = -1.0/(delta_x*delta_x)

### slove the eigen equation  
val, vec = np.linalg.eig(H)

### sort the enegy (eigen value) and wave function (eigen vector)
for i in range(val.size):
    for j in range(i+1,val.size):
        if (val[i]>val[j]):
            temp = val[i]
            val[i] = val[j]
            val[j] = temp
            for k in range(val.size):
                temp = vec[k][i]
                vec[k][i] = vec[k][j]
                vec[k][j] = temp
                
### define wave function            
wave = np.zeros(N+1)
wave[0] = 0
wave[N] = 0
for i in range(1,N):
    wave[i] = vec[i-1][state-1]

### normalize wave function
temp = 0
for i in range(1,N):
    temp += wave[i]**2.0
ans = delta_x/2.0*(wave[0]**2.0+wave[N]**2.0+2.0*temp)
norm_wave = -wave/np.sqrt(ans)
prob = norm_wave**2.0

### visualization
plt.subplot(1,2,1)
plt.plot(x_points,norm_wave)
plt.ylabel(f"$\psi_{state}(x)$")
plt.xlabel("x")
plt.subplot(1,2,2)
plt.plot(x_points,prob)
plt.ylabel(f"Probability(x)")
plt.xlabel("x")
plt.show()