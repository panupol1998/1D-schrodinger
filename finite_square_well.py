'''

The finite square well in 1 dimension
conditions:
box length = L
potential(V) = 0, 0.3L<=x<=0.7L & 30, otherwise
wave function : psi(x=0) = psi(x=L) = 0

'''

import numpy as np
import matplotlib.pyplot as plt

### define function to normalize wave function
def norm_psi(psi,x):
    temp = 0
    N = x.size-1
    delta_x = x[1]-x[0]
    for i in range(1,N):
        temp += psi[i]**2.0
    ans = delta_x/2.0*(psi[0]**2.0+psi[N]**2.0+2.0*temp)
    return -psi/np.sqrt(ans)

L = 10
N = 200
x_points = np.linspace(0,L,N+1)
delta_x = L/N
H = np.zeros((N-1,N-1))
V = np.zeros(N-1)

### define potential energy operator (V) in matrix form
for i in range(N-1):
    x = (i+1)*delta_x
    if 0.3*L<=x<=0.7*L:
        V[i] = 0.0
    else:
        V[i] = 30.0
V_new = np.append(np.append(30,V),30)

### define hamiltonian operator (T+V) in matrix form
for i in range(1,N):
    index = i-1
    H[index,index] = 2.0*13.6*0.529**2.0/(delta_x*delta_x)+V[index]
    if (index-1>=0):
        H[index,index-1] = -13.6*0.529**2.0/(delta_x*delta_x)
    if (index+1<N-1):
        H[index,index+1] = -13.6*0.529**2.0/(delta_x*delta_x)

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

### visualization
for state in range(1,5):
    wave = np.zeros(N+1)
    wave[0] = 0
    wave[N] = 0
    for i in range(1,N):
        wave[i] = vec[i-1][state-1]
    norm_wave = norm_psi(wave,x_points) + val[state-1] # increase the enegy to wave function 
    plt.plot(x_points,norm_wave,label='{} state'.format(state))
plt.plot(x_points,V_new,label='potential')
plt.legend()
plt.ylabel("energy")
plt.xlabel("x")
plt.show()