'''

The periodic potential in 1 dimension

'''

import numpy as np
import matplotlib.pyplot as plt

Lx = 1.0
N = 100
state = 3
delta_x = Lx/N
H = np.zeros((N,N), dtype=complex)
k_points = np.linspace(-np.pi/Lx,np.pi/Lx,N+1) # wave vector in 1D brillouin zone
band = np.zeros((state,N+1), dtype=complex)
x_points = np.linspace(0,Lx,N+1)
V = -10.0*np.exp(-50.0*(x_points-Lx/2.0)**2.0) # define potential energy operator (V)

### define hamiltonian operator in matrix form
i = complex(0,1)
for index_k,k in enumerate(k_points):
    for index in range(N):
        H[index,index] = 2.0*13.6*0.529*0.529/(delta_x**2.0)+V[index+1]
        if index+1<N:
            H[index,index+1] = -13.6*0.529*0.529/(delta_x**2.0)
            H[index+1,index] = -13.6*0.529*0.529/(delta_x**2.0)
        else:
            H[0,index] = -13.6*0.529*0.529*np.exp(-i*k*Lx)/(delta_x**2.0) # periodic condition
            H[index,0] = -13.6*0.529*0.529*np.exp(+i*k*Lx)/(delta_x**2.0) # periodic condition

    val, vec = np.linalg.eig(H) ### slove the eigen equation 
    val.sort() # sort the enegy (eigen values)
    for n in range(state):
        band[n,index_k] = val[n]

### visualization
plt.subplot(1,2,1)
plt.plot(x_points,V)
plt.title('Periodic potential')
plt.xlabel('x (a)')
plt.ylabel('potential energy (eV)')

plt.subplot(1,2,2)
for n in range(state):
    plt.plot(k_points,band[n])
plt.xticks([-np.pi,0,np.pi],['$-\pi$',0,'$\pi$'])
plt.title('band structure')
plt.xlabel('wave vector (1/a)')
plt.ylabel('Energy (eV)')
plt.show()