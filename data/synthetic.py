import numpy as np
import torch
from scipy.integrate import odeint
import math


def lorenz(y, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    m = len(y)
    dxdt = np.zeros(m)
    for i in range(m):
        dxdt[i] = (y[(i+1) % m] - y[(i-2) % m]) * y[(i-1) % m] - y[i] + F

    return dxdt


def simulate_lorenz_96( A_p ,T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=0):
    if seed is not None:
        np.random.seed(seed)

    m = len(A_p) # Number of series

    y0 = np.random.normal(scale=0.01, size=m)
    # Use scipy to solve ODE.
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    Y = odeint(lorenz, y0, t, args=(F,))
    Y += np.random.normal(scale=sd, size=(T + burn_in, m))

    X = np.zeros((Y.shape[0],A_p.sum().item()))

    # Randomly initalize coefficient A_ , A_*Y = X since we have Y = A*X

    for i in range(m):
        start_ind = A_p[0:i].sum().item()
        end_ind = A_p[0:i].sum().item()+A_p[i].item()
        
        if(A_p[i].item()==1):  
            Ai_ =1.0
        else:
            Ai_ = np.random.rand(A_p[i].item())
        X[:,start_ind:end_ind] = Y[:,i].reshape(-1,1) * Ai_ # (T+burn_in,1) *(1,A_p[i])
        

    # Set up Granger causality ground truth.
    GC = np.zeros((m, m), dtype=int)
    for i in range(m):
        GC[i, i] = 1
        GC[i, (i + 1) % m] = 1
        GC[i, (i - 1) % m] = 1
        GC[i, (i - 2) % m] = 1

    return X[burn_in:], GC



if __name__ == "__main__":

    for seed in range(0,10):
        A_p_list = [ [1,2,1,2,3,1,2,1,2,1,2,2], [1,3,2,4,3,3,2,2,3,1,4,2], [2,4,5,4,4,3,3,2,3,4,4,2], [4,4,5,5,4,6,3,5,3,4,4,3], [5,6,5,6,4,6,5,5,6,5,4,3],
                    [3,4,5,3,2,3], [2,4,5,4,4,3,3,2,3], [2,4,5,4,4,3,3,2,3,5,3,2,3,4,3], [2,4,5,4,4,3,3,2,3,5,3,2,3,4,3], [2,4,5,4,4,3,3,2,3,5,3,2,3,4,3,2,3,5]
                     ]

        for A_p in A_p_list:

            
            X_np, GC = simulate_lorenz_96( torch.tensor(A_p),T=1500,seed=seed)

            m = len(A_p)
            mean_X = np.zeros((X_np.shape[0],m))

            for i in range(m):
                start_ind = sum(A_p[0:i])
                end_ind = sum(A_p[0:i])+A_p[i]
                temp = X_np[:,start_ind:end_ind]
                mean_X[:,i] = np.mean(temp,axis=1)

            np.savetxt("synthetic/mean_{}_{}_{}.txt".format(A_p, X_np.shape[0], seed), mean_X) 

        A_p = [2,4,5,4,4,3,3,2,3,4,4,2]
        for T in [500, 1000, 2000, 2500]:
            X_np, GC = simulate_lorenz_96( torch.tensor(A_p),T=T,seed=seed)

            m = len(A_p)
            mean_X = np.zeros((X_np.shape[0],m))

            for i in range(m):
                start_ind = sum(A_p[0:i])
                end_ind = sum(A_p[0:i])+A_p[i]
                temp = X_np[:,start_ind:end_ind]
                mean_X[:,i] = np.mean(temp,axis=1)

            np.savetxt("synthetic/mean_{}_{}_{}.txt".format(A_p, X_np.shape[0], seed), mean_X) 

        


