#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 20:25:28 2021

@author: luiztheodoro
"""

#import math
import numpy as np
import multiprocessing as mp
from scipy import linalg as linalg
from itertools import repeat
import matplotlib.pyplot as plt

#função f
def ff(delta,r,X):
    ret = (1.0-r)*X + delta*(X**3)
    return ret

#função df
def dff(delta,r,X):
    ret = (1.0-r) + 3*delta*(X**2)
    return ret

#função g
def fg(delta,r,Z,Z_0):
    ret = np.full_like(Z,0.0)
    ret[0,:] = 0.0
    ret[1,:] = 0.0
    ret[2,:] = -ff(delta,r,Z[0,:]) + dff(delta,r,Z_0[0])*Z[0,:]
    return ret

def erk2(h=0.0, N=0.0, M=0.0, m=0.0, params=[], Z_0=np.zeros((3,1))):
    beta, eta, alpha, delta, r, sigma = params
    tpdist = np.vectorize(lambda x: 0.0 if x < 2/3 \
                           else 3.0**0.5 if 2/3 <= x < 5/6 \
                           else -(3.0**0.5))
    Z_temp = Z_0*np.ones((3,M))
    EZq = np.zeros((3,N))
    A = np.array([[0.0,0.0,1.0],[0.0,-beta,1.0],[-dff(delta,r,Z_0[0]),-alpha,-eta]])
    Ah = h*A
    eAh = linalg.expm(Ah)
    for j in range(1,N):
        zeta = tpdist(np.random.random((m,M)))
        b = np.append(np.zeros((2,M)), np.dot(sigma,zeta), axis=0)
        g1 = fg(delta, r, Z_temp, Z_0)
        K = g1 + (h**0.5) * b
        eAhZ = np.matmul(eAh, Z_temp)
        AK = np.matmul(A,K)
        g2 = fg(delta, r, eAhZ + K, Z_0)
        Z_temp = eAhZ + (h/2)*(g1 + g2 + AK) + (h**0.5)*b
        EZq[:,j] = np.mean(Z_temp**2, axis=1)
    return EZq
    

def main():
    np.random.seed(100)
    np.seterr('raise')
    
    m = 1
    beta = 0.5
    eta = 0.02
    alpha = 0.5625
    delta = 0.0
    r = 0.5
    sigma = 0.1*np.ones((1,m))
    
    params = [beta, eta, alpha, delta, r, sigma]
    
    T = 1000
    M = 1000
    P = 5
    
    Z_0 = np.array([[0.0],[0.0],[0.0]])
    EZq_erk2 = np.zeros((3,P))
    
    pool = mp.Pool(P)
    iterable = list(zip((2**(p-P) for p in range(P)),\
                        (int(T/(2**(p-P))) for p in range(P)),\
                        repeat(M),repeat(m),repeat(params),repeat(Z_0)))
    result = pool.starmap(erk2,iterable)
    
    for idx, val in enumerate(result): EZq_erk2[:,idx] = np.mean(val,axis=1)
    
    #EZq_erk2[:,p] = np.mean(result[p],axis=1) for p in range(P)

    # for p in range(P):
    #     h = 2**(p-P)
    #     N = int(T/h)
    #     EZq = erk2(h, N, M, m, params, Z_0)
    #     EZq_erk2[:,p] = np.mean(EZq, axis=1)
    
    EZq_true = np.zeros((3,1))
    
    d = 1-r+(beta**2)+beta*eta
    D = eta*(1-r+(beta**2)+beta*eta) + alpha*(beta+eta)
    
    EZq_true[0,0] = ((sigma**2)/(1-r))*d/D
    EZq_true[1,0] = ((sigma**2)/2)*1/D
    EZq_true[2,0] = ((sigma**2)/2)*d/D
    EZq_err = abs(EZq_erk2 - EZq_true)
    Dtvals = 2.**(np.arange(0,P,1)-6)
    
    plt.loglog(Dtvals, EZq_err[0,:], 'b*-')
    plt.loglog(Dtvals, EZq_err[1,:], 'g*-')
    plt.loglog(Dtvals, EZq_err[2,:], 'y*-')
    plt.loglog(Dtvals, Dtvals, 'r--')
    plt.axis([1e-2, 1, 1e-4, 1])

if __name__ == "__main__":
    main()