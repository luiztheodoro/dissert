#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 20:25:28 2021

@author: luiztheodoro
"""

#import math
import numpy as np
from scipy import linalg as linalg
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

#função K
# def fK(param,Z,Z_0,zeta,h,b):  
#     beta, eta, alpha, delta, r, sigma = param
#     ret = fg(delta,r,Z,Z_0)*h + (h**0.5) * b
#     return ret

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
    
    param = [beta, eta, alpha, delta, r, sigma]
    
    T = 10000
    h = 0.5
    N = int(T/h)
    M = 1000
    P=5
    
    tpdist = np.vectorize(lambda x: 0.0 if x < 2/3 \
                                      else 3.0**0.5 if 2/3 <= x < 5/6 \
                                      else -(3.0**0.5))
        
    Z_0 = np.array([[0.0],[0.0],[0.0]])
    A = np.array([[0.0,0.0,1.0],[0.0,-beta,1.0],[-dff(delta,r,Z_0[0]),-alpha,-eta]])
    Ah = h*A
    eAh = linalg.expm(Ah)
    EZq_erk2 = np.zeros((3,P))
    Z_temp = Z_0*np.ones((3,M))
    
    for p in range(P):
        h = 2**(p-(P+1))
        N = int(T/h)
        #Z_n = np.zeros((3,M,N))
        EZq = np.zeros((3,N))
        #zeta = tpdist(np.random.random((m,M,N)))
        for j in range(1,N):
            #Z_m = Z_n[:,:,j-1]
            #zeta_m = zeta[:,:,j-1]
            zeta = tpdist(np.random.random((m,M)))
            b = np.append(np.zeros((2,M)), np.dot(sigma,zeta), axis=0)
            g = fg(delta, r, Z_temp, Z_0)
            #K = fK(param, Z_m, Z_0, zeta_m, h, b)
            #K = fK(param, Z_temp, Z_0, zeta, h, b , g)
            K = g + (h**0.5) * b
            #eAhZ_m = np.matmul(eAh, Z_m)
            eAhZ_m = np.matmul(eAh, Z_temp)
            AK = np.matmul(A,K)
            g2 = fg(delta, r, eAhZ_m + K, Z_0)
            Z_temp = eAhZ_m + (h/2)*(g + g2 + AK) + (h**0.5)*b
            #Z_n[:,:,j] = eAhZ_m + (h/2)*( fg(delta, r, Z_m, Z_0) + \
            #             fg(delta, r, eAhZ_m + K, Z_0) + AK ) + (h**0.5)*b
            #EZq[:,j] = np.mean(Z_n[:,:,j]**2, axis=1)
            EZq[:,j] = np.mean(Z_temp**2, axis=1)
        u = np.mean(EZq, axis=1)
        EZq_erk2[:,p] = u
    
    EZq_true = np.zeros((3,1))
    
    d = 1-r+(beta**2)+beta*eta
    D = eta*(1-r+(beta**2)+beta*eta) + alpha*(beta+eta)
    
    EZq_true[0,0] = ((sigma**2)/(1-r))*d/D
    EZq_true[1,0] = ((sigma**2)/2)*1/D
    EZq_true[2,0] = ((sigma**2)/2)*d/D
    
    EZq_err = abs(EZq_erk2 - EZq_true)
    Dtvals = 2.**(np.arange(0,P,1)-6)
    
    # t = np.arange(0,N*h,h)
    # plt.axis([0, 1000, 0, 0.04])
    # plt.plot(t,EZq[2,:])
    
    plt.loglog(Dtvals, EZq_err[0,:], 'b*-')
    plt.loglog(Dtvals, Dtvals, 'r--')
    plt.axis([1e-2, 1, 1e-4, 1])

if __name__ == "__main__":
    main()