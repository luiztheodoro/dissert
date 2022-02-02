#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 20:25:28 2021

@author: luiztheodoro
"""

#import math
import numpy as np
import matplotlib.pyplot as plt

np.seterr('raise')

def ff(delta,r,X):
    ret = (1.0-r)*X + delta*(X**3)
    return ret


def fa(param,Z):
    beta, eta, alpha, delta, r, sigma = param
    ret = np.full_like(Z,0.0)
    ret[0,:] = Z[2,:]
    ret[1,:] = -beta*Z[1,:] + Z[2,:]
    ret[2,:] = -eta*Z[2,:] - ff(delta,r,Z[0,:]) - alpha*Z[1,:]
    return ret

def fK(param,Z,zeta,h):
    beta, eta, alpha, delta, r, sigma = param
    b =  np.append(np.zeros((2,M)), np.dot(sigma,zeta), axis=0) 
    ret = Z + fa(param,Z)*h + (h**0.5) * b
    return ret

np.random.seed(100)

m = 1

beta = 0.5
eta = 0.02
alpha = 0.5625
delta = 0.5
r = 0.5
sigma = 1.0e-5*np.ones((1,m))

param = [beta, eta, alpha, delta, r, sigma]

T = 974.3
h = 1.2
N = int(T/h)
M = 1000

tpdist = np.vectorize(lambda x: 0.0 if (x < 2/3) \
                                  else 3.0**0.5 if (x >= 2/3) and (x < 5/6) \
                                  else -(3.0**0.5))

zeta = tpdist(np.random.random((m,M,N)))

Z_0 = np.array([[0.0],[0.0],[0.0]])
Z_n = np.zeros((3,M,N))
EZq = np.zeros((3,N))
for j in range(1,N):
    Z_m = Z_n[:,:,j-1]
    zeta_m = zeta[:,:,j-1]
    b = np.append(np.zeros((2,M)), np.dot(sigma,zeta_m), axis=0) 
    K = fK(param, Z_m, zeta_m, h)
    Z_n[:,:,j] = Z_m + (h/2)*( fa(param, Z_m) + fa(param, K) ) + (h**0.5)*b
    EZq[:,j] = np.mean(Z_n[:,:,j]**2, axis=1)

t = np.arange(0,N*h,h)
plt.axis([0, 1000, -100, 500])
plt.plot(t,EZq[2,:])