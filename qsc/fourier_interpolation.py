#!/usr/bin/env python3

"""
This module contains a subroutine for spectrally accurate interpolation of
data that is known on a uniform grid in a periodic domain.
"""

import numpy as np
#from numba import njit

# Get machine precision
eps = np.finfo(float).eps
    
#@njit
def fourier_interpolation(fk, x):
    """
    Interpolate data that is known on a uniform grid in [0, 2pi).

    This routine is based on the
    matlab routine fourint.m in the DMSuite package by S.C. Reddy and J.A.C. Weideman, available at
    http://www.mathworks.com/matlabcentral/fileexchange/29
    or here:
    http://dip.sun.ac.za/~weideman/research/differ.html  

    fk:  Vector of y-coordinates of data, at equidistant points
         x(k) = (k-1)*2*pi/N,  k = 1...N
    x:   Vector of x-values where interpolant is to be evaluated.
    output: Vector of interpolated values.
    """

    N = len(fk)
    M = len(x)

    # Compute equidistant points
    #xk = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    xk = (np.arange(N) * 2 * np.pi) / N

    # Weights for trig interpolation
    w = (-1.0) ** np.arange(0, N)
    #w = np.array((-1) ** np.arange(0, N), dtype='f')

    """
    x2 = x / 2
    xk2 = xk / 2

    # Compute quantities x - x(k)
    xk2_2D, x2_2D = np.meshgrid(xk2, x2)
    Dold = x2_2D - xk2_2D
    D = 0.5 * (np.outer(x, np.ones(N)) - np.outer(np.ones(M), xk))
    print(Dold - D)
    """
    D = 0.5 * (np.outer(x, np.ones(N)) - np.outer(np.ones(M), xk))
    
    if np.mod(N, 2) == 0:
        # Formula for N even
        D = 1 / np.tan(D + eps * (D==0))
    else:
        # Formula for N odd
        D = 1 / np.sin(D + eps * (D==0))

    # Evaluate interpolant as matrix-vector products
    #return np.matmul(D, w * fk) / np.matmul(D, w)
    return np.dot(D, w * fk) / np.dot(D, w)
    #return (D @ w * fk) / (D @ w)
    #return D.dot(w * fk) / D.dot(w)
