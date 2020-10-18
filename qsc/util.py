#!/usr/bin/env python3

"""
Various utility functions
"""

import numpy as np
import scipy.optimize
import logging
from qsc.fourier_interpolation import fourier_interpolation

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mu0 = 4 * np.pi * 1e-7

def fourier_minimum(y):
    """
    Given uniformly spaced data y on a periodic domain, find the
    minimum of the spectral interpolant.
    """
    # Handle the case of a constant:
    if (np.max(y) - np.min(y)) / np.max([1e-14, np.abs(np.mean(y))]) < 1e-14:
        return y[0]
    
    n = len(y)
    dx = 2 * np.pi / n
    # Compute a rough guess for the minimum, given by the minimum of
    # the discrete data:
    index = np.argmin(y)

    def func(x):
        interp = fourier_interpolation(y, np.array([x]))
        logger.debug('fourier_minimum.func called at x={}, y={}'.format(x, interp[0]))
        return interp[0]

    # In case two adjacent points in the y grid are extremely close,
    # use an initial bracket that is wider than 2 grid points, if
    # there are enough grid points.
    if n > 5:
        bracket = np.array([index - 2, index, index + 2]) * dx
    else:
        bracket = np.array([index - 1, index, index + 1]) * dx

    logger.info('bracket={}, f(bracket)={}'.format(bracket, [func(bracket[0]), func(bracket[1]), func(bracket[2])]))
    #solution = scipy.optimize.minimize_scalar(func, bracket=bracket, options={"disp": True})
    solution = scipy.optimize.minimize_scalar(func, bracket=bracket)
    return solution.fun
