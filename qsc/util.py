#!/usr/bin/env python3

"""
Various utility functions
"""

import logging
import numpy as np
import scipy.optimize
from qsc.fourier_interpolation import fourier_interpolation
from scipy.interpolate import CubicSpline as spline

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mu0 = 4 * np.pi * 1e-7

class Struct():
    """
    This class is just an empty mutable object to which we can attach
    attributes.
    """
    pass

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

    # Try to find a bracketing interval, using successively wider
    # intervals.
    f0 = func(index * dx)
    found_bracket = False
    for j in range(1, 4):
        bracket = np.array([index - j, index, index + j]) * dx
        fm = func(bracket[0])
        fp = func(bracket[2])
        if f0 < fm and f0 < fp:
            found_bracket = True
            break
    if not found_bracket:
        # We could throw an exception, though scipy will do that anyway
        pass

    logger.info('bracket={}, f(bracket)={}'.format(bracket, [func(bracket[0]), func(bracket[1]), func(bracket[2])]))
    #solution = scipy.optimize.minimize_scalar(func, bracket=bracket, options={"disp": True})
    solution = scipy.optimize.minimize_scalar(func, bracket=bracket)
    return solution.fun

def B_mag(self, r, theta, phi, Boozer_toroidal = False):
    '''
    Function to calculate the modulus of the magnetic field B for a given
    near-axis radius r, a Boozer poloidal angle theta (not vartheta) and
    a cylindrical toroidal angle phi if Boozer_toroidal = True or the
    Boozer angle varphi if Boozer_toroidal = True

    Args:
      r: the near-axis radius
      theta: the Boozer poloidal angle
      phi: the cylindrical or Boozer toroidal angle
      Boozer_toroidal: False if phi is the cylindrical toroidal angle, True for the Boozer one
    '''
    if Boozer_toroidal == False:
        thetaN = theta - (self.iota - self.iotaN) * (phi + self.nu_spline(phi))
    else:
        thetaN = theta - (self.iota - self.iotaN) * phi

    B = self.B0*(1 + r * self.etabar * np.cos(thetaN))

    # Add O(r^2) terms if necessary:
    if self.order != 'r1':
        if Boozer_toroidal == False:
            self.B20_spline = self.convert_to_spline(self.B20)
        else:
            self.B20_spline = spline(np.append(self.varphi, 2 * np.pi / self.nfp),
                                     np.append(self.B20, self.B20[0]),
                                     bc_type='periodic')

        B += (r**2) * (self.B20_spline(phi) + self.B2c * np.cos(2 * thetaN) + self.B2s * np.sin(2 * thetaN))

    return B
