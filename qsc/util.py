#!/usr/bin/env python3

"""
Various utility functions
"""

import numpy as np
import scipy.optimize
import logging
from qsc.fourier_interpolation import fourier_interpolation
from scipy.interpolate import CubicSpline as spline
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

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
        thetaN = theta-(self.iota-self.iotaN)*(phi+self.nu_spline(phi))
    else:
        thetaN = theta-(self.iota-self.iotaN)*phi
    if self.order == 'r1':
        return self.B0*(1+r*self.etabar*np.cos(thetaN))
    else:
        if Boozer_toroidal == False:
            self.B20_spline = self.convert_to_spline(self.B20)
        else:
            self.B20_spline=spline(np.append(self.varphi,2*np.pi/self.nfp), np.append(self.B20,self.B20[0]), bc_type='periodic')
        return self.B0*(1+r*self.etabar*np.cos(thetaN))+r**2*(self.B20_spline(phi)+self.B2c*np.cos(2*thetaN)+self.B2s*np.sin(2*thetaN))

def magB(self, radius, theta, phi):
    return self.B0*(1+radius*self.d*np.cos(theta-self.alpha))

def magB_fieldline(self, r, alpha, phi):
    return self.magB(r,alpha+self.iotaN*phi,phi)

def B_fieldline(self, r, alpha=0, phimax = None, nphi = 400):
    if phimax == None:
        phimax = 200*np.pi
    plt.figure(figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.xlabel(r'$\varphi$')
    plt.ylabel(r'$B(\varphi)$')
    plt.title("r = "+str(r)+", alpha = "+str(alpha))
    plt.plot(magB_fieldline(r,alpha,np.linspace(0,phimax,nphi)))
    plt.tight_layout()
    plt.show()
    plt.close()

def B_contour(self, r=0.1, ntheta=30, nphi=30, ncontours=10):
    theta_array=np.linspace(0,2*np.pi,ntheta)
    phi_array=np.linspace(0,2*np.pi,nphi)
    theta_2D, phi_2D = np.meshgrid(theta_array,phi_array)
    magB_2D = magB(r,phi_2D,theta_2D)
    magB_2D.shape = phi_2D.shape
    fig,ax=plt.subplots(1,1)
    contourplot = ax.contourf(phi_2D, theta_2D, magB_2D, ncontours)
    fig.colorbar(contourplot)
    ax.set_title('r='+str(r))
    ax.set_xlabel(r'$\varphi$')
    ax.set_ylabel(r'$\vartheta$')
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=1.0))
    plt.tight_layout()
    plt.show()
    plt.close()
