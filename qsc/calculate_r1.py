"""
This module contains the functions for solving the sigma equation
and computing diagnostics of the O(r^1) solution.
"""

import numpy as np
import logging
from .util import fourier_minimum
from .newton import newton
from .fourier_interpolation import fourier_interpolation

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _residual(self, x):
    """
    Residual in the sigma equation, used for Newton's method.  x is
    the state vector, corresponding to sigma on the phi grid,
    except that the first element of x is actually iota.
    """
    sigma = np.copy(x[1::])
    iota = x[0]
    if self.omn == True:
        location_section_I_II   = np.argmin(np.abs(self.varphi-self.delta))
        location_section_II_III = np.argmin(np.abs(self.varphi-(2*np.pi-self.delta)))
        varphiI   = self.varphi[0:location_section_I_II]
        varphiIII = self.varphi[location_section_II_III::]

        self.alpha_no_buffer = self.c0 + iota * (self.varphi - np.pi)
        self.alpha = self.c0 + iota * (self.varphi - np.pi)
        self.alpha[0:location_section_I_II]   = (1/self.delta**4)*( self.alpha0 * (self.delta-varphiI)**4 + varphiI * ( self.delta**3 * (4 * self.c0 + iota * (self.delta - 4 * np.pi)) + (self.c0 - np.pi * iota) * ( - 6 * varphiI * self.delta**2 + 4 * varphiI**2 * self.delta - varphiI**3) ))
        self.alpha[location_section_II_III::] = (1/self.delta**4)*( self.delta**4 * (self.c0 + iota * (-np.pi + varphiIII)) + (-self.c0 + 2 * self.m * np.pi + self.alpha0 - np.pi * iota) * ( varphiIII**4 - varphiIII**3 * 4 * (2*np.pi - self.delta) + varphiIII**2 * 6 * (2*np.pi - self.delta)**2 - varphiIII * 4 * (2*np.pi - self.delta)**3 + (2*np.pi - self.delta)**4) )

        gamma_iota = np.zeros((self.nphi,))
        gamma_0    = np.zeros((self.nphi,))
        gamma_iota[0:location_section_I_II]   = 4 * np.pi * (self.delta - varphiI)**3 / self.delta**4
        gamma_iota[location_section_II_III::] = 4 * np.pi * (self.delta + varphiIII - 2 * np.pi)**3 / self.delta**4
        gamma_0[0:location_section_I_II]   = -4 * (self.c0 - self.alpha0) * (self.delta - varphiI)**3 / self.delta**4
        gamma_0[location_section_II_III::] =  4 * (self.c0 - self.alpha0 - 2 * self.m * np.pi) * (self.delta + varphiIII - 2 * np.pi)**3 / self.delta**4
        self.gamma_iota = gamma_iota
        self.gamma_0    = gamma_0
        self.gamma      = gamma_iota * iota + gamma_0
    else:
        self.gamma = iota + self.helicity * self.nfp - np.matmul(self.d_d_varphi, self.alpha)
    r = np.matmul(self.d_d_varphi, sigma) \
        + self.gamma * \
        (self.etabar_squared_over_curvature_squared * self.etabar_squared_over_curvature_squared + 1 + sigma * sigma) \
        - 2 * self.etabar_squared_over_curvature_squared * (-self.spsi * self.torsion + self.I2 / self.Bbar) * self.G0 / self.B0
    #logger.debug("_residual called with x={}, r={}".format(x, r))
    sigma_spline = self.convert_to_spline(sigma)
    return np.append(r,sigma_spline(0)-self.sigma0)

def _jacobian(self, x):
    """
    Compute the Jacobian matrix for solving the sigma equation. x is
    the state vector, corresponding to sigma on the phi grid,
    except that the first element of x is actually iota.
    """
    sigma = np.copy(x[1::])

    # d (Riccati equation) / d sigma:
    jac = np.copy(self.d_d_varphi)
    for j in range(self.nphi):
        jac[j, j] += self.gamma [j] * 2 * sigma[j]

    # d (Riccati equation) / d iota:
    if self.omn == True:
        gamma_iota = self.gamma_iota
    else:
        gamma_iota = 1
    jac = np.append(np.transpose([gamma_iota * (self.etabar_squared_over_curvature_squared * self.etabar_squared_over_curvature_squared + 1 + sigma * sigma)]),jac,axis=1)

    # d (sigma[0]-sigma0) / dsigma:
    jac_last_row = np.zeros(self.nphi+1)
    factor = self.phi[2]/self.phi[1]
    jac_last_row[3]=1/(1-factor)
    jac_last_row[2]=-factor/(1-factor)
    jac = np.append(jac,[jac_last_row],axis=0)

    #logger.debug("_jacobian called with x={}, jac={}".format(x, jac))
    return jac

def solve_sigma_equation(self):
    """
    Solve the sigma equation.
    """
    x0 = np.full(self.nphi+1, self.sigma0)
    x0[0] = 0 # Initial guess for iota
    """
    soln = scipy.optimize.root(self._residual, x0, jac=self._jacobian, method='lm')
    self.iota = soln.x[0]
    self.sigma = np.copy(soln.x)
    self.sigma[0] = self.sigma0
    """
    self.sigma = newton(self._residual, x0, jac=self._jacobian)
    self.iota = self.sigma[0]
    self.iotaN = self.iota + self.helicity * self.nfp
    self.sigma = self.sigma[1::]

def _determine_helicity(self):
    """
    Determine the integer N associated with the type of quasisymmetry
    by counting the number of times the normal vector rotates
    poloidally as you follow the axis around toroidally.
    """
    quadrant = np.zeros(self.nphi + 1)
    for j in range(self.nphi):
        if self.normal_cylindrical[j,0] >= 0:
            if self.normal_cylindrical[j,2] >= 0:
                quadrant[j] = 1
            else:
                quadrant[j] = 4
        else:
            if self.normal_cylindrical[j,2] >= 0:
                quadrant[j] = 2
            else:
                quadrant[j] = 3
    quadrant[self.nphi] = quadrant[0]

    counter = 0
    for j in range(self.nphi):
        if quadrant[j] == 4 and quadrant[j+1] == 1:
            counter += 1
        elif quadrant[j] == 1 and quadrant[j+1] == 4:
            counter -= 1
        else:
            counter += quadrant[j+1] - quadrant[j]

    # It is necessary to flip the sign of axis_helicity in order
    # to maintain "iota_N = iota + axis_helicity" under the parity
    # transformations.
    counter *= self.spsi * self.sG
    self.helicity = counter / 4

def r1_diagnostics(self):
    """
    Compute various properties of the O(r^1) solution, once sigma and
    iota are solved for.
    """
    self.X1s = self.d_bar * np.sin(self.alpha)
    self.X1c = self.d_bar * np.cos(self.alpha)
    self.Y1s = self.sG * (self.Bbar / self.B0) * (1 / self.d_bar) * (self.sigma * np.sin(self.alpha) + np.cos(self.alpha))
    self.Y1c = self.sG * (self.Bbar / self.B0) * (1 / self.d_bar) * (self.sigma * np.cos(self.alpha) - np.sin(self.alpha))

    # Spline interpolant for the first order components of the magnetic field
    # as a function of phi, not varphi
    self.d_spline = self.convert_to_spline(self.d)
    self.alpha_spline = self.convert_to_spline(self.alpha)

    # If helicity is nonzero, then the original X1s/X1c/Y1s/Y1c variables are defined with respect to a "poloidal" angle that
    # is actually helical, with the theta=0 curve wrapping around the magnetic axis as you follow phi around toroidally. Therefore
    # here we convert to an untwisted poloidal angle, such that the theta=0 curve does not wrap around the axis.
    if self.helicity == 0:
        self.X1s_untwisted = 0
        self.X1c_untwisted = self.X1c
        self.Y1s_untwisted = self.Y1s
        self.Y1c_untwisted = self.Y1c
    else:
        angle = -self.helicity * self.nfp * self.varphi
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        self.X1s_untwisted = self.X1s *   cosangle  + self.X1c * sinangle
        self.X1c_untwisted = self.X1s * (-sinangle) + self.X1c * cosangle
        self.Y1s_untwisted = self.Y1s *   cosangle  + self.Y1c * sinangle
        self.Y1c_untwisted = self.Y1s * (-sinangle) + self.Y1c * cosangle

    # Use (R,Z) for elongation in the (R,Z) plane,
    # or use (X,Y) for elongation in the plane perpendicular to the magnetic axis.
    p = self.X1s * self.X1s + self.X1c * self.X1c + self.Y1s * self.Y1s + self.Y1c * self.Y1c
    q = self.X1s * self.Y1c - self.X1c * self.Y1s
    self.elongation = (p + np.sqrt(p * p - 4 * q * q)) / (2 * np.abs(q))
    self.mean_elongation = np.sum(self.elongation * self.d_l_d_phi) / np.sum(self.d_l_d_phi)
    index = np.argmax(self.elongation)
    self.max_elongation = -fourier_minimum(-self.elongation)

    self.d_X1c_d_varphi = np.matmul(self.d_d_varphi, self.X1c)
    self.d_Y1s_d_varphi = np.matmul(self.d_d_varphi, self.Y1s)
    self.d_Y1c_d_varphi = np.matmul(self.d_d_varphi, self.Y1c)

    self.calculate_grad_B_tensor()

