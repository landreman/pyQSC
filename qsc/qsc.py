"""
This module contains the top-level routines for the quasisymmetric
stellarator construction.
"""

import numpy as np
import scipy.optimize
import logging
#from numba import jit
from .spectral_diff_matrix import spectral_diff_matrix
from .util import fourier_minimum, mu0
from .newton import newton
from .grad_B_tensor import grad_B_tensor, grad_grad_B_tensor

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Qsc():
    def __init__(self, rc, zs, rs=[], zc=[], nfp=1, etabar=1., sigma0=0., B0=1.,
                 I2=0., sG=1, spsi=1, nphi=31, B2s=0., B2c=0., p2=0., order="r1"):
        """
        Create a quasisymmetric stellarator.
        """
        # First, force {rc, zs, rs, zc} to have the same length, for
        # simplicity.
        nfourier = np.max([len(rc), len(zs), len(rs), len(zc)])
        self.nfourier = nfourier
        self.rc = np.zeros(nfourier)
        self.zs = np.zeros(nfourier)
        self.rs = np.zeros(nfourier)
        self.zc = np.zeros(nfourier)
        self.rc[:len(rc)] = rc
        self.zs[:len(zs)] = zs
        self.rs[:len(rs)] = rs
        self.zc[:len(zc)] = zc

        # Force nphi to be odd:
        if np.mod(nphi, 2) == 0:
            nphi += 1

        if sG != 1 and sG != -1:
            raise ValueError('sG must be +1 or -1')
        
        if spsi != 1 and spsi != -1:
            raise ValueError('spsi must be +1 or -1')

        self.nfp = nfp
        self.etabar = etabar
        self.sigma0 = sigma0
        self.B0 = B0
        self.I2 = I2
        self.sG = sG
        self.spsi = spsi
        self.nphi = nphi
        self.B2s = B2s
        self.B2c = B2c
        self.p2 = p2
        self.order = order
        self._set_names()

        self.calculate()
        
    def calculate(self):
        """
        Driver for the main calculations.
        """
        self.init_axis()
        self.solve_sigma_equation()
        self.r1_diagnostics()
        if self.order == 'r2':
            self.calculate_r2()

    def init_axis(self):
        """
        Initialize the curvature, torsion, differentiation matrix, etc.
        """
        # Shorthand:
        nphi = self.nphi
        nfp = self.nfp
        
        phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        d_phi = phi[1] - phi[0]
        R0 = np.zeros(nphi)
        Z0 = np.zeros(nphi)
        R0p = np.zeros(nphi)
        Z0p = np.zeros(nphi)
        R0pp = np.zeros(nphi)
        Z0pp = np.zeros(nphi)
        R0ppp = np.zeros(nphi)
        Z0ppp = np.zeros(nphi)
        for jn in range(0, self.nfourier):
            n = jn * nfp
            sinangle = np.sin(n * phi)
            cosangle = np.cos(n * phi)
            R0 += self.rc[jn] * cosangle + self.rs[jn] * sinangle
            Z0 += self.zc[jn] * cosangle + self.zs[jn] * sinangle
            R0p += self.rc[jn] * (-n * sinangle) + self.rs[jn] * (n * cosangle)
            Z0p += self.zc[jn] * (-n * sinangle) + self.zs[jn] * (n * cosangle)
            R0pp += self.rc[jn] * (-n * n * cosangle) + self.rs[jn] * (-n * n * sinangle)
            Z0pp += self.zc[jn] * (-n * n * cosangle) + self.zs[jn] * (-n * n * sinangle)
            R0ppp += self.rc[jn] * (n * n * n * sinangle) + self.rs[jn] * (-n * n * n * cosangle)
            Z0ppp += self.zc[jn] * (n * n * n * sinangle) + self.zs[jn] * (-n * n * n * cosangle)

        d_l_d_phi = np.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
        d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
        B0_over_abs_G0 = nphi / np.sum(d_l_d_phi)
        abs_G0_over_B0 = 1 / B0_over_abs_G0
        self.d_l_d_varphi = abs_G0_over_B0
        G0 = self.sG * abs_G0_over_B0 * self.B0

        # For these next arrays, the first dimension is phi, and the 2nd dimension is (R, phi, Z).
        d_r_d_phi_cylindrical = np.array([R0p, R0, Z0p]).transpose()
        d2_r_d_phi2_cylindrical = np.array([R0pp - R0, 2 * R0p, Z0pp]).transpose()
        d3_r_d_phi3_cylindrical = np.array([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp]).transpose()

        tangent_cylindrical = np.zeros((nphi, 3))
        d_tangent_d_l_cylindrical = np.zeros((nphi, 3))
        for j in range(3):
            tangent_cylindrical[:,j] = d_r_d_phi_cylindrical[:,j] / d_l_d_phi
            d_tangent_d_l_cylindrical[:,j] = (-d_r_d_phi_cylindrical[:,j] * d2_l_d_phi2 / d_l_d_phi \
                                              + d2_r_d_phi2_cylindrical[:,j]) / (d_l_d_phi * d_l_d_phi)

        curvature = np.sqrt(d_tangent_d_l_cylindrical[:,0] * d_tangent_d_l_cylindrical[:,0] + \
                            d_tangent_d_l_cylindrical[:,1] * d_tangent_d_l_cylindrical[:,1] + \
                            d_tangent_d_l_cylindrical[:,2] * d_tangent_d_l_cylindrical[:,2])

        axis_length = np.sum(d_l_d_phi) * d_phi * nfp
        rms_curvature = np.sqrt((np.sum(curvature * curvature * d_l_d_phi) * d_phi * nfp) / axis_length)
        mean_of_R = np.sum(R0 * d_l_d_phi) * d_phi * nfp / axis_length
        mean_of_Z = np.sum(Z0 * d_l_d_phi) * d_phi * nfp / axis_length
        standard_deviation_of_R = np.sqrt(np.sum((R0 - mean_of_R) ** 2 * d_l_d_phi) * d_phi * nfp / axis_length)
        standard_deviation_of_Z = np.sqrt(np.sum((Z0 - mean_of_Z) ** 2 * d_l_d_phi) * d_phi * nfp / axis_length)

        normal_cylindrical = np.zeros((nphi, 3))
        for j in range(3):
            normal_cylindrical[:,j] = d_tangent_d_l_cylindrical[:,j] / curvature
        self.normal_cylindrical = normal_cylindrical
        self._determine_helicity()

        # b = t x n
        binormal_cylindrical = np.zeros((nphi, 3))
        binormal_cylindrical[:,0] = tangent_cylindrical[:,1] * normal_cylindrical[:,2] - tangent_cylindrical[:,2] * normal_cylindrical[:,1]
        binormal_cylindrical[:,1] = tangent_cylindrical[:,2] * normal_cylindrical[:,0] - tangent_cylindrical[:,0] * normal_cylindrical[:,2]
        binormal_cylindrical[:,2] = tangent_cylindrical[:,0] * normal_cylindrical[:,1] - tangent_cylindrical[:,1] * normal_cylindrical[:,0]

        # We use the same sign convention for torsion as the
        # Landreman-Sengupta-Plunk paper, wikipedia, and
        # mathworld.wolfram.com/Torsion.html.  This sign convention is
        # opposite to Garren & Boozer's sign convention!
        torsion_numerator = (d_r_d_phi_cylindrical[:,0] * (d2_r_d_phi2_cylindrical[:,1] * d3_r_d_phi3_cylindrical[:,2] - d2_r_d_phi2_cylindrical[:,2] * d3_r_d_phi3_cylindrical[:,1]) \
                             + d_r_d_phi_cylindrical[:,1] * (d2_r_d_phi2_cylindrical[:,2] * d3_r_d_phi3_cylindrical[:,0] - d2_r_d_phi2_cylindrical[:,0] * d3_r_d_phi3_cylindrical[:,2]) 
                             + d_r_d_phi_cylindrical[:,2] * (d2_r_d_phi2_cylindrical[:,0] * d3_r_d_phi3_cylindrical[:,1] - d2_r_d_phi2_cylindrical[:,1] * d3_r_d_phi3_cylindrical[:,0]))

        torsion_denominator = (d_r_d_phi_cylindrical[:,1] * d2_r_d_phi2_cylindrical[:,2] - d_r_d_phi_cylindrical[:,2] * d2_r_d_phi2_cylindrical[:,1]) ** 2 \
            + (d_r_d_phi_cylindrical[:,2] * d2_r_d_phi2_cylindrical[:,0] - d_r_d_phi_cylindrical[:,0] * d2_r_d_phi2_cylindrical[:,2]) ** 2 \
            + (d_r_d_phi_cylindrical[:,0] * d2_r_d_phi2_cylindrical[:,1] - d_r_d_phi_cylindrical[:,1] * d2_r_d_phi2_cylindrical[:,0]) ** 2

        torsion = torsion_numerator / torsion_denominator

        self.etabar_squared_over_curvature_squared = self.etabar * self.etabar / (curvature * curvature)
        
        self.d_d_phi = spectral_diff_matrix(self.nphi, xmax=2 * np.pi / self.nfp)
        self.d_d_varphi = np.zeros((nphi, nphi))
        for j in range(nphi):
            self.d_d_varphi[j,:] = self.d_d_phi[j,:] / (B0_over_abs_G0 * d_l_d_phi[j])

        # Compute the Boozer toroidal angle:
        self.varphi = np.zeros(nphi)
        for j in range(1, nphi):
            # To get toroidal angle on the full mesh, we need d_l_d_phi on the half mesh.
            self.varphi[j] = self.varphi[j-1] + (d_l_d_phi[j-1] + d_l_d_phi[j])
        self.varphi = self.varphi * (0.5 * d_phi * 2 * np.pi / axis_length)
            
        # Add all results to self:
        self.phi = phi
        self.d_phi = d_phi
        self.R0 = R0
        self.Z0 = Z0
        self.G0 = G0
        self.d_l_d_phi = d_l_d_phi
        self.axis_length = axis_length
        self.curvature = curvature
        self.torsion = torsion
        self.X1s = np.zeros(nphi)
        self.X1c = self.etabar / curvature

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
        
    def _residual(self, x):
        """
        Residual in the sigma equation, used for Newton's method.  x is
        the state vector, corresponding to sigma on the phi grid,
        except that the first element of x is actually iota.
        """
        sigma = np.copy(x)
        sigma[0] = self.sigma0
        iota = x[0]
        r = np.matmul(self.d_d_varphi, sigma) \
            + (iota + self.helicity * self.nfp) * \
            (self.etabar_squared_over_curvature_squared * self.etabar_squared_over_curvature_squared + 1 + sigma * sigma) \
            - 2 * self.etabar_squared_over_curvature_squared * (-self.spsi * self.torsion + self.I2 / self.B0) * self.G0 / self.B0
        #logger.debug("_residual called with x={}, r={}".format(x, r))
        return r

    def _jacobian(self, x):
        """
        Compute the Jacobian matrix for solving the sigma equation. x is
        the state vector, corresponding to sigma on the phi grid,
        except that the first element of x is actually iota.
        """
        sigma = np.copy(x)
        sigma[0] = self.sigma0
        iota = x[0]

        # d (Riccati equation) / d sigma:
        # For convenience we will fill all the columns now, and re-write the first column in a moment.
        jac = np.copy(self.d_d_varphi)
        for j in range(self.nphi):
            jac[j, j] += (iota + self.helicity * self.nfp) * 2 * sigma[j]

        # d (Riccati equation) / d iota:
        jac[:, 0] = self.etabar_squared_over_curvature_squared * self.etabar_squared_over_curvature_squared + 1 + sigma * sigma

        #logger.debug("_jacobian called with x={}, jac={}".format(x, jac))
        return jac

    def solve_sigma_equation(self):
        """
        Solve the sigma equation.
        """
        x0 = np.full(self.nphi, self.sigma0)
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
        self.sigma[0] = self.sigma0
        
        
    def r1_diagnostics(self):
        """
        Compute various properties of the O(r^1) solution, once sigma and
        iota are solved for.
        """
        self.Y1s = self.sG * self.spsi * self.curvature / self.etabar
        self.Y1c = self.sG * self.spsi * self.curvature * self.sigma / self.etabar

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

        self.grad_B_tensor = grad_B_tensor(self)
        self.L_grad_B = self.grad_B_tensor.L_grad_B
        self.inv_L_grad_B = 1.0 / self.L_grad_B
        self.min_L_grad_B = fourier_minimum(self.L_grad_B)
        
    def get_dofs(self):
        """
        Return a 1D numpy vector of all possible optimizable
        degrees-of-freedom, for simsopt.
        """
        return np.concatenate((self.rc, self.zs, self.rs, self.zc,
                               np.array([self.etabar, self.sigma0, self.B2s, self.B2c, self.p2, self.I2, self.B0])))

    def set_dofs(self, x):
        """
        For interaction with simsopt, set the optimizable degrees of
        freedom from a 1D numpy vector.
        """
        assert len(x) == self.nfourier * 4 + 7
        self.rc = x[self.nfourier * 0 : self.nfourier * 1]
        self.zs = x[self.nfourier * 1 : self.nfourier * 2]
        self.rs = x[self.nfourier * 2 : self.nfourier * 3]
        self.zc = x[self.nfourier * 3 : self.nfourier * 4]
        self.etabar = x[self.nfourier * 4 + 0]
        self.sigma0 = x[self.nfourier * 4 + 1]
        self.B2s = x[self.nfourier * 4 + 2]
        self.B2c = x[self.nfourier * 4 + 3]
        self.p2 = x[self.nfourier * 4 + 4]
        self.I2 = x[self.nfourier * 4 + 5]
        self.B0 = x[self.nfourier * 4 + 6]
        self.calculate()
        logger.info('set_dofs called with x={}. Now iota={}, elongation={}'.format(x, self.iota, self.max_elongation))
        
    def _set_names(self):
        """
        For simsopt, sets the list of names for each degree of freedom.
        """
        names = []
        names += ['rc({})'.format(j) for j in range(self.nfourier)]
        names += ['zs({})'.format(j) for j in range(self.nfourier)]
        names += ['rs({})'.format(j) for j in range(self.nfourier)]
        names += ['zc({})'.format(j) for j in range(self.nfourier)]
        names += ['etabar', 'sigma0', 'B2s', 'B2c', 'p2', 'I2', 'B0']
        self.names = names

    @classmethod
    def from_paper(cls, name, **kwargs):
        """
        Get one of the configurations that has been used in our papers.
        """
        if name == "r1 section 5.1":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.1 """
            return cls(rc=[1, 0.045], zs=[0, -0.045], nfp=3, etabar=-0.9, **kwargs)
        elif name == "r1 section 5.2":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.2 """
            return cls(rc=[1, 0.265], zs=[0, -0.21], nfp=4, etabar=-2.25, **kwargs)
        elif name == "r1 section 5.3":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.3 """
            return cls(rc=[1, 0.042], zs=[0, -0.042], zc=[0, -0.025], nfp=3, etabar=-1.1, sigma0=-0.6, **kwargs)
        elif name == "r2 section 5.1" or name == '5.1' or name == 1:
            """ The configuration from Landreman & Sengupta (2019), section 5.1 """
            return cls(rc=[1, 0.155, 0.0102], zs=[0, 0.154, 0.0111], nfp=2, etabar=0.64, order='r2', B2c=-0.00322, **kwargs)
        elif name == "r2 section 5.2" or name == '5.2' or name == 2:
            """ The configuration from Landreman & Sengupta (2019), section 5.2 """
            return cls(rc=[1, 0.173, 0.0168, 0.00101], zs=[0, 0.159, 0.0165, 0.000985], nfp=2, etabar=0.632, order='r2', B2c=-0.158, **kwargs)
        elif name == "r2 section 5.3" or name == '5.3' or name == 3:
            """ The configuration from Landreman & Sengupta (2019), section 5.3 """
            return cls(rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, order='r2', B2c=-0.7, p2=-600000., **kwargs)
        elif name == "r2 section 5.4" or name == '5.4' or name == 4:
            """ The configuration from Landreman & Sengupta (2019), section 5.4 """
            return cls(rc=[1, 0.17, 0.01804, 0.001409, 5.877e-05],
                       zs=[0, 0.1581, 0.01820, 0.001548, 7.772e-05], nfp=4, etabar=1.569, order='r2', B2c=0.1348, **kwargs)
        elif name == "r2 section 5.5" or name == '5.5' or name == 5:
            """ The configuration from Landreman & Sengupta (2019), section 5.5 """
            return cls(rc=[1, 0.3], zs=[0, 0.3], nfp=5, etabar=2.5, sigma0=0.3, I2=1.6, order='r2', B2c=1., B2s=3., p2=-0.5e7, **kwargs)
        else:
            raise ValueError('Unrecognized configuration name')

    def calculate_r2(self):
        """
        Compute the O(r^2) quantities.
        """
        logger.debug('Calculating O(r^2) terms')
        # First, some shorthand:
        nphi = self.nphi
        B0_over_abs_G0 = self.B0 / np.abs(self.G0)
        abs_G0_over_B0 = 1 / B0_over_abs_G0
        X1c = self.X1c
        Y1s = self.Y1s
        Y1c = self.Y1c
        sigma = self.sigma
        d_d_varphi = self.d_d_varphi
        iota_N = self.iotaN
        iota = self.iota
        curvature = self.curvature
        torsion = self.torsion
        etabar = self.etabar
        B0 = self.B0
        G0 = self.G0
        I2 = self.I2
        B2s = self.B2s
        B2c = self.B2c
        p2 = self.p2
        sG = self.sG
        spsi = self.spsi
        I2_over_B0 = self.I2 / self.B0
        
        if np.abs(iota_N) < 1e-8:
            print('Warning: |iota_N| is very small so O(r^2) solve will be poorly conditioned. iota_N=', iota_N)
        
        V1 = X1c * X1c + Y1c * Y1c + Y1s * Y1s
        V2 = 2 * Y1s * Y1c
        V3 = X1c * X1c + Y1c * Y1c - Y1s * Y1s
        
        factor = - B0_over_abs_G0 / 8;
        Z20 = factor*np.matmul(d_d_varphi,V1)
        Z2s = factor*(np.matmul(d_d_varphi,V2) - 2 * iota_N * V3)
        Z2c = factor*(np.matmul(d_d_varphi,V3) + 2 * iota_N * V2)

        qs = -iota_N * X1c - Y1s * torsion * abs_G0_over_B0
        qc = np.matmul(d_d_varphi,X1c) - Y1c * torsion * abs_G0_over_B0
        rs = np.matmul(d_d_varphi,Y1s) - iota_N * Y1c
        rc = np.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * torsion * abs_G0_over_B0

        X2s = B0_over_abs_G0 * (np.matmul(d_d_varphi,Z2s) - 2*iota_N*Z2c + B0_over_abs_G0 * ( abs_G0_over_B0*abs_G0_over_B0*B2s/B0 + (qc * qs + rc * rs)/2)) / curvature

        X2c = B0_over_abs_G0 * (np.matmul(d_d_varphi,Z2c) + 2*iota_N*Z2s - B0_over_abs_G0 * (-abs_G0_over_B0*abs_G0_over_B0*B2c/B0 \
               + abs_G0_over_B0*abs_G0_over_B0*etabar*etabar/2 - (qc * qc - qs * qs + rc * rc - rs * rs)/4)) / curvature

        beta_1s = -4 * spsi * sG * mu0 * p2 * etabar * abs_G0_over_B0 / (iota_N * B0 * B0)

        Y2s_from_X20 = -sG * spsi * curvature * curvature / (etabar * etabar)
        Y2s_inhomogeneous = sG * spsi * (-curvature/2 + curvature*curvature/(etabar*etabar)*(-X2c + X2s * sigma))

        Y2c_from_X20 = -sG * spsi * curvature * curvature * sigma / (etabar * etabar)
        Y2c_inhomogeneous = sG * spsi * curvature * curvature / (etabar * etabar) * (X2s + X2c * sigma)

        # Note: in the fX* and fY* quantities below, I've omitted the
        # contributions from X20 and Y20 to the d/dzeta terms. These
        # contributions are handled later when we assemble the large
        # matrix.

        fX0_from_X20 = -4 * sG * spsi * abs_G0_over_B0 * (Y2c_from_X20 * Z2s - Y2s_from_X20 * Z2c)
        fX0_from_Y20 = -torsion * abs_G0_over_B0 - 4 * sG * spsi * abs_G0_over_B0 * (Z2s) \
            - spsi * I2_over_B0 * (-2) * abs_G0_over_B0
        fX0_inhomogeneous = curvature * abs_G0_over_B0 * Z20 - 4 * sG * spsi * abs_G0_over_B0 * (Y2c_inhomogeneous * Z2s - Y2s_inhomogeneous * Z2c) \
            - spsi * I2_over_B0 * (0.5 * curvature * sG * spsi) * abs_G0_over_B0 + beta_1s * abs_G0_over_B0 / 2 * Y1c

        fXs_from_X20 = -torsion * abs_G0_over_B0 * Y2s_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (Y2c_from_X20 * Z20) \
            - spsi * I2_over_B0 * (- 2 * Y2s_from_X20) * abs_G0_over_B0
        fXs_from_Y20 = - 4 * spsi * sG * abs_G0_over_B0 * (-Z2c + Z20)
        fXs_inhomogeneous = np.matmul(d_d_varphi,X2s) - 2 * iota_N * X2c - torsion * abs_G0_over_B0 * Y2s_inhomogeneous + curvature * abs_G0_over_B0 * Z2s \
            - 4 * spsi * sG * abs_G0_over_B0 * (Y2c_inhomogeneous * Z20) \
            - spsi * I2_over_B0 * (0.5 * curvature * spsi * sG - 2 * Y2s_inhomogeneous) * abs_G0_over_B0 \
            - (0.5) * abs_G0_over_B0 * beta_1s * Y1s

        fXc_from_X20 = - torsion * abs_G0_over_B0 * Y2c_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (-Y2s_from_X20 * Z20) \
            - spsi * I2_over_B0 * (- 2 * Y2c_from_X20) * abs_G0_over_B0
        fXc_from_Y20 = - torsion * abs_G0_over_B0 - 4 * spsi * sG * abs_G0_over_B0 * (Z2s) \
            - spsi * I2_over_B0 * (-2) * abs_G0_over_B0
        fXc_inhomogeneous = np.matmul(d_d_varphi,X2c) + 2 * iota_N * X2s - torsion * abs_G0_over_B0 * Y2c_inhomogeneous + curvature * abs_G0_over_B0 * Z2c \
            - 4 * spsi * sG * abs_G0_over_B0 * (-Y2s_inhomogeneous * Z20) \
            - spsi * I2_over_B0 * (0.5 * curvature * sG * spsi - 2 * Y2c_inhomogeneous) * abs_G0_over_B0 \
            - (0.5) * abs_G0_over_B0 * beta_1s * Y1c

        fY0_from_X20 = torsion * abs_G0_over_B0 - spsi * I2_over_B0 * (2) * abs_G0_over_B0
        fY0_from_Y20 = np.zeros(nphi)
        fY0_inhomogeneous = -4 * spsi * sG * abs_G0_over_B0 * (X2s * Z2c - X2c * Z2s) \
            - spsi * I2_over_B0 * (-0.5 * curvature * X1c * X1c) * abs_G0_over_B0 - (0.5) * abs_G0_over_B0 * beta_1s * X1c

        fYs_from_X20 = -2 * iota_N * Y2c_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (Z2c)
        fYs_from_Y20 = np.full(nphi, -2 * iota_N)
        fYs_inhomogeneous = np.matmul(d_d_varphi,Y2s_inhomogeneous) - 2 * iota_N * Y2c_inhomogeneous + torsion * abs_G0_over_B0 * X2s \
            - 4 * spsi * sG * abs_G0_over_B0 * (-X2c * Z20) - 2 * spsi * I2_over_B0 * X2s * abs_G0_over_B0

        fYc_from_X20 = 2 * iota_N * Y2s_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (-Z2s)
        fYc_from_Y20 = np.zeros(nphi)
        fYc_inhomogeneous = np.matmul(d_d_varphi,Y2c_inhomogeneous) + 2 * iota_N * Y2s_inhomogeneous + torsion * abs_G0_over_B0 * X2c \
            - 4 * spsi * sG * abs_G0_over_B0 * (X2s * Z20) \
            - spsi * I2_over_B0 * (-0.5 * curvature * X1c * X1c + 2 * X2c) * abs_G0_over_B0 + 0.5 * abs_G0_over_B0 * beta_1s * X1c

        matrix = np.zeros((2 * nphi, 2 * nphi))
        right_hand_side = np.zeros(2 * nphi)
        for j in range(nphi):
            # Handle the terms involving d X_0 / d zeta and d Y_0 / d zeta:
            # ----------------------------------------------------------------

            # Equation 1, terms involving X0:
            # Contributions arise from Y1c * fYs - Y1s * fYc.
            matrix[j, 0:nphi] = Y1c[j] * d_d_varphi[j, :] * Y2s_from_X20 - Y1s[j] * d_d_varphi[j, :] * Y2c_from_X20

            # Equation 1, terms involving Y0:
            # Contributions arise from -Y1s * fY0 - Y1s * fYc, and they happen to be equal.
            matrix[j, nphi:(2*nphi)] = -2 * Y1s[j] * d_d_varphi[j, :]

            # Equation 2, terms involving X0:
            # Contributions arise from -X1c * fX0 + Y1s * fYs + Y1c * fYc
            matrix[j+nphi, 0:nphi] = -X1c[j] * d_d_varphi[j, :] + Y1s[j] * d_d_varphi[j, :] * Y2s_from_X20 + Y1c[j] * d_d_varphi[j, :] * Y2c_from_X20

            # Equation 2, terms involving Y0:
            # Contributions arise from -Y1c * fY0 + Y1c * fYc, but they happen to cancel.

            # Now handle the terms involving X_0 and Y_0 without d/dzeta derivatives:
            # ----------------------------------------------------------------

            matrix[j, j       ] = matrix[j, j       ] + X1c[j] * fXs_from_X20[j] - Y1s[j] * fY0_from_X20[j] + Y1c[j] * fYs_from_X20[j] - Y1s[j] * fYc_from_X20[j]
            matrix[j, j + nphi] = matrix[j, j + nphi] + X1c[j] * fXs_from_Y20[j] - Y1s[j] * fY0_from_Y20[j] + Y1c[j] * fYs_from_Y20[j] - Y1s[j] * fYc_from_Y20[j]

            matrix[j + nphi, j       ] = matrix[j + nphi, j       ] - X1c[j] * fX0_from_X20[j] + X1c[j] * fXc_from_X20[j] - Y1c[j] * fY0_from_X20[j] + Y1s[j] * fYs_from_X20[j] + Y1c[j] * fYc_from_X20[j]
            matrix[j + nphi, j + nphi] = matrix[j + nphi, j + nphi] - X1c[j] * fX0_from_Y20[j] + X1c[j] * fXc_from_Y20[j] - Y1c[j] * fY0_from_Y20[j] + Y1s[j] * fYs_from_Y20[j] + Y1c[j] * fYc_from_Y20[j]
          

        right_hand_side[0:nphi] = -(X1c * fXs_inhomogeneous - Y1s * fY0_inhomogeneous + Y1c * fYs_inhomogeneous - Y1s * fYc_inhomogeneous)
        right_hand_side[nphi:2 * nphi] = -(- X1c * fX0_inhomogeneous + X1c * fXc_inhomogeneous - Y1c * fY0_inhomogeneous + Y1s * fYs_inhomogeneous + Y1c * fYc_inhomogeneous)

        solution = np.linalg.solve(matrix, right_hand_side)
        X20 = solution[0:nphi]
        Y20 = solution[nphi:2 * nphi]

        # Now that we have X20 and Y20 explicitly, we can reconstruct Y2s, Y2c, and B20:
        Y2s = Y2s_inhomogeneous + Y2s_from_X20 * X20
        Y2c = Y2c_inhomogeneous + Y2c_from_X20 * X20 + Y20

        B20 = B0 * (curvature * X20 - B0_over_abs_G0 * np.matmul(d_d_varphi,Z20) + (0.5) * etabar * etabar - mu0 * p2 / (B0 * B0) \
                    - 0.25 * B0_over_abs_G0 * B0_over_abs_G0 * (qc * qc + qs * qs + rc * rc + rs * rs))

        d_l_d_phi = self.d_l_d_phi
        normalizer = 1 / np.sum(d_l_d_phi)
        self.B20_mean = np.sum(B20 * d_l_d_phi) * normalizer
        self.B20_anomaly = B20 - self.B20_mean
        self.B20_residual = np.sqrt(np.sum((B20 - self.B20_mean) * (B20 - self.B20_mean) * d_l_d_phi) * normalizer) / B0
        self.B20_variation = np.max(B20) - np.min(B20)

        self.N_helicity = - self.helicity * self.nfp
        self.Bbar = spsi * B0
        self.G2 = -mu0 * p2 * G0 / (B0 * B0) - iota * I2

        self.d_curvature_d_varphi = np.matmul(d_d_varphi, curvature)
        self.d_torsion_d_varphi = np.matmul(d_d_varphi, torsion)
        self.d_X20_d_varphi = np.matmul(d_d_varphi, X20)
        self.d_X2s_d_varphi = np.matmul(d_d_varphi, X2s)
        self.d_X2c_d_varphi = np.matmul(d_d_varphi, X2c)
        self.d_Y20_d_varphi = np.matmul(d_d_varphi, Y20)
        self.d_Y2s_d_varphi = np.matmul(d_d_varphi, Y2s)
        self.d_Y2c_d_varphi = np.matmul(d_d_varphi, Y2c)
        self.d_Z20_d_varphi = np.matmul(d_d_varphi, Z20)
        self.d_Z2s_d_varphi = np.matmul(d_d_varphi, Z2s)
        self.d_Z2c_d_varphi = np.matmul(d_d_varphi, Z2c)
        self.d2_X1c_d_varphi2 = np.matmul(d_d_varphi, self.d_X1c_d_varphi)
        self.d2_Y1c_d_varphi2 = np.matmul(d_d_varphi, self.d_Y1c_d_varphi)
        self.d2_Y1s_d_varphi2 = np.matmul(d_d_varphi, self.d_Y1s_d_varphi)

        self.B0_order_a_squared_to_cancel = -sG * B0 * B0 * (self.G2 + I2 * self.N_helicity) * abs_G0_over_B0 / (2*G0*G0) \
            -sG * spsi * B0 * 2 * (X2c * Y2s - X2s * Y2c) \
            -sG * B0 * B0 / (2*G0) * (abs_G0_over_B0 * X20 * curvature - self.d_Z20_d_varphi) \
            -sG * spsi * B0 * I2 / (4*G0) * (-abs_G0_over_B0 * torsion * (X1c*X1c + Y1c*Y1c + Y1s*Y1s) + Y1c * self.d_X1c_d_varphi - X1c * self.d_Y1c_d_varphi)
        
        # Store all important results in self:
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        
        self.X20 = X20
        self.X2s = X2s
        self.X2c = X2c
        self.Y20 = Y20
        self.Y2s = Y2s
        self.Y2c = Y2c
        self.Z20 = Z20
        self.Z2s = Z2s
        self.Z2c = Z2c
        self.beta_1s = beta_1s
        self.B20 = B20

        # O(r^2) diagnostics:
        self.mercier()
        t = grad_grad_B_tensor(self)
        self.grad_grad_B_inverse_scale_length_vs_varphi = t.grad_grad_B_inverse_scale_length_vs_varphi
        self.grad_grad_B_inverse_scale_length = t.grad_grad_B_inverse_scale_length

    def mercier(self):
        """
        Compute the terms in Mercier's criterion.
        """

        # See Overleaf note "Mercier criterion near the magnetic axis- detailed notes".
        # See also "20200604-02 Checking sign in Mercier DGeod near axis.docx"

        # Shorthand:
        d_l_d_phi = self.d_l_d_phi
        B0 = self.B0
        G0 = self.G0
        p2 = self.p2
        etabar = self.etabar
        curvature = self.curvature
        sigma = self.sigma
        iotaN = self.iotaN
        iota = self.iota
        pi = np.pi
        
        #integrand = d_l_d_phi * (Y1c * Y1c + X1c * (X1c + Y1s)) / (Y1c * Y1c + (X1c + Y1s) * (X1c + Y1s))
        integrand = d_l_d_phi * (etabar*etabar*etabar*etabar + curvature*curvature*curvature*curvature*sigma*sigma + etabar*etabar*curvature*curvature) \
            / (etabar*etabar*etabar*etabar + curvature*curvature*curvature*curvature*(1+sigma*sigma) + 2*etabar*etabar*curvature*curvature)

        integral = np.sum(integrand) * self.d_phi * self.nfp * 2 * pi / self.axis_length

        #DGeod_times_r2 = -(2 * sG * spsi * mu0 * mu0 * p2 * p2 * G0 * G0 * G0 * G0 * etabar * etabar &
        self.DGeod_times_r2 = -(2 * mu0 * mu0 * p2 * p2 * G0 * G0 * G0 * G0 * etabar * etabar \
                           / (pi * pi * pi * B0 * B0 * B0 * B0 * B0 * B0 * B0 * B0 * B0 * B0 * iotaN * iotaN)) \
                           * integral

        self.d2_volume_d_psi2 = 4*pi*pi*abs(G0)/(B0*B0*B0)*(3*etabar*etabar - 4*self.B20_mean/B0 + 2 * (self.G2 + iota * self.I2)/G0)

        self.DWell_times_r2 = (mu0 * p2 * abs(G0) / (8 * pi * pi * pi * pi * B0 * B0 * B0)) * \
            (self.d2_volume_d_psi2 - 8 * pi * pi * mu0 * p2 * abs(G0) / (B0 * B0 * B0 * B0 * B0))

        self.DMerc_times_r2 = self.DWell_times_r2 + self.DGeod_times_r2
