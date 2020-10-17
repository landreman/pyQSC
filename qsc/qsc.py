"""
This module contains the top-level routines for the quasisymmetric
stellarator construction.
"""

import numpy as np
import scipy.optimize
import logging
#from numba import jit
from .spectral_diff_matrix import spectral_diff_matrix
from .util import fourier_minimum
from .newton import newton
from .grad_B_tensor import grad_B_tensor

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Qsc():
    def __init__(self, rc, zs, rs=[], zc=[], nfp=1, etabar=1., sigma0=0., B0=1.,
                 I2=0., sG=1, spsi=1, nphi=31):
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
        self._set_names()

        self.calculate()
        
    def calculate(self):
        """
        Driver for the main calculations.
        """
        self.init_axis()
        self.solve_sigma_equation()
        self.r1_diagnostics()

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
        self.min_L_grad_B = fourier_minimum(self.L_grad_B)
        
    def get_dofs(self):
        """
        Return a 1D numpy vector of all possible optimizable
        degrees-of-freedom, for simsopt.
        """
        return np.concatenate((self.rc, self.zs, self.rs, self.zc,
                               np.array([self.etabar, self.sigma0])))

    def set_dofs(self, x):
        """
        For interaction with simsopt, set the optimizable degrees of
        freedom from a 1D numpy vector.
        """
        assert len(x) == self.nfourier * 4 + 2
        self.rc = x[self.nfourier * 0 : self.nfourier * 1]
        self.zs = x[self.nfourier * 1 : self.nfourier * 2]
        self.rs = x[self.nfourier * 2 : self.nfourier * 3]
        self.zc = x[self.nfourier * 3 : self.nfourier * 4]
        self.etabar = x[self.nfourier * 4 + 0]
        self.sigma0 = x[self.nfourier * 4 + 1]
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
        names += ['etabar', 'sigma0']
        self.names = names

    @classmethod
    def r1_section51(cls, **kwargs):
        """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.1 """
        return cls(rc=[1, 0.045], zs=[0, -0.045], nfp=3, etabar=-0.9, **kwargs)

    @classmethod
    def r1_section52(cls, **kwargs):
        """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.2 """
        return cls(rc=[1, 0.265], zs=[0, -0.21], nfp=4, etabar=-2.25, **kwargs)

    @classmethod
    def r1_section53(cls, **kwargs):
        """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.3 """
        return cls(rc=[1, 0.042], zs=[0, -0.042], zc=[0, -0.025], nfp=3, etabar=-1.1, sigma0=-0.6, **kwargs)

    @classmethod
    def r2_section51(cls, **kwargs):
        """ The configuration from Landreman & Sengupta (2019), section 5.1 """
        return cls(rc=[1, 0.155, 0.0102], zs=[0, 0.154, 0.0111], nfp=2, etabar=0.64, **kwargs)

    @classmethod
    def r2_section52(cls, **kwargs):
        """ The configuration from Landreman & Sengupta (2019), section 5.2 """
        return cls(rc=[1, 0.173, 0.0168, 0.00101], zs=[0, 0.159, 0.0165, 0.000985], nfp=2, etabar=0.632, **kwargs)

    @classmethod
    def r2_section53(cls, **kwargs):
        """ The configuration from Landreman & Sengupta (2019), section 5.3 """
        return cls(rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, **kwargs)

    @classmethod
    def r2_section54(cls, **kwargs):
        """ The configuration from Landreman & Sengupta (2019), section 5.4 """
        return cls(rc=[1, 0.17, 0.01804, 0.001409, 5.877e-05],
                       zs=[0, 0.1581, 0.01820, 0.001548, 7.772e-05], nfp=4, etabar=1.569, **kwargs)

    @classmethod
    def r2_section55(cls, **kwargs):
        """ The configuration from Landreman & Sengupta (2019), section 5.5 """
        return cls(rc=[1, 0.3], zs=[0, 0.3], nfp=5, etabar=2.5, sigma0=0.3, I2=1.6, **kwargs)

