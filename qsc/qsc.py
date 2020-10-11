"""
This module contains the top-level routines for the quasisymmetric
stellarator construction.
"""

import numpy as np
from .spectral_diff_matrix import spectral_diff_matrix

class Qsc():
    def __init__(self, rc, zs, rs=[], zc=[], nfp=1, etabar=1, sigma0=0, B0=0,
                 I2=0, sG=1, spsi=1, nphi=15):
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

        self.nfp = nfp
        self.etabar = etabar
        self.sigma0 = sigma0
        self.B0 = B0
        self.I2 = I2
        self.sG = sG
        self.spsi = spsi
        self.nphi = nphi

        self.init_axis()
        
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

        self.B1Squared_over_curvatureSquared = self.etabar * self.etabar / (curvature * curvature)
        
        self.d_d_phi = spectral_diff_matrix(self.nphi, xmax=2 * np.pi / self.nfp)
        self.d_d_varphi = np.zeros((nphi, nphi))
        for j in range(3):
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
        self.axis_length = axis_length
        self.curvature = curvature
        self.torsion = torsion
        self.X1s = np.zeros(nphi)
        self.X1c = self.etabar / curvature
        
    def _residual(x):
        """
        Residual in the sigma equation, used for Newton's method.  x is
        the state vector, corresponding to sigma on the phi grid,
        except that the first element of x is actually iota.
        """
        sigma = np.copy(x)
        sigma[0] = self.sigma0
        iota = x[0]
        r = np.matmul(self.d_d_varphi, x) \
            + (iota + self.helicity * self.nfp) * \
            (self.B1Squared_over_curvatureSquared * self.B1Squared_over_curvatureSquared + 1 + sigma * sigma) \
            - 2 * B1Squared_over_curvatureSquared * (-self.spsi * self.torsion + self.I2 / self.B0) * self.G0 / self.B0
        return r
        
    def solve(newton_tol=1e-13, newton_maxit=10, linesearch_maxit=5):
        """
        Solve the sigma equation
        """
