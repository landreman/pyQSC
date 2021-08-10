"""
This module contains the routine to initialize quantities like
curvature and torsion from the magnetix axis shape.
"""

import numpy as np
import logging
from .spectral_diff_matrix import spectral_diff_matrix
from .util import fourier_minimum
from scipy.interpolate import CubicSpline as spline
from scipy.interpolate import splev, splrep

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define periodic spline interpolant conversion used in several scripts and plotting
def convert_to_spline(self,array):
    # sp=spline(np.append(self.phi,self.phi[0]+2*np.pi/self.nfp), np.append(array,array[0]), bc_type='periodic')
    # return sp
    spl=splrep(np.append(self.phi,self.phi[0]+2*np.pi/self.nfp), np.append(array,array[0]), per=1, k=5)
    return lambda x: splev(x, spl)

def init_axis(self):
    """
    Initialize the curvature, torsion, differentiation matrix, etc.
    """
    # Shorthand:
    nphi = self.nphi
    nfp = self.nfp
    phi = self.phi
    d_phi = self.d_phi
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
    d3_l_d_phi3 = (R0p * R0p + R0pp * R0pp + Z0pp * Z0pp + R0 * R0pp + R0p * R0ppp + Z0p * Z0ppp - d2_l_d_phi2 * d2_l_d_phi2) / d_l_d_phi
    G0 = self.sG * np.sum(self.B0 * d_l_d_phi) / nphi
    self.d_l_d_varphi = self.sG * G0 / self.B0   


    # For these next arrays, the first dimension is phi, and the 2nd dimension is (R, phi, Z).
    d_r_d_phi_cylindrical = np.array([R0p, R0, Z0p]).transpose()
    d2_r_d_phi2_cylindrical = np.array([R0pp - R0, 2 * R0p, Z0pp]).transpose()
    d3_r_d_phi3_cylindrical = np.array([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp]).transpose()

    tangent_cylindrical = np.zeros((nphi, 3))
    d_tangent_d_l_cylindrical = np.zeros((nphi, 3))
    d2_tangent_d_l2_cylindrical = np.zeros((nphi, 3))
    for j in range(3):
        tangent_cylindrical[:,j] = d_r_d_phi_cylindrical[:,j] / d_l_d_phi
        d_tangent_d_l_cylindrical[:,j] = (-d_r_d_phi_cylindrical[:,j] * d2_l_d_phi2 / d_l_d_phi \
                                          + d2_r_d_phi2_cylindrical[:,j]) / (d_l_d_phi * d_l_d_phi)
        d2_tangent_d_l2_cylindrical[:,j] = (d3_r_d_phi3_cylindrical[:,j]\
                                          -d2_r_d_phi2_cylindrical[:,j] * 3 * d2_l_d_phi2 / d_l_d_phi\
                                          -d_r_d_phi_cylindrical[:,j] / d_l_d_phi * (d3_l_d_phi3 - 3 * d2_l_d_phi2 * d2_l_d_phi2 / d_l_d_phi)\
                                          ) / (d_l_d_phi * d_l_d_phi * d_l_d_phi)

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

    # If looking for omnigenity, use signed Frenet-Serret frame
    # new_normal = np.array([d2_tangent_d_l2_cylindrical[j,:]-np.dot(d2_tangent_d_l2_cylindrical[j,:],tangent_cylindrical[j,:])*tangent_cylindrical[j,:] for j in range(self.nphi)])
    # new_normal = np.array([new_normal[j,:]/np.sqrt(np.dot(new_normal[j,:],new_normal[j,:])) for j in range(self.nphi)])
    # self.new_normal = new_normal
    # normal_cylindrical = new_normal
    sign_curvature_change = np.ones((self.nphi,))
    if self.nfp == 1:
        if self.nfourier == 2:
            if (self.rc[0]+2*self.rc[1]) == 0 or (self.rc[0]+self.rc[1])**2+self.zs[1]**2==0:
                sign_curvature_change[int(self.nphi/2)::] = [-sign_curvature_change[i] for i in range(int(self.nphi/2),self.nphi)]
        if self.nfourier == 3:
            if (self.rc[0]+2*self.rc[1]+5*self.rc[2]) == 0 or (self.rc[0]+self.rc[1]+self.rc[2])**2+(self.zs[1]+2*self.zs[2])**2==0:
                sign_curvature_change[int(self.nphi/2)::] = [-sign_curvature_change[i] for i in range(int(self.nphi/2),self.nphi)]
        if self.nfourier == 4:
            if (self.rc[0]+2*self.rc[1]+5*(self.rc[2] + 2*self.rc[3])) == 0 or (self.rc[0]+self.rc[1]+self.rc[2]+self.rc[3])**2+(self.zs[1]+2*self.zs[2]+3*self.zs[3])**2==0:
                sign_curvature_change[int(self.nphi/2)::] = [-sign_curvature_change[i] for i in range(int(self.nphi/2),self.nphi)]
    signed_curvature = curvature * sign_curvature_change
    curvature = signed_curvature
    for j in range(3):
        normal_cylindrical[:,j]   =   normal_cylindrical[:,j]*sign_curvature_change
        binormal_cylindrical[:,j] = binormal_cylindrical[:,j]*sign_curvature_change

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

    self.Bbar = self.spsi * np.mean(self.B0)
    # etabar_squared_over_curvature_squared = B1s^2+B1c^2/(B0*Bbar*curvature^2) = etabar^2/curvature^2 in quasisymmetry
    self.etabar_squared_over_curvature_squared = (self.B1s * self.B1s + self.B1c * self.B1c) / (signed_curvature * signed_curvature * self.B0 * self.Bbar)

    self.d_d_phi = spectral_diff_matrix(self.nphi, xmin = phi[0], xmax = phi[0] + 2*np.pi/self.nfp)#xmax=2 * np.pi / self.nfp)
    self.d_d_varphi = np.zeros((nphi, nphi))
    for j in range(nphi):
        self.d_d_varphi[j,:] = self.d_d_phi[j,:] * self.sG * G0 / (self.B0[j] * d_l_d_phi[j])

    # Compute the Boozer toroidal angle:
    self.varphi = np.zeros(nphi)
    d_l_d_phi_spline = self.convert_to_spline(d_l_d_phi)
    d_l_d_phi_from_zero = d_l_d_phi_spline(np.linspace(0,2*np.pi/self.nfp,self.nphi,endpoint=False))
    for j in range(1, nphi):
        # To get toroidal angle on the full mesh, we need d_l_d_phi on the half mesh.
        self.varphi[j] = self.varphi[j-1] + (d_l_d_phi_from_zero[j-1] + d_l_d_phi_from_zero[j])
    self.varphi = self.varphi * (0.5 * d_phi * 2 * np.pi / axis_length)

    # Add all results to self:
    self.d_phi = d_phi
    self.R0 = R0
    self.Z0 = Z0
    self.G0 = G0
    self.d_l_d_phi = d_l_d_phi
    self.axis_length = axis_length
    self.curvature = curvature
    self.signed_curvature = signed_curvature
    self.torsion = torsion
    self.X1s = self.B1s / (signed_curvature * self.B0)
    self.X1c = self.B1c / (signed_curvature * self.B0)
    self.min_R0 = fourier_minimum(self.R0)
    self.tangent_cylindrical = tangent_cylindrical
    self.normal_cylindrical = normal_cylindrical 
    self.binormal_cylindrical = binormal_cylindrical

    # The output is not stellarator-symmetric if (1) R0s is nonzero, (2) Z0c is nonzero, or (3) sigma_initial is nonzero
    self.lasym = np.max(np.abs(self.rs))>0 or np.max(np.abs(self.zc))>0 or np.abs(self.sigma0)>0

    # Functions that converts a toroidal angle phi0 on the axis to the axis radial and vertical coordinates
    self.R0_func = self.convert_to_spline(sum([self.rc[i]*np.cos(i*self.nfp*self.phi) +\
                                               self.rs[i]*np.sin(i*self.nfp*self.phi) \
                                              for i in range(len(self.rc))]))
    self.Z0_func = self.convert_to_spline(sum([self.zc[i]*np.cos(i*self.nfp*self.phi) +\
                                               self.zs[i]*np.sin(i*self.nfp*self.phi) \
                                              for i in range(len(self.zs))]))

    # Spline interpolants for the cylindrical components of the Frenet-Serret frame
    self.normal_R_spline = self.convert_to_spline(self.normal_cylindrical[:,0])
    self.normal_phi_spline = self.convert_to_spline(self.normal_cylindrical[:,1])
    self.normal_z_spline = self.convert_to_spline(self.normal_cylindrical[:,2])
    self.binormal_R_spline = self.convert_to_spline(self.binormal_cylindrical[:,0])
    self.binormal_phi_spline = self.convert_to_spline(self.binormal_cylindrical[:,1])
    self.binormal_z_spline = self.convert_to_spline(self.binormal_cylindrical[:,2])
    self.tangent_R_spline = self.convert_to_spline(self.tangent_cylindrical[:,0])
    self.tangent_phi_spline = self.convert_to_spline(self.tangent_cylindrical[:,1])
    self.tangent_z_spline = self.convert_to_spline(self.tangent_cylindrical[:,2])