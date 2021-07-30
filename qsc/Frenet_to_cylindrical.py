"""
This module contains the routines to compute
a given flux surface shape at a fixed
off-axis cylindrical toroidal angle
"""

import numpy as np
from scipy.optimize import root_scalar

def Frenet_to_cylindrical_residual_func(phi0, X_spline, Y_spline, phi_target,\
                                        R0_func, normal_R_spline, normal_phi_spline,\
                                        binormal_R_spline, binormal_phi_spline):
    """
    This function takes a point on the magnetic axis with a given
    toroidal angle phi0, computes the actual toroidal angle phi
    for an associated point at r>0 and finds the difference between
    this phi and the target value of phi

    Args:
        phi0: toroidal angle on the axis
        X_spline: spline interpolant for X
        Y_spline: spline interpolant for X
        phi_target: target toroidal angle phi
        R0_func: radial coordinate R0(phi0) of the axis shape
        Z0_func: vertical coordinate R0(phi0) of the axis shape
        normal_R_spline: spline interpolant for the R component of the axis normal vector
        normal_phi_spline: spline interpolant for the phi component of the axis normal vector
        binormal_R_spline: spline interpolant for the R component of the axis binormal vector
        binormal_phi_spline: spline interpolant for the phi component of the axis binormal vector
    """
    sinphi0 = np.sin(phi0)
    cosphi0 = np.cos(phi0)
    R0_at_phi0   = R0_func(phi0)
    X_at_phi0    = X_spline(phi0)
    Y_at_phi0    = Y_spline(phi0)
    normal_R     = normal_R_spline(phi0)
    normal_phi   = normal_phi_spline(phi0)
    binormal_R   = binormal_R_spline(phi0)
    binormal_phi = binormal_phi_spline(phi0)

    normal_x   =   normal_R * cosphi0 -   normal_phi * sinphi0
    normal_y   =   normal_R * sinphi0 +   normal_phi * cosphi0
    binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
    binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0

    total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
    total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y

    Frenet_to_cylindrical_residual = np.arctan2(total_y, total_x) - phi_target
    # We expect the residual to be less than pi in absolute value, so if it is not, the reason must be the branch cut:
    if (Frenet_to_cylindrical_residual >  np.pi): Frenet_to_cylindrical_residual = Frenet_to_cylindrical_residual - 2*np.pi
    if (Frenet_to_cylindrical_residual < -np.pi): Frenet_to_cylindrical_residual = Frenet_to_cylindrical_residual + 2*np.pi
    return Frenet_to_cylindrical_residual

def Frenet_to_cylindrical_1_point(phi0, arr):
    """
    This function takes a point on the magnetic axis with a given
    toroidal angle phi0 and computes the cylindrical coordinate
    components R and Z for an associated point at r>0

    Args:
        phi0: toroidal angle on the axis
        arr: Array with 8 entries corresponding to these entries at phi0
            X_spline: spline interpolant for X
            Y_spline: spline interpolant for X
            R0_func: radial coordinate R0(phi0) of the axis shape
            Z0_func: vertical coordinate R0(phi0) of the axis shape
            normal_R_spline: spline interpolant for the R component of the axis normal vector
            normal_phi_spline: spline interpolant for the phi component of the axis normal vector
            normal_Z_spline: spline interpolant for the Z component of the axis normal vector
            binormal_R_spline: spline interpolant for the R component of the axis binormal vector
            binormal_phi_spline: spline interpolant for the phi component of the axis binormal vector
            binormal_Z_spline: spline interpolant for the Z component of the axis binormal vector
    """
    sinphi0 = np.sin(phi0)
    cosphi0 = np.cos(phi0)
    X_at_phi0, Y_at_phi0, R0_at_phi0, z0_at_phi0,\
    normal_R, normal_phi, normal_z,\
    binormal_R, binormal_phi, binormal_z = arr

    normal_x   =   normal_R * cosphi0 -   normal_phi * sinphi0
    normal_y   =   normal_R * sinphi0 +   normal_phi * cosphi0
    binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
    binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0

    total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
    total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y

    total_z = z0_at_phi0           + X_at_phi0 * normal_z + Y_at_phi0 * binormal_z
    total_R = np.sqrt(total_x * total_x + total_y * total_y)
    return total_R, total_z

def Frenet_to_cylindrical(self, r, ntheta=20):
    """
    Function to convert the near-axis coordinate system to
    a cylindrical one for a surface at a particular radius,
    outputing the following arrays: R(theta,varphi),
    phi(theta,varphi) and Z(theta,varphi) with R,phi,Z cylindrical
    coordinates and theta and varphi Boozer coordinates

    Args:
        r:  near-axis radius r of the desired boundary surface
        ntheta: resolution in the poloidal angle theta
    """
    nphi_conversion = self.nphi
    theta = np.linspace(0,2*np.pi,ntheta,endpoint=False)
    phi_conversion = np.linspace(0,2*np.pi/self.nfp,nphi_conversion,endpoint=False)
    R_2D = np.zeros((ntheta,nphi_conversion))
    Z_2D = np.zeros((ntheta,nphi_conversion))
    phi0_2D = np.zeros((ntheta,nphi_conversion))
    for j_theta in range(ntheta):
        costheta = np.cos(theta[j_theta])
        sintheta = np.sin(theta[j_theta])
        X_at_this_theta = r * (self.X1c_untwisted * costheta + self.X1s_untwisted * sintheta)
        Y_at_this_theta = r * (self.Y1c_untwisted * costheta + self.Y1s_untwisted * sintheta)
        X_spline = self.convert_to_spline(X_at_this_theta)
        Y_spline = self.convert_to_spline(Y_at_this_theta)
        for j_phi in range(nphi_conversion):
            # Solve for the phi0 such that r0 + X1 n + Y1 b has the desired phi
            phi_target = phi_conversion[j_phi]
            phi0_rootSolve_min = phi_target - 1.0 / self.nfp
            phi0_rootSolve_max = phi_target + 1.0 / self.nfp
            res = root_scalar(Frenet_to_cylindrical_residual_func, xtol=1e-15, rtol=1e-15, maxiter=1000,\
                              args=(X_spline,Y_spline,phi_target, self.R0_func, self.normal_R_spline,\
                                    self.normal_phi_spline, self.binormal_R_spline, self.binormal_phi_spline),\
                              bracket=[phi0_rootSolve_min, phi0_rootSolve_max], x0=phi_target)
            phi0_solution = res.root
            arr = [X_spline(phi0_solution), Y_spline(phi0_solution), self.R0_func(phi0_solution), self.Z0_func(phi0_solution),\
                   self.normal_R_spline(phi0_solution), self.normal_phi_spline(phi0_solution), self.normal_z_spline(phi0_solution),\
                   self.binormal_R_spline(phi0_solution), self.binormal_phi_spline(phi0_solution), self.binormal_z_spline(phi0_solution)]
            final_R, final_z = Frenet_to_cylindrical_1_point(phi0_solution, arr)
            R_2D[j_theta,j_phi] = final_R
            Z_2D[j_theta,j_phi] = final_z
            phi0_2D[j_theta,j_phi] = phi0_solution
    return R_2D, Z_2D, phi0_2D
