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
    # Given a point on the axis with toroidal angle phi0, compute phi for the associated point at r>0,
    # and find the difference between this phi and the target value of phi.

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

def Frenet_to_cylindrical_1_point(phi0, X_spline, Y_spline,\
                                  R0_func, Z0_func,\
                                  normal_R_spline, normal_phi_spline, normal_z_spline,\
                                  binormal_R_spline, binormal_phi_spline, binormal_z_spline):
    # Given a point on the axis with toroidal angle phi0, compute R and z for the associated point at r>0.

    sinphi0 = np.sin(phi0)
    cosphi0 = np.cos(phi0)
    R0_at_phi0   = R0_func(phi0)
    z0_at_phi0   = Z0_func(phi0)
    X_at_phi0    = X_spline(phi0)
    Y_at_phi0    = Y_spline(phi0)
    normal_R     = normal_R_spline(phi0)
    normal_phi   = normal_phi_spline(phi0)
    normal_z     = normal_z_spline(phi0)
    binormal_R   = binormal_R_spline(phi0)
    binormal_phi = binormal_phi_spline(phi0)
    binormal_z   = binormal_z_spline(phi0)

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
    N_phi_conversion = self.nphi
    theta = np.linspace(0,2*np.pi,ntheta,endpoint=False)
    phi_conversion = np.linspace(0,2*np.pi/self.nfp,N_phi_conversion,endpoint=False)
    R_2D = np.zeros((ntheta,N_phi_conversion))
    Z_2D = np.zeros((ntheta,N_phi_conversion))
    phi0_2D = np.zeros((ntheta,N_phi_conversion))
    for j_theta in range(ntheta):
        costheta = np.cos(theta[j_theta])
        sintheta = np.sin(theta[j_theta])
        X_at_this_theta = r * (self.X1c_untwisted * costheta + self.X1s_untwisted * sintheta)
        Y_at_this_theta = r * (self.Y1c_untwisted * costheta + self.Y1s_untwisted * sintheta)
        def X_spline(phi): return self.convert_to_spline(phi,X_at_this_theta)
        def Y_spline(phi): return self.convert_to_spline(phi,Y_at_this_theta)
        for j_phi in range(N_phi_conversion):
            # Solve for the phi0 such that r0 + X1 n + Y1 b has the desired phi
            phi_target = phi_conversion[j_phi]
            phi0_rootSolve_min = phi_target - 1.0 / self.nfp
            phi0_rootSolve_max = phi_target + 1.0 / self.nfp
            res = root_scalar(Frenet_to_cylindrical_residual_func,\
                              args=(X_spline,Y_spline,phi_target, self.R0_func, self.normal_R_spline,\
                                    self.normal_phi_spline, self.binormal_R_spline, self.binormal_phi_spline),\
                              bracket=[phi0_rootSolve_min, phi0_rootSolve_max], x0=phi_target)
            phi0_solution = res.root

            final_R, final_z = Frenet_to_cylindrical_1_point(phi0_solution,X_spline,Y_spline,\
                                                            self.R0_func, self.Z0_func,\
                                                            self.normal_R_spline, self.normal_phi_spline, self.normal_z_spline,\
                                                            self.binormal_R_spline, self.binormal_phi_spline, self.binormal_z_spline)
            R_2D[j_theta,j_phi] = final_R
            Z_2D[j_theta,j_phi] = final_z
            phi0_2D[j_theta,j_phi] = phi0_solution
    return R_2D, Z_2D, phi0_2D
