"""
This module contains the routines to compute
a given flux surface shape at a fixed
off-axis cylindrical toroidal angle
"""

import numpy as np
from scipy.optimize import root_scalar

def Frenet_to_cylindrical_residual_func(phi0, phi_target, qsc):
    """
    This function takes a point on the magnetic axis with a given
    toroidal angle phi0, computes the actual toroidal angle phi
    for an associated point at r>0 and finds the difference between
    this phi and the target value of phi

    Args:
        phi0 (float): toroidal angle on the axis
        phi_target (float): standard cylindrical toroidal angle
    """
    sinphi0 = np.sin(phi0)
    cosphi0 = np.cos(phi0)
    R0_at_phi0   = qsc.R0_func(phi0)
    X_at_phi0    = qsc.X_spline(phi0)
    Y_at_phi0    = qsc.Y_spline(phi0)
    normal_R     = qsc.normal_R_spline(phi0)
    normal_phi   = qsc.normal_phi_spline(phi0)
    binormal_R   = qsc.binormal_R_spline(phi0)
    binormal_phi = qsc.binormal_phi_spline(phi0)

    normal_x   =   normal_R * cosphi0 -   normal_phi * sinphi0
    normal_y   =   normal_R * sinphi0 +   normal_phi * cosphi0
    binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
    binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0

    total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
    total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y

    if qsc.order != 'r1':
        Z_at_phi0    = qsc.Z_spline(phi0)
        tangent_R    = qsc.tangent_R_spline(phi0)
        tangent_phi  = qsc.tangent_phi_spline(phi0)

        tangent_x = tangent_R * cosphi0 - tangent_phi * sinphi0
        tangent_y = tangent_R * sinphi0 + tangent_phi * cosphi0

        total_x = total_x + Z_at_phi0 * tangent_x
        total_y = total_y + Z_at_phi0 * tangent_y

    Frenet_to_cylindrical_residual = np.arctan2(total_y, total_x) - phi_target
    # We expect the residual to be less than pi in absolute value, so if it is not, the reason must be the branch cut:
    if (Frenet_to_cylindrical_residual >  np.pi): Frenet_to_cylindrical_residual = Frenet_to_cylindrical_residual - 2*np.pi
    if (Frenet_to_cylindrical_residual < -np.pi): Frenet_to_cylindrical_residual = Frenet_to_cylindrical_residual + 2*np.pi
    return Frenet_to_cylindrical_residual

def Frenet_to_cylindrical_1_point(phi0, qsc):
    """
    This function takes a point on the magnetic axis with a given
    toroidal angle phi0 and computes the cylindrical coordinate
    components R and Z for an associated point at r>0

    Args:
        phi0: toroidal angle on the axis
    """
    sinphi0 = np.sin(phi0)
    cosphi0 = np.cos(phi0)
    R0_at_phi0   = qsc.R0_func(phi0)
    z0_at_phi0   = qsc.Z0_func(phi0)
    X_at_phi0    = qsc.X_spline(phi0)
    Y_at_phi0    = qsc.Y_spline(phi0)
    Z_at_phi0    = qsc.Z_spline(phi0)
    normal_R     = qsc.normal_R_spline(phi0)
    normal_phi   = qsc.normal_phi_spline(phi0)
    normal_z     = qsc.normal_z_spline(phi0)
    binormal_R   = qsc.binormal_R_spline(phi0)
    binormal_phi = qsc.binormal_phi_spline(phi0)
    binormal_z   = qsc.binormal_z_spline(phi0)

    normal_x   =   normal_R * cosphi0 -   normal_phi * sinphi0
    normal_y   =   normal_R * sinphi0 +   normal_phi * cosphi0
    binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
    binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0

    total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
    total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y

    total_z = z0_at_phi0           + X_at_phi0 * normal_z + Y_at_phi0 * binormal_z

    if qsc.order != 'r1':
        tangent_R   = qsc.tangent_R_spline(phi0)
        tangent_phi = qsc.tangent_phi_spline(phi0)
        tangent_z   = qsc.tangent_z_spline(phi0)

        tangent_x = tangent_R * cosphi0 - tangent_phi * sinphi0
        tangent_y = tangent_R * sinphi0 + tangent_phi * cosphi0

        total_x = total_x + Z_at_phi0 * tangent_x
        total_y = total_y + Z_at_phi0 * tangent_y
        total_z = total_z + Z_at_phi0 * tangent_z

    total_R = np.sqrt(total_x * total_x + total_y * total_y)
    total_phi=np.arctan2(total_y, total_x)

    return total_R, total_z, total_phi

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
        Z_at_this_theta = 0 * X_at_this_theta
        if self.order != 'r1':
            # We need O(r^2) terms:
            cos2theta = np.cos(2 * theta[j_theta])
            sin2theta = np.sin(2 * theta[j_theta])
            X_at_this_theta += r * r * (self.X20_untwisted + self.X2c_untwisted * cos2theta + self.X2s_untwisted * sin2theta)
            Y_at_this_theta += r * r * (self.Y20_untwisted + self.Y2c_untwisted * cos2theta + self.Y2s_untwisted * sin2theta)
            Z_at_this_theta += r * r * (self.Z20_untwisted + self.Z2c_untwisted * cos2theta + self.Z2s_untwisted * sin2theta)
            if self.order == 'r3':
                # We need O(r^3) terms:
                costheta  = np.cos(theta[j_theta])
                sintheta  = np.sin(theta[j_theta])
                cos3theta = np.cos(3 * theta[j_theta])
                sin3theta = np.sin(3 * theta[j_theta])
                r3 = r * r * r
                X_at_this_theta += r3 * (self.X3c1_untwisted * costheta + self.X3s1_untwisted * sintheta
                                         + self.X3c3_untwisted * cos3theta + self.X3s3_untwisted * sin3theta)
                Y_at_this_theta += r3 * (self.Y3c1_untwisted * costheta + self.Y3s1_untwisted * sintheta
                                         + self.Y3c3_untwisted * cos3theta + self.Y3s3_untwisted * sin3theta)
                Z_at_this_theta += r3 * (self.Z3c1_untwisted * costheta + self.Z3s1_untwisted * sintheta
                                         + self.Z3c3_untwisted * cos3theta + self.Z3s3_untwisted * sin3theta)
        self.X_spline = self.convert_to_spline(X_at_this_theta)
        self.Y_spline = self.convert_to_spline(Y_at_this_theta)
        self.Z_spline = self.convert_to_spline(Z_at_this_theta)
        for j_phi in range(nphi_conversion):
            # Solve for the phi0 such that r0 + X1 n + Y1 b has the desired phi
            phi_target = phi_conversion[j_phi]
            phi0_rootSolve_min = phi_target - 1.0 / self.nfp
            phi0_rootSolve_max = phi_target + 1.0 / self.nfp
            res = root_scalar(Frenet_to_cylindrical_residual_func, xtol=1e-15, rtol=1e-15, maxiter=1000,\
                              args=(phi_target, self), bracket=[phi0_rootSolve_min, phi0_rootSolve_max], x0=phi_target)
            phi0_solution = res.root
            final_R, final_z, _ = Frenet_to_cylindrical_1_point(phi0_solution, self)
            R_2D[j_theta,j_phi] = final_R
            Z_2D[j_theta,j_phi] = final_z
            phi0_2D[j_theta,j_phi] = phi0_solution
            
    return R_2D, Z_2D, phi0_2D

def to_RZ(self,points):
    """
    Function to convert a set of points in (r,theta,phi0) coordinates
    where r=sqrt(2*psi/B0) is the near-axis radius, theta is the
    Boozer poloidal angle and phi0 is the cylindrical angle phi
    on the axis to cylindrical coordinates (R,Z)

    Args:
        points: an array of floats with dimension Nx3 with N the
        number of points to evaluate with each points having
        the (r,theta,phi0) values to evaluate
    """
    R = []
    Z = []
    phi = []
    for point in points:
        r      = point[0]
        theta  = point[1]
        phi0   = point[2]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        cos2theta = np.cos(2*theta)
        sin2theta = np.sin(2*theta)
        cos3theta = np.cos(3*theta)
        sin3theta = np.sin(3*theta)
        X_at_this_theta = r * (self.X1c_untwisted * costheta + self.X1s_untwisted * sintheta)
        Y_at_this_theta = r * (self.Y1c_untwisted * costheta + self.Y1s_untwisted * sintheta)
        Z_at_this_theta = 0 * X_at_this_theta
        if self.order != 'r1':
            # We need O(r^2) terms:
            X_at_this_theta += r * r * (self.X20_untwisted + self.X2c_untwisted * cos2theta + self.X2s_untwisted * sin2theta)
            Y_at_this_theta += r * r * (self.Y20_untwisted + self.Y2c_untwisted * cos2theta + self.Y2s_untwisted * sin2theta)
            Z_at_this_theta += r * r * (self.Z20_untwisted + self.Z2c_untwisted * cos2theta + self.Z2s_untwisted * sin2theta)
            if self.order == 'r3':
                # We need O(r^3) terms:
                r3 = r * r * r
                X_at_this_theta += r3 * (self.X3c1_untwisted * costheta + self.X3s1_untwisted * sintheta
                                         + self.X3c3_untwisted * cos3theta + self.X3s3_untwisted * sin3theta)
                Y_at_this_theta += r3 * (self.Y3c1_untwisted * costheta + self.Y3s1_untwisted * sintheta
                                         + self.Y3c3_untwisted * cos3theta + self.Y3s3_untwisted * sin3theta)
                Z_at_this_theta += r3 * (self.Z3c1_untwisted * costheta + self.Z3s1_untwisted * sintheta
                                         + self.Z3c3_untwisted * cos3theta + self.Z3s3_untwisted * sin3theta)
        self.X_spline = self.convert_to_spline(X_at_this_theta)
        self.Y_spline = self.convert_to_spline(Y_at_this_theta)
        self.Z_spline = self.convert_to_spline(Z_at_this_theta)
        final_R, final_Z, final_phi = Frenet_to_cylindrical_1_point(phi0, self)
        R.append(final_R)
        Z.append(final_Z)
        phi.append(final_phi)

    return R, Z, phi