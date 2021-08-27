"""
This module contains the calculation for the O(r^2) solution
"""

import logging
import numpy as np
from .util import mu0

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_r3(self):
    """
    Compute the O(r^3) quantities.
    Using the order_r_option_r3_flux_constraint equations
    from the quasisymmetry code
    """
    logger.debug('Calculating O(r^3) terms')
 
    # Shorthand
    sign_psi = self.spsi
    sign_G   = self.sG
    G2       = self.G2
    N_helicity = self.iota - self.iotaN
    B0 = self.B0
    G0 = self.G0
    I2 = self.I2
    X1c = self.X1c
    Y1c = self.Y1c
    Y1s = self.Y1s
    X20 = self.X20
    X2s = self.X2s
    X2c = self.X2c
    Y20 = self.Y20
    Y2s = self.Y2s
    Y2c = self.Y2c
    Z20 = self.Z20
    Z2s = self.Z2s
    Z2c = self.Z2c
    B20 = self.B20
    B1c = self.etabar * B0
    torsion = self.torsion
    curvature = self.curvature
    abs_G0_over_B0 = self.abs_G0_over_B0
    d_X1c_d_zeta = np.matmul(self.d_d_varphi,self.X1c)
    d_Y1c_d_zeta = np.matmul(self.d_d_varphi,self.Y1c)
    d_Z20_d_zeta = np.matmul(self.d_d_varphi,self.Z20)

    # The expression below is computed in "20190305-01 GarrenBoozer r2 corrected radius.nb" in the section "Approach of adding r^3 terms, assuming quasisymmetry"
    # 20190714: To account for QH cases, changed iota -> iota_N where it occurs 3 lines below:
    flux_constraint_coefficient = (-4*B0**2*G0*X20**2*Y1c**2 + 8*B0**2*G0*X20*X2c*Y1c**2 - 4*B0**2*G0*X2c**2*Y1c**2 - \
        4*B0**2*G0*X2s**2*Y1c**2 + 8*B0*G0*B1c*X1c*X2s*Y1c*Y1s + 16*B0**2*G0*X20*X2s*Y1c*Y1s + \
        2*B0**2*I2*self.iotaN*X1c**2*Y1s**2 - G0*B1c**2*X1c**2*Y1s**2 - 4*B0*G0*B20*X1c**2*Y1s**2 - \
        8*B0*G0*B1c*X1c*X20*Y1s**2 - 4*B0**2*G0*X20**2*Y1s**2 - 8*B0*G0*B1c*X1c*X2c*Y1s**2 - \
        8*B0**2*G0*X20*X2c*Y1s**2 - 4*B0**2*G0*X2c**2*Y1s**2 - 4*B0**2*G0*X2s**2*Y1s**2 + \
        8*B0**2*G0*X1c*X20*Y1c*Y20 - 8*B0**2*G0*X1c*X2c*Y1c*Y20 - 8*B0**2*G0*X1c*X2s*Y1s*Y20 - \
        4*B0**2*G0*X1c**2*Y20**2 - 8*B0**2*G0*X1c*X20*Y1c*Y2c + 8*B0**2*G0*X1c*X2c*Y1c*Y2c + \
        24*B0**2*G0*X1c*X2s*Y1s*Y2c + 8*B0**2*G0*X1c**2*Y20*Y2c - 4*B0**2*G0*X1c**2*Y2c**2 + \
        8*B0**2*G0*X1c*X2s*Y1c*Y2s - 8*B0*G0*B1c*X1c**2*Y1s*Y2s - 8*B0**2*G0*X1c*X20*Y1s*Y2s - \
        24*B0**2*G0*X1c*X2c*Y1s*Y2s - 4*B0**2*G0*X1c**2*Y2s**2 - 4*B0**2*G0*X1c**2*Z20**2 - \
        4*B0**2*G0*Y1c**2*Z20**2 - 4*B0**2*G0*Y1s**2*Z20**2 - 4*B0**2*abs_G0_over_B0*I2*Y1c*Y1s*Z2c + \
        8*B0**2*G0*X1c**2*Z20*Z2c + 8*B0**2*G0*Y1c**2*Z20*Z2c - 8*B0**2*G0*Y1s**2*Z20*Z2c - \
        4*B0**2*G0*X1c**2*Z2c**2 - 4*B0**2*G0*Y1c**2*Z2c**2 - 4*B0**2*G0*Y1s**2*Z2c**2 + \
        2*B0**2*abs_G0_over_B0*I2*X1c**2*Z2s + 2*B0**2*abs_G0_over_B0*I2*Y1c**2*Z2s - 2*B0**2*abs_G0_over_B0*I2*Y1s**2*Z2s + \
        16*B0**2*G0*Y1c*Y1s*Z20*Z2s - 4*B0**2*G0*X1c**2*Z2s**2 - 4*B0**2*G0*Y1c**2*Z2s**2 - \
        4*B0**2*G0*Y1s**2*Z2s**2 + B0**2*abs_G0_over_B0*I2*X1c**3*Y1s*torsion + B0**2*abs_G0_over_B0*I2*X1c*Y1c**2*Y1s*torsion + \
        B0**2*abs_G0_over_B0*I2*X1c*Y1s**3*torsion - B0**2*I2*X1c*Y1c*Y1s*d_X1c_d_zeta + \
        B0**2*I2*X1c**2*Y1s*d_Y1c_d_zeta)/(16*B0**2*G0*X1c**2*Y1s**2)

    self.X3c1 = self.X1c * flux_constraint_coefficient
    self.Y3c1 = self.Y1c * flux_constraint_coefficient
    self.Y3s1 = self.Y1s * flux_constraint_coefficient
    self.X3s1 = self.X1s * flux_constraint_coefficient
    self.Z3c1 = 0
    self.Z3s1 = 0

    self.X3c3 = 0
    self.X3s3 = 0
    self.Y3c3 = 0
    self.Y3s3 = 0
    self.Z3c3 = 0
    self.Z3s3 = 0

    # The expression below is derived in the O(r^2) paper, and in "20190318-01 Wrick's streamlined Garren-Boozer method, MHD.nb" in the section "Not assuming quasisymmetry".
    # Note Q = (1/2) * (XYEquation0 without X3 and Y3 terms) where XYEquation0 is the quantity in the above notebook.
    Q = -sign_psi * B0 * abs_G0_over_B0 / (2*G0*G0) * (self.iotaN * I2 + mu0 * self.p2 * G0 / (B0 * B0)) + 2 * (X2c * Y2s - X2s * Y2c) \
            + sign_psi * B0 / (2*G0) * (abs_G0_over_B0 * X20 * curvature - d_Z20_d_zeta) \
            + I2 / (4 * G0) * (-abs_G0_over_B0 * torsion * (X1c*X1c + Y1s*Y1s + Y1c*Y1c) + Y1c * d_X1c_d_zeta - X1c * d_Y1c_d_zeta)
    predicted_flux_constraint_coefficient = - Q / (2 * sign_G * sign_psi)

    B0_order_a_squared_to_cancel = -sign_G * B0 * B0 * (G2 + I2 * N_helicity) * abs_G0_over_B0 / (2*G0*G0) \
        -sign_G * sign_psi * B0 * 2 * (X2c * Y2s - X2s * Y2c) \
        -sign_G * B0 * B0 / (2*G0) * (abs_G0_over_B0 * X20 * curvature - d_Z20_d_zeta) \
        -sign_G * sign_psi * B0 * I2 / (4*G0) * (-abs_G0_over_B0 * torsion * (X1c*X1c + Y1c*Y1c + Y1s*Y1s) + Y1c * d_X1c_d_zeta - X1c * d_Y1c_d_zeta)

    logger.debug('max|flux_constraint_coefficient - predicted_flux_constraint_coefficient|:',np.max(abs(flux_constraint_coefficient - predicted_flux_constraint_coefficient)))
    logger.debug('max|flux_constraint_coefficient - B0_order_a_squared_to_cancel/(2*B0)|:',np.max(abs(flux_constraint_coefficient - B0_order_a_squared_to_cancel/(2*B0))))

    if np.max(abs(flux_constraint_coefficient - predicted_flux_constraint_coefficient)) > 1e-7\
    or np.max(abs(flux_constraint_coefficient - B0_order_a_squared_to_cancel/(2*B0))) > 1e-7:
        print("WARNING!!! Methods of computing lambda disagree!! Higher nphi resolution might be needed.")

    self.flux_constraint_coefficient = flux_constraint_coefficient
    self.B0_order_a_squared_to_cancel = B0_order_a_squared_to_cancel

    if self.helicity == 0:
        self.X3c1_untwisted = self.X3c1
        self.Y3c1_untwisted = self.Y3c1
        self.Y3s1_untwisted = self.Y3s1
        self.X3s1_untwisted = self.X3s1
        self.X3s3_untwisted = self.X3s3
        self.X3c3_untwisted = self.X3c3
        self.Y3c3_untwisted = self.Y3c3
        self.Y3s3_untwisted = self.Y3s3
        self.Z3s1_untwisted = self.Z3s1
        self.Z3s3_untwisted = self.Z3s3
        self.Z3c1_untwisted = self.Z3c1
        self.Z3c3_untwisted = self.Z3c3
    else:
        angle = -self.helicity * self.nfp * self.varphi
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        self.X3s1_untwisted = self.X3s1 *   cosangle  + self.X3c1 * sinangle
        self.X3c1_untwisted = self.X3s1 * (-sinangle) + self.X3c1 * cosangle
        self.Y3s1_untwisted = self.Y3s1 *   cosangle  + self.Y3c1 * sinangle
        self.Y3c1_untwisted = self.Y3s1 * (-sinangle) + self.Y3c1 * cosangle
        self.Z3s1_untwisted = self.Z3s1 *   cosangle  + self.Z3c1 * sinangle
        self.Z3c1_untwisted = self.Z3s1 * (-sinangle) + self.Z3c1 * cosangle
        sinangle = np.sin(3*angle)
        cosangle = np.cos(3*angle)
        self.X3s3_untwisted = self.X3s3 *   cosangle  + self.X3c3 * sinangle
        self.X3c3_untwisted = self.X3s3 * (-sinangle) + self.X3c3 * cosangle
        self.Y3s3_untwisted = self.Y3s3 *   cosangle  + self.Y3c3 * sinangle
        self.Y3c3_untwisted = self.Y3s3 * (-sinangle) + self.Y3c3 * cosangle
        self.Z3s3_untwisted = self.Z3s3 *   cosangle  + self.Z3c3 * sinangle
        self.Z3c3_untwisted = self.Z3s3 * (-sinangle) + self.Z3c3 * cosangle
