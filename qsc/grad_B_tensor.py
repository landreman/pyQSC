#!/usr/bin/env python3

"""
Functions for computing the grad B tensor and grad grad B tensor.
"""

import numpy as np
import logging
from .util import Struct, fourier_minimum

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_grad_B_tensor(self):
    """
    Compute the components of the grad B tensor, and the scale
    length L_{\nabla B} associated with the Frobenius norm of this
    tensor.

    self should be an instance of Qsc with X1c, Y1s etc populated.
    """
    s = self # Shorthand
    tensor = Struct()
    
    factor = s.spsi * s.B0 / s.d_l_d_varphi
    tensor.tn = s.sG * s.B0 * s.curvature
    tensor.nt = tensor.tn
    tensor.bb = factor * (s.X1c * s.d_Y1s_d_varphi - s.iotaN * s.X1c * s.Y1c)
    tensor.nn = factor * (s.d_X1c_d_varphi * s.Y1s + s.iotaN * s.X1c * s.Y1c)
    tensor.bn = factor * (-s.sG * s.spsi * s.d_l_d_varphi * s.torsion \
                        - s.iotaN * s.X1c * s.X1c)
    tensor.nb = factor * (s.d_Y1c_d_varphi * s.Y1s - s.d_Y1s_d_varphi * s.Y1c \
                        + s.sG * s.spsi * s.d_l_d_varphi * s.torsion \
                        + s.iotaN * (s.Y1s * s.Y1s + s.Y1c * s.Y1c))

    self.grad_B_tensor = tensor
    self.grad_B_colon_grad_B = tensor.tn * tensor.tn + tensor.nt * tensor.nt \
        + tensor.bb * tensor.bb + tensor.nn * tensor.nn \
        + tensor.nb * tensor.nb + tensor.bn * tensor.bn

    self.L_grad_B = s.B0 * np.sqrt(2 / self.grad_B_colon_grad_B)
    self.inv_L_grad_B = 1.0 / self.L_grad_B
    self.min_L_grad_B = fourier_minimum(self.L_grad_B)
    
def calculate_grad_grad_B_tensor(self, two_ways=False):
    """
    Compute the components of the grad grad B tensor, and the scale
    length L_{\nabla \nabla B} associated with the Frobenius norm of this
    tensor.

    self should be an instance of Qsc with X1c, Y1s etc populated.
    """

    # Shortcuts
    s = self
    
    X1c = s.X1c
    Y1s = s.Y1s
    Y1c = s.Y1c

    X20 = s.X20
    X2s = s.X2s
    X2c = s.X2c

    Y20 = s.Y20
    Y2s = s.Y2s
    Y2c = s.Y2c

    Z20 = s.Z20
    Z2s = s.Z2s
    Z2c = s.Z2c

    iota_N0 = s.iotaN
    iota = s.iota
    lp = np.abs(s.G0) / s.B0

    curvature = s.curvature
    torsion = s.torsion

    sign_G = s.sG
    sign_psi = s.spsi
    B0 = s.B0
    G0 = s.G0
    I2 = s.I2
    G2 = s.G2
    p2 = s.p2

    B20 = s.B20
    B2s = s.B2s
    B2c = s.B2c

    d_X1c_d_varphi = s.d_X1c_d_varphi
    d_Y1s_d_varphi = s.d_Y1s_d_varphi
    d_Y1c_d_varphi = s.d_Y1c_d_varphi

    d_X20_d_varphi = s.d_X20_d_varphi
    d_X2s_d_varphi = s.d_X2s_d_varphi
    d_X2c_d_varphi = s.d_X2c_d_varphi

    d_Y20_d_varphi = s.d_Y20_d_varphi
    d_Y2s_d_varphi = s.d_Y2s_d_varphi
    d_Y2c_d_varphi = s.d_Y2c_d_varphi

    d_Z20_d_varphi = s.d_Z20_d_varphi
    d_Z2s_d_varphi = s.d_Z2s_d_varphi
    d_Z2c_d_varphi = s.d_Z2c_d_varphi

    d2_X1c_d_varphi2 = s.d2_X1c_d_varphi2
    d2_Y1s_d_varphi2 = s.d2_Y1s_d_varphi2
    d2_Y1c_d_varphi2 = s.d2_Y1c_d_varphi2
    d_curvature_d_varphi = s.d_curvature_d_varphi
    d_torsion_d_varphi = s.d_torsion_d_varphi

    grad_grad_B = np.zeros((s.nphi, 3, 3, 3))
    grad_grad_B_alt = np.zeros((s.nphi, 3, 3, 3))

    # The elements that follow are computed in the Mathematica notebook "20200407-01 Grad grad B tensor near axis"
    # and then formatted for fortran by the python script process_grad_grad_B_tensor_code

    # The order is (normal, binormal, tangent). So element 123 means nbt.

    # Element 111
    grad_grad_B[:,0,0,0] =(B0*B0*B0*B0*lp*lp*(8*iota_N0*X2c*Y1c*\
                                              Y1s + 4*iota_N0*X2s*\
                                              (-Y1c*Y1c + Y1s*Y1s) + \
                                              2*iota_N0*X1c*Y1s*Y20 + \
                                              2*iota_N0*X1c*Y1s*Y2c - \
                                              2*iota_N0*X1c*Y1c*Y2s + \
                                              5*iota_N0*X1c*X1c*Y1c*Y1s*\
                                              curvature - \
                                              2*Y1c*Y20*d_X1c_d_varphi + \
                                              2*Y1c*Y2c*d_X1c_d_varphi + \
                                              2*Y1s*Y2s*d_X1c_d_varphi + \
                                              5*X1c*Y1s*Y1s*curvature*\
                                              d_X1c_d_varphi + \
                                              2*Y1c*Y1c*d_X20_d_varphi + \
                                              2*Y1s*Y1s*d_X20_d_varphi - \
                                              2*Y1c*Y1c*d_X2c_d_varphi + \
                                              2*Y1s*Y1s*d_X2c_d_varphi - \
                                              4*Y1c*Y1s*d_X2s_d_varphi))/\
                                              (G0*G0*G0)

    # Element 112
    grad_grad_B[:,0,0,1] =(B0*B0*B0*B0*lp*lp*(Y1c*Y1c*\
                                              (-6*iota_N0*Y2s + \
                                               5*iota_N0*X1c*Y1s*\
                                               curvature + \
                                               2*(lp*X20*torsion - \
                                                  lp*X2c*torsion + \
                                                  d_Y20_d_varphi - \
                                                  d_Y2c_d_varphi)) + \
                                              Y1s*(5*iota_N0*X1c*Y1s*Y1s*\
                                                   curvature + \
                                                   2*(lp*X1c*Y2s*torsion + \
                                                      Y2s*d_Y1c_d_varphi - \
                                                      (Y20 + Y2c)*\
                                                      d_Y1s_d_varphi) + \
                                                   Y1s*(6*iota_N0*Y2s + \
                                                        2*lp*X20*torsion + \
                                                        2*lp*X2c*torsion + \
                                                        5*lp*X1c*X1c*curvature*\
                                                        torsion + \
                                                        5*X1c*curvature*\
                                                        d_Y1c_d_varphi + \
                                                        2*d_Y20_d_varphi + \
                                                        2*d_Y2c_d_varphi)) + \
                                              Y1c*(2*(lp*X1c*\
                                                      (-Y20 + Y2c)*torsion - \
                                                      Y20*d_Y1c_d_varphi + \
                                                      Y2c*d_Y1c_d_varphi + \
                                                      Y2s*d_Y1s_d_varphi) + \
                                                   Y1s*(12*iota_N0*Y2c - \
                                                        4*lp*X2s*torsion - \
                                                        5*X1c*curvature*\
                                                        d_Y1s_d_varphi - \
                                                        4*d_Y2s_d_varphi))))/(G0*G0*G0)

    # Element 113
    grad_grad_B[:,0,0,2] =-((B0*B0*B0*lp*lp*(2*Y1c*Y1c*\
                                             (2*B2c*G0*lp + B0*G2*lp + B0*I2*lp*iota - \
                                              2*G0*lp*B20 + 2*B0*G0*iota_N0*Z2s + \
                                              B0*G0*lp*X20*curvature - \
                                              B0*G0*lp*X2c*curvature - \
                                              B0*G0*d_Z20_d_varphi + \
                                              B0*G0*d_Z2c_d_varphi) + \
                                             Y1s*(-2*B0*G0*lp*X1c*Y2s*\
                                                  curvature + \
                                                  Y1s*(-4*B2c*G0*lp + 2*B0*G2*lp + \
                                                       2*B0*I2*lp*iota - 4*G0*lp*B20 - \
                                                       4*B0*G0*iota_N0*Z2s + \
                                                       2*B0*G0*lp*X20*curvature + \
                                                       2*B0*G0*lp*X2c*curvature + \
                                                       B0*G0*lp*X1c*X1c*curvature*curvature - \
                                                       2*B0*G0*d_Z20_d_varphi - \
                                                       2*B0*G0*d_Z2c_d_varphi)) + \
                                             2*G0*Y1c*(B0*lp*X1c*\
                                                       (Y20 - Y2c)*curvature + \
                                                       2*Y1s*(2*B2s*lp - 2*B0*iota_N0*Z2c - \
                                                              B0*lp*X2s*curvature + \
                                                              B0*d_Z2s_d_varphi))))/(G0*G0*G0*G0))

    # Element 121
    grad_grad_B[:,0,1,0] =-((B0*B0*B0*B0*lp*lp*(3*iota_N0*X1c*X1c*X1c*Y1s*\
                                                curvature + \
                                                3*lp*X1c*X1c*Y1s*Y1s*curvature*\
                                                torsion + \
                                                2*(X2s*Y1s*\
                                                   (-2*lp*Y1c*torsion + \
                                                    d_X1c_d_varphi) + \
                                                   X20*(lp*Y1c*Y1c*torsion + \
                                                        lp*Y1s*Y1s*torsion - \
                                                        Y1c*d_X1c_d_varphi) + \
                                                   X2c*(-(lp*Y1c*Y1c*\
                                                          torsion) + \
                                                        lp*Y1s*Y1s*torsion + \
                                                        Y1c*d_X1c_d_varphi)) - \
                                                2*X1c*(3*iota_N0*X2s*Y1c - \
                                                       iota_N0*X20*Y1s - \
                                                       3*iota_N0*X2c*Y1s + \
                                                       lp*Y1c*Y20*torsion - \
                                                       lp*Y1c*Y2c*torsion - \
                                                       lp*Y1s*Y2s*torsion - \
                                                       Y1c*d_X20_d_varphi + \
                                                       Y1c*d_X2c_d_varphi + \
                                                       Y1s*d_X2s_d_varphi)))/\
                            (G0*G0*G0))

    # Element 122
    grad_grad_B[:,0,1,1] =(B0*B0*B0*B0*lp*lp*(-4*iota_N0*X1c*Y1s*\
                                              Y2c + 4*iota_N0*X1c*Y1c*\
                                              Y2s - 3*iota_N0*X1c*X1c*Y1c*\
                                              Y1s*curvature + \
                                              2*X20*Y1c*d_Y1c_d_varphi + \
                                              2*X20*Y1s*d_Y1s_d_varphi + \
                                              3*X1c*X1c*Y1s*curvature*\
                                              d_Y1s_d_varphi + \
                                              2*X2s*(iota_N0*Y1c*Y1c - \
                                                     Y1s*(iota_N0*Y1s + \
                                                          d_Y1c_d_varphi) - \
                                                     Y1c*d_Y1s_d_varphi) - \
                                              2*X2c*(Y1c*\
                                                     (2*iota_N0*Y1s + d_Y1c_d_varphi) \
                                                     - Y1s*d_Y1s_d_varphi) - \
                                              2*X1c*Y1c*d_Y20_d_varphi + \
                                              2*X1c*Y1c*d_Y2c_d_varphi + \
                                              2*X1c*Y1s*d_Y2s_d_varphi))/\
                                              (G0*G0*G0)
    #       (2*iota_N0*Y1s + d_Y1c_d_varphi) \\

    # Element 123
    grad_grad_B[:,0,1,2] =(2*B0*B0*B0*lp*lp*X1c*\
                           (Y1c*(2*B2c*G0*lp + B0*G2*lp + B0*I2*lp*iota - \
                                 2*G0*lp*B20 + 2*B0*G0*iota_N0*Z2s + \
                                 2*B0*G0*lp*X20*curvature - \
                                 2*B0*G0*lp*X2c*curvature - \
                                 B0*G0*d_Z20_d_varphi + \
                                 B0*G0*d_Z2c_d_varphi) + \
                            G0*Y1s*(2*B2s*lp - 2*B0*iota_N0*Z2c - \
                                    2*B0*lp*X2s*curvature + \
                                    B0*d_Z2s_d_varphi)))/(G0*G0*G0*G0)

    # Element 131
    grad_grad_B[:,0,2,0] =(B0*B0*B0*B0*lp*(-4*lp*lp*X2s*Y1c*Y1s*\
                                           curvature + \
                                           2*lp*lp*X2c*(-Y1c*Y1c + Y1s*Y1s)*\
                                           curvature + \
                                           2*lp*lp*X20*(Y1c*Y1c + Y1s*Y1s)*\
                                           curvature - \
                                           2*lp*lp*X1c*Y1c*Y20*\
                                           curvature + \
                                           2*lp*lp*X1c*Y1c*Y2c*\
                                           curvature + \
                                           2*lp*lp*X1c*Y1s*Y2s*\
                                           curvature + \
                                           3*lp*lp*X1c*X1c*Y1s*Y1s*\
                                           curvature*curvature + \
                                           lp*iota_N0*X1c*X1c*X1c*Y1s*\
                                           torsion - lp*iota_N0*X1c*\
                                           Y1c*Y1c*Y1s*torsion - \
                                           lp*iota_N0*X1c*Y1s*Y1s*Y1s*\
                                           torsion - Y1s*Y1s*\
                                           d_X1c_d_varphi*d_X1c_d_varphi + \
                                           iota_N0*X1c*X1c*Y1s*\
                                           d_Y1c_d_varphi - \
                                           lp*X1c*Y1s*Y1s*torsion*\
                                           d_Y1c_d_varphi - \
                                           iota_N0*X1c*X1c*Y1c*\
                                           d_Y1s_d_varphi + \
                                           lp*X1c*Y1c*Y1s*\
                                           torsion*d_Y1s_d_varphi + \
                                           X1c*Y1s*Y1s*d2_X1c_d_varphi2))/\
                                           (G0*G0*G0)

    # Element 132
    grad_grad_B[:,0,2,1] =(B0*B0*B0*B0*lp*(-(Y1s*d_X1c_d_varphi*\
                                             (iota_N0*Y1c*Y1c + \
                                              Y1s*(iota_N0*Y1s + \
                                                   d_Y1c_d_varphi) - \
                                              Y1c*d_Y1s_d_varphi)) + \
                                           lp*X1c*X1c*Y1s*\
                                           (2*iota_N0*Y1c*torsion - \
                                            torsion*d_Y1s_d_varphi + \
                                            Y1s*d_torsion_d_varphi) + \
                                           X1c*(Y1c*d_Y1s_d_varphi*\
                                                (-(iota_N0*Y1c) + d_Y1s_d_varphi) \
                                                + Y1s*Y1s*(lp*torsion*\
                                                           d_X1c_d_varphi + \
                                                           iota_N0*d_Y1s_d_varphi + \
                                                           d2_Y1c_d_varphi2) - \
                                                Y1s*(d_Y1c_d_varphi*\
                                                     d_Y1s_d_varphi + \
                                                     Y1c*(-2*iota_N0*d_Y1c_d_varphi + \
                                                          d2_Y1s_d_varphi2)))))/(G0*G0*G0)
    #       (-(iota_N0*Y1c) + d_Y1s_d_varphi) \\

    # Element 133
    grad_grad_B[:,0,2,2] =(B0*B0*B0*B0*lp*lp*X1c*Y1s*\
                           (-(Y1s*curvature*\
                              d_X1c_d_varphi) + \
                            X1c*(-(iota_N0*Y1c*\
                                   curvature) + \
                                 Y1s*d_curvature_d_varphi)))/\
                                 (G0*G0*G0)

    # Element 211
    grad_grad_B[:,1,0,0] =(-2*B0*B0*B0*B0*lp*lp*X1c*\
                           (-2*iota_N0*X2s*Y1c + \
                            2*iota_N0*X2c*Y1s - \
                            iota_N0*X1c*Y2s + \
                            iota_N0*X1c*X1c*Y1s*curvature + \
                            lp*X1c*Y1s*Y1s*curvature*\
                            torsion - Y20*\
                            d_X1c_d_varphi + \
                            Y2c*d_X1c_d_varphi + \
                            Y1c*d_X20_d_varphi - \
                            Y1c*d_X2c_d_varphi - \
                            Y1s*d_X2s_d_varphi))/(G0*G0*G0)

    # Element 212
    grad_grad_B[:,1,0,1] =(2*B0*B0*B0*B0*lp*lp*X1c*\
                           (lp*X1c*Y20*torsion - \
                            lp*X1c*Y2c*torsion + \
                            Y20*d_Y1c_d_varphi - \
                            Y2c*d_Y1c_d_varphi - \
                            Y2s*d_Y1s_d_varphi + \
                            Y1c*(3*iota_N0*Y2s - \
                                 lp*X20*torsion + \
                                 lp*X2c*torsion - \
                                 d_Y20_d_varphi + d_Y2c_d_varphi) \
                            + Y1s*(iota_N0*Y20 - \
                                   3*iota_N0*Y2c - \
                                   iota_N0*X1c*Y1c*curvature + \
                                   lp*X2s*torsion + \
                                   X1c*curvature*\
                                   d_Y1s_d_varphi + d_Y2s_d_varphi))\
                           )/(G0*G0*G0)
    #       d_Y20_d_varphi + d_Y2c_d_varphi) \\
    #       d_Y1s_d_varphi + d_Y2s_d_varphi))\\

    # Element 213
    grad_grad_B[:,1,0,2] =(2*B0*B0*B0*lp*lp*X1c*\
                           (Y1c*(2*B2c*G0*lp + B0*G2*lp + B0*I2*lp*iota - \
                                 2*G0*lp*B20 + 2*B0*G0*iota_N0*Z2s + \
                                 B0*G0*lp*X20*curvature - \
                                 B0*G0*lp*X2c*curvature - \
                                 B0*G0*d_Z20_d_varphi + \
                                 B0*G0*d_Z2c_d_varphi) + \
                            G0*(B0*lp*X1c*(Y20 - Y2c)*\
                                curvature + \
                                Y1s*(2*B2s*lp - 2*B0*iota_N0*Z2c - \
                                     B0*lp*X2s*curvature + \
                                     B0*d_Z2s_d_varphi))))/(G0*G0*G0*G0)

    # Element 221
    grad_grad_B[:,1,1,0] =(-2*B0*B0*B0*B0*lp*lp*X1c*\
                           (lp*X2c*Y1c*torsion + \
                            lp*X2s*Y1s*torsion - \
                            X2c*d_X1c_d_varphi + \
                            X20*(-(lp*Y1c*torsion) + \
                                 d_X1c_d_varphi) + \
                            X1c*(3*iota_N0*X2s + \
                                 lp*Y20*torsion - \
                                 lp*Y2c*torsion - \
                                 d_X20_d_varphi + d_X2c_d_varphi)))/\
                                 (G0*G0*G0)

    # Element 222
    grad_grad_B[:,1,1,1] =(-2*B0*B0*B0*B0*lp*lp*X1c*\
                           (-(iota_N0*X2c*Y1s) + \
                            2*iota_N0*X1c*Y2s - \
                            X2c*d_Y1c_d_varphi + \
                            X20*(iota_N0*Y1s + \
                                 d_Y1c_d_varphi) + \
                            X2s*(iota_N0*Y1c - \
                                 d_Y1s_d_varphi) - \
                            X1c*d_Y20_d_varphi + \
                            X1c*d_Y2c_d_varphi))/(G0*G0*G0)

    # Element 223
    grad_grad_B[:,1,1,2] =(-2*B0*B0*B0*lp*lp*X1c*X1c*\
                           (2*B2c*G0*lp + B0*G2*lp + B0*I2*lp*iota - 2*G0*lp*B20 + \
                            2*B0*G0*iota_N0*Z2s + \
                            2*B0*G0*lp*X20*curvature - \
                            2*B0*G0*lp*X2c*curvature - \
                            B0*G0*d_Z20_d_varphi + \
                            B0*G0*d_Z2c_d_varphi))/(G0*G0*G0*G0)

    # Element 231
    grad_grad_B[:,1,2,0] =(B0*B0*B0*B0*lp*X1c*(-2*lp*lp*X20*Y1c*\
                                               curvature + \
                                               2*lp*lp*X2c*Y1c*curvature + \
                                               2*lp*lp*X2s*Y1s*curvature + \
                                               2*lp*lp*X1c*Y20*curvature - \
                                               2*lp*lp*X1c*Y2c*curvature + \
                                               2*lp*iota_N0*X1c*Y1c*Y1s*\
                                               torsion - iota_N0*X1c*Y1s*\
                                               d_X1c_d_varphi + \
                                               lp*Y1s*Y1s*torsion*\
                                               d_X1c_d_varphi + \
                                               iota_N0*X1c*X1c*d_Y1s_d_varphi - \
                                               lp*X1c*Y1s*torsion*\
                                               d_Y1s_d_varphi - \
                                               lp*X1c*Y1s*Y1s*\
                                               d_torsion_d_varphi))/(G0*G0*G0)

    # Element 232
    grad_grad_B[:,1,2,1] =(B0*B0*B0*B0*lp*X1c*(-(lp*iota_N0*X1c*X1c*\
                                                 Y1s*torsion) + \
                                               lp*Y1s*torsion*\
                                               (iota_N0*Y1c*Y1c + \
                                                Y1s*(iota_N0*Y1s + \
                                                     d_Y1c_d_varphi) - \
                                                Y1c*d_Y1s_d_varphi) + \
                                               X1c*((iota_N0*Y1c - \
                                                     d_Y1s_d_varphi)*d_Y1s_d_varphi \
                                                    + Y1s*(-(iota_N0*d_Y1c_d_varphi) + \
                                                           d2_Y1s_d_varphi2))))/(G0*G0*G0)
    #       d_Y1s_d_varphi)*d_Y1s_d_varphi \\

    # Element 233
    grad_grad_B[:,1,2,2] =(B0*B0*B0*B0*lp*lp*X1c*X1c*Y1s*curvature*\
                           (iota_N0*X1c + 2*lp*Y1s*torsion))/\
                           (G0*G0*G0)

    # Element 311
    grad_grad_B[:,2,0,0] =(B0*B0*B0*B0*lp*X1c*Y1s*\
                           (lp*iota_N0*X1c*X1c*torsion - \
                            lp*iota_N0*Y1c*Y1c*torsion - \
                            lp*iota_N0*Y1s*Y1s*torsion - \
                            lp*Y1s*torsion*\
                            d_Y1c_d_varphi + \
                            X1c*(2*lp*lp*Y1s*curvature*curvature + \
                                 iota_N0*d_Y1c_d_varphi) + \
                            d_X1c_d_varphi*d_Y1s_d_varphi + \
                            Y1c*(iota_N0*d_X1c_d_varphi + \
                                 lp*torsion*d_Y1s_d_varphi) + \
                            Y1s*d2_X1c_d_varphi2))/(G0*G0*G0)

    # Element 312
    grad_grad_B[:,2,0,1] =(B0*B0*B0*B0*lp*X1c*Y1s*\
                           (lp*X1c*(2*iota_N0*Y1c*\
                                    torsion + \
                                    Y1s*d_torsion_d_varphi) + \
                            Y1s*(2*lp*torsion*\
                                 d_X1c_d_varphi + \
                                 2*iota_N0*d_Y1s_d_varphi + \
                                 d2_Y1c_d_varphi2) + \
                            Y1c*(2*iota_N0*d_Y1c_d_varphi - \
                                 d2_Y1s_d_varphi2)))/(G0*G0*G0)

    # Element 313
    grad_grad_B[:,2,0,2] =(B0*B0*B0*B0*lp*lp*X1c*X1c*Y1s*\
                           (-(iota_N0*Y1c*curvature) + \
                            curvature*d_Y1s_d_varphi + \
                            Y1s*d_curvature_d_varphi))/\
                            (G0*G0*G0)

    # Element 321
    grad_grad_B[:,2,1,0] =-((B0*B0*B0*B0*lp*X1c*X1c*Y1s*\
                             (-2*lp*iota_N0*Y1c*torsion + \
                              2*iota_N0*d_X1c_d_varphi + \
                              2*lp*torsion*d_Y1s_d_varphi + \
                              lp*Y1s*d_torsion_d_varphi))/\
                            (G0*G0*G0))

    # Element 322
    grad_grad_B[:,2,1,1] =-((B0*B0*B0*B0*lp*X1c*Y1s*\
                             (lp*iota_N0*X1c*X1c*torsion - \
                              lp*iota_N0*Y1c*Y1c*torsion - \
                              lp*iota_N0*Y1s*Y1s*torsion - \
                              lp*Y1s*torsion*\
                              d_Y1c_d_varphi - \
                              d_X1c_d_varphi*d_Y1s_d_varphi + \
                              Y1c*(iota_N0*d_X1c_d_varphi + \
                                   lp*torsion*d_Y1s_d_varphi) + \
                              X1c*(iota_N0*d_Y1c_d_varphi - \
                                   d2_Y1s_d_varphi2)))/(G0*G0*G0))

    # Element 323
    grad_grad_B[:,2,1,2] =(B0*B0*B0*B0*lp*lp*X1c*X1c*Y1s*curvature*\
                           (iota_N0*X1c + 2*lp*Y1s*torsion))/\
                           (G0*G0*G0)

    # Element 331
    grad_grad_B[:,2,2,0] =(B0*B0*B0*B0*lp*lp*X1c*X1c*Y1s*\
                           (-(iota_N0*Y1c*curvature) + \
                            curvature*d_Y1s_d_varphi + \
                            Y1s*d_curvature_d_varphi))/\
                            (G0*G0*G0)

    # Element 332
    grad_grad_B[:,2,2,1] =-((B0*B0*B0*B0*lp*lp*X1c*Y1s*curvature*\
                             (iota_N0*Y1c*Y1c + \
                              Y1s*(iota_N0*Y1s + \
                                   d_Y1c_d_varphi) - \
                              Y1c*d_Y1s_d_varphi))/(G0*G0*G0))

    # Element 333
    grad_grad_B[:,2,2,2] =(-2*B0*B0*B0*B0*lp*lp*lp*X1c*X1c*Y1s*Y1s*\
                           curvature*curvature)/(G0*G0*G0)


    self.grad_grad_B = grad_grad_B

    # Compute the (inverse) scale length
    squared = grad_grad_B * grad_grad_B
    norm_squared = np.sum(squared, axis=(1,2,3))
    self.grad_grad_B_inverse_scale_length_vs_varphi = np.sqrt(np.sqrt(norm_squared) / (4*B0))
    self.grad_grad_B_inverse_scale_length = np.max(self.grad_grad_B_inverse_scale_length_vs_varphi)

    if not two_ways:
        return

    # Build the whole tensor again using Rogerio's approach,
    # "20200424-01 Rogerio's GradGradB calculation.nb"
    # and verify the two calculations match.

    # Element 111
    grad_grad_B_alt[:,0,0,0] =(-2*B0*(-4*sign_G*sign_psi*iota_N0*X2c*Y1c*\
                                      Y1s + iota_N0*X1c*X1c*\
                                      Y1c*(Y1c*\
                                           (-Y20 + Y2c) + \
                                           Y1s*(Y2s - \
                                                2*sign_G*sign_psi*curvature)) + \
                                      X20*Y1c*Y1c*Y1s*\
                                      d_X1c_d_varphi - \
                                      X2c*Y1c*Y1c*Y1s*\
                                      d_X1c_d_varphi + \
                                      X20*Y1s*Y1s*Y1s*\
                                      d_X1c_d_varphi + \
                                      X2c*Y1s*Y1s*Y1s*\
                                      d_X1c_d_varphi + \
                                      sign_G*sign_psi*Y1c*Y20*\
                                      d_X1c_d_varphi - \
                                      sign_G*sign_psi*Y1c*Y2c*\
                                      d_X1c_d_varphi - \
                                      sign_G*sign_psi*Y1s*Y2s*\
                                      d_X1c_d_varphi - \
                                      2*X2s*(sign_G*sign_psi*iota_N0*Y1s*Y1s + \
                                             Y1c*Y1c*\
                                             (-(sign_G*sign_psi*iota_N0) + \
                                              iota_N0*X1c*Y1s) + \
                                             Y1c*Y1s*Y1s*\
                                             d_X1c_d_varphi) + \
                                      X1c*(iota_N0*X2c*Y1c*\
                                           (-Y1c*Y1c + Y1s*Y1s) + \
                                           iota_N0*X20*Y1c*\
                                           (Y1c*Y1c + Y1s*Y1s) - \
                                           sign_G*sign_psi*iota_N0*Y1s*Y20 - \
                                           sign_G*sign_psi*iota_N0*Y1s*Y2c + \
                                           sign_G*sign_psi*iota_N0*Y1c*Y2s - \
                                           Y1c*Y1s*Y20*\
                                           d_X1c_d_varphi + \
                                           Y1c*Y1s*Y2c*\
                                           d_X1c_d_varphi + \
                                           Y1s*Y1s*Y2s*\
                                           d_X1c_d_varphi - \
                                           2*sign_G*sign_psi*Y1s*Y1s*curvature*\
                                           d_X1c_d_varphi) - \
                                      sign_G*sign_psi*Y1c*Y1c*d_X20_d_varphi - \
                                      sign_G*sign_psi*Y1s*Y1s*d_X20_d_varphi + \
                                      sign_G*sign_psi*Y1c*Y1c*d_X2c_d_varphi - \
                                      sign_G*sign_psi*Y1s*Y1s*d_X2c_d_varphi + \
                                      2*sign_G*sign_psi*Y1c*Y1s*\
                                      d_X2s_d_varphi))/(lp*sign_psi)

    # Element 112
    grad_grad_B_alt[:,0,0,1] =(2*B0*(2*iota_N0*X2s*Y1c*Y1c*Y1c*\
                                     Y1s + 2*iota_N0*X2s*\
                                     Y1c*Y1s*Y1s*Y1s + \
                                     iota_N0*X1c*Y1c*Y1c*Y1c*\
                                     Y20 + iota_N0*X1c*Y1c*\
                                     Y1s*Y1s*Y20 - \
                                     iota_N0*X1c*Y1c*Y1c*Y1c*\
                                     Y2c + 6*sign_G*sign_psi*iota_N0*Y1c*\
                                     Y1s*Y2c - \
                                     iota_N0*X1c*Y1c*Y1s*Y1s*\
                                     Y2c - 3*sign_G*sign_psi*iota_N0*Y1c*Y1c*\
                                     Y2s - iota_N0*X1c*\
                                     Y1c*Y1c*Y1s*Y2s + \
                                     3*sign_G*sign_psi*iota_N0*Y1s*Y1s*Y2s - \
                                     iota_N0*X1c*Y1s*Y1s*Y1s*\
                                     Y2s + 2*sign_G*sign_psi*iota_N0*X1c*\
                                     Y1c*Y1c*Y1s*curvature + \
                                     2*sign_G*sign_psi*iota_N0*X1c*Y1s*Y1s*Y1s*\
                                     curvature - \
                                     2*lp*sign_G*sign_psi*X2s*Y1c*\
                                     Y1s*torsion + \
                                     2*lp*X1c*X2s*Y1c*\
                                     Y1s*Y1s*torsion - \
                                     lp*sign_G*sign_psi*X1c*Y1c*\
                                     Y20*torsion + \
                                     lp*X1c*X1c*Y1c*Y1s*\
                                     Y20*torsion + \
                                     lp*sign_G*sign_psi*X1c*Y1c*\
                                     Y2c*torsion - \
                                     lp*X1c*X1c*Y1c*Y1s*\
                                     Y2c*torsion + \
                                     lp*sign_G*sign_psi*X1c*Y1s*\
                                     Y2s*torsion - \
                                     lp*X1c*X1c*Y1s*Y1s*Y2s*\
                                     torsion + \
                                     2*lp*sign_G*sign_psi*X1c*X1c*Y1s*Y1s*\
                                     curvature*torsion + \
                                     2*X2s*Y1c*Y1s*Y1s*\
                                     d_Y1c_d_varphi - \
                                     sign_G*sign_psi*Y1c*Y20*\
                                     d_Y1c_d_varphi + \
                                     X1c*Y1c*Y1s*\
                                     Y20*d_Y1c_d_varphi + \
                                     sign_G*sign_psi*Y1c*Y2c*\
                                     d_Y1c_d_varphi - \
                                     X1c*Y1c*Y1s*\
                                     Y2c*d_Y1c_d_varphi + \
                                     sign_G*sign_psi*Y1s*Y2s*\
                                     d_Y1c_d_varphi - \
                                     X1c*Y1s*Y1s*Y2s*\
                                     d_Y1c_d_varphi + \
                                     2*sign_G*sign_psi*X1c*Y1s*Y1s*\
                                     curvature*d_Y1c_d_varphi - \
                                     2*X2s*Y1c*Y1c*Y1s*\
                                     d_Y1s_d_varphi - \
                                     X1c*Y1c*Y1c*Y20*\
                                     d_Y1s_d_varphi - \
                                     sign_G*sign_psi*Y1s*Y20*\
                                     d_Y1s_d_varphi + \
                                     X1c*Y1c*Y1c*Y2c*\
                                     d_Y1s_d_varphi - \
                                     sign_G*sign_psi*Y1s*Y2c*\
                                     d_Y1s_d_varphi + \
                                     sign_G*sign_psi*Y1c*Y2s*\
                                     d_Y1s_d_varphi + \
                                     X1c*Y1c*Y1s*\
                                     Y2s*d_Y1s_d_varphi - \
                                     2*sign_G*sign_psi*X1c*Y1c*Y1s*\
                                     curvature*d_Y1s_d_varphi + \
                                     X2c*(Y1c*Y1c - Y1s*Y1s)*\
                                     (iota_N0*Y1c*Y1c + iota_N0*Y1s*Y1s - \
                                      lp*sign_G*sign_psi*torsion + \
                                      Y1s*(lp*X1c*torsion + \
                                           d_Y1c_d_varphi) - \
                                      Y1c*d_Y1s_d_varphi) - \
                                     X20*(Y1c*Y1c + Y1s*Y1s)*\
                                     (iota_N0*Y1c*Y1c + iota_N0*Y1s*Y1s - \
                                      lp*sign_G*sign_psi*torsion + \
                                      Y1s*(lp*X1c*torsion + \
                                           d_Y1c_d_varphi) - \
                                      Y1c*d_Y1s_d_varphi) + \
                                     sign_G*sign_psi*Y1c*Y1c*d_Y20_d_varphi + \
                                     sign_G*sign_psi*Y1s*Y1s*d_Y20_d_varphi - \
                                     sign_G*sign_psi*Y1c*Y1c*d_Y2c_d_varphi + \
                                     sign_G*sign_psi*Y1s*Y1s*d_Y2c_d_varphi - \
                                     2*sign_G*sign_psi*Y1c*Y1s*\
                                     d_Y2s_d_varphi))/(lp*sign_psi)

    # Element 113
    grad_grad_B_alt[:,0,0,2] =(-2*(Y1c*Y1c*(G2*sign_psi + I2*sign_psi*iota - \
                                            2*lp*sign_G*sign_psi*B20 + \
                                            2*lp*sign_G*sign_psi*B2c + \
                                            2*B0*sign_G*sign_psi*iota_N0*Z2s + \
                                            B0*lp*sign_G*sign_psi*X20*curvature - \
                                            B0*lp*sign_G*sign_psi*X2c*curvature + \
                                            B0*lp*X1c*X20*Y1s*\
                                            curvature - \
                                            B0*lp*X1c*X2c*Y1s*\
                                            curvature - \
                                            B0*sign_G*sign_psi*d_Z20_d_varphi + \
                                            B0*sign_G*sign_psi*d_Z2c_d_varphi) + \
                                   Y1s*(B0*lp*X1c*\
                                        (X20 + X2c)*Y1s*Y1s*\
                                        curvature - \
                                        B0*lp*sign_G*sign_psi*X1c*Y2s*\
                                        curvature + \
                                        Y1s*(G2*sign_psi + I2*sign_psi*iota - \
                                             2*lp*sign_G*sign_psi*B20 - \
                                             2*lp*sign_G*sign_psi*B2c - \
                                             2*B0*sign_G*sign_psi*iota_N0*Z2s + \
                                             B0*lp*sign_G*sign_psi*X20*curvature + \
                                             B0*lp*sign_G*sign_psi*X2c*curvature + \
                                             B0*lp*X1c*X1c*Y2s*\
                                             curvature + \
                                             B0*lp*sign_G*sign_psi*X1c*X1c*\
                                             curvature*curvature - \
                                             B0*sign_G*sign_psi*d_Z20_d_varphi - \
                                             B0*sign_G*sign_psi*d_Z2c_d_varphi)) + \
                                   Y1c*(4*lp*sign_G*sign_psi*B2s*\
                                        Y1s - \
                                        B0*(2*lp*X1c*X2s*\
                                            Y1s*Y1s*curvature + \
                                            lp*sign_G*sign_psi*X1c*\
                                            (-Y20 + Y2c)*\
                                            curvature + \
                                            Y1s*\
                                            (4*sign_G*sign_psi*iota_N0*Z2c + \
                                             2*lp*sign_G*sign_psi*X2s*curvature + \
                                             lp*X1c*X1c*Y20*\
                                             curvature - \
                                             lp*X1c*X1c*Y2c*\
                                             curvature - \
                                             2*sign_G*sign_psi*d_Z2s_d_varphi)))))/\
                                             (lp*sign_psi)

    # Element 121
    grad_grad_B_alt[:,0,1,0] =(-2*B0*(iota_N0*X1c*X1c*X1c*\
                                      (Y1c*(Y20 - Y2c) + \
                                       Y1s*(-Y2s + \
                                            sign_G*sign_psi*curvature)) - \
                                      X1c*X1c*(iota_N0*X2c*\
                                               (-Y1c*Y1c + Y1s*Y1s) + \
                                               iota_N0*X20*\
                                               (Y1c*Y1c + Y1s*Y1s) + \
                                               Y1s*(-2*iota_N0*X2s*\
                                                    Y1c + \
                                                    lp*(Y1c*\
                                                        (-Y20 + Y2c) + \
                                                        Y1s*\
                                                        (Y2s - sign_G*sign_psi*curvature))*\
                                                    torsion)) + \
                                      sign_G*sign_psi*(X2s*Y1s*\
                                                       (-2*lp*Y1c*torsion + \
                                                        d_X1c_d_varphi) + \
                                                       X20*(lp*Y1c*Y1c*\
                                                            torsion + \
                                                            lp*Y1s*Y1s*torsion - \
                                                            Y1c*d_X1c_d_varphi) + \
                                                       X2c*(-(lp*Y1c*Y1c*\
                                                              torsion) + \
                                                            lp*Y1s*Y1s*torsion + \
                                                            Y1c*d_X1c_d_varphi)) + \
                                      X1c*(3*sign_G*sign_psi*iota_N0*X2c*\
                                           Y1s + \
                                           lp*X2c*Y1c*Y1c*Y1s*\
                                           torsion - \
                                           lp*X2c*Y1s*Y1s*Y1s*\
                                           torsion - \
                                           lp*sign_G*sign_psi*Y1c*Y20*\
                                           torsion + \
                                           lp*sign_G*sign_psi*Y1c*Y2c*\
                                           torsion + \
                                           lp*sign_G*sign_psi*Y1s*Y2s*\
                                           torsion - \
                                           X20*Y1s*\
                                           (-(sign_G*sign_psi*iota_N0) + \
                                            lp*Y1c*Y1c*torsion + \
                                            lp*Y1s*Y1s*torsion) + \
                                           X2s*Y1c*\
                                           (-3*sign_G*sign_psi*iota_N0 + \
                                            2*lp*Y1s*Y1s*torsion) + \
                                           sign_G*sign_psi*Y1c*d_X20_d_varphi - \
                                           sign_G*sign_psi*Y1c*d_X2c_d_varphi - \
                                           sign_G*sign_psi*Y1s*d_X2s_d_varphi)))/\
                                           (lp*sign_psi)

    # Element 122
    grad_grad_B_alt[:,0,1,1] =(2*B0*(-(X1c*X1c*\
                                       (Y1c*(Y20 - Y2c) + \
                                        Y1s*(-Y2s + \
                                             sign_G*sign_psi*curvature))*\
                                       (iota_N0*Y1c - d_Y1s_d_varphi)) \
                                     + X2s*(iota_N0*Y1c*Y1c*\
                                            (sign_G*sign_psi - 2*X1c*Y1s) - \
                                            sign_G*sign_psi*Y1s*\
                                            (iota_N0*Y1s + \
                                             d_Y1c_d_varphi) + \
                                            Y1c*(-(sign_G*sign_psi) + \
                                                 2*X1c*Y1s)*\
                                            d_Y1s_d_varphi) + \
                                     sign_G*sign_psi*(X20*\
                                                      (Y1c*d_Y1c_d_varphi + \
                                                       Y1s*d_Y1s_d_varphi) + \
                                                      X2c*(-(Y1c*\
                                                             (2*iota_N0*Y1s + \
                                                              d_Y1c_d_varphi)) + \
                                                           Y1s*d_Y1s_d_varphi)) + \
                                     X1c*(-(X2c*\
                                            (Y1c*Y1c - Y1s*Y1s)*\
                                            (iota_N0*Y1c - \
                                             d_Y1s_d_varphi)) + \
                                          X20*(Y1c*Y1c + Y1s*Y1s)*\
                                          (iota_N0*Y1c - d_Y1s_d_varphi) \
                                          + sign_G*sign_psi*(Y1c*\
                                                             (2*iota_N0*Y2s - \
                                                              d_Y20_d_varphi + \
                                                              d_Y2c_d_varphi) + \
                                                             Y1s*\
                                                             (-2*iota_N0*Y2c + \
                                                              d_Y2s_d_varphi)))))/(lp*sign_psi)

    # Element 123
    grad_grad_B_alt[:,0,1,2] =(2*X1c*(Y1c*\
                                      (G2 + I2*iota - 2*lp*sign_G*B20 + \
                                       2*lp*sign_G*B2c + \
                                       2*B0*sign_G*iota_N0*Z2s + \
                                       2*B0*lp*sign_G*X20*curvature - \
                                       2*B0*lp*sign_G*X2c*curvature - \
                                       B0*sign_G*d_Z20_d_varphi + \
                                       B0*sign_G*d_Z2c_d_varphi) + \
                                      sign_G*Y1s*(2*lp*B2s + \
                                                  B0*(-2*iota_N0*Z2c - \
                                                      2*lp*X2s*curvature + \
                                                      d_Z2s_d_varphi))))/(lp)

    # Element 131
    grad_grad_B_alt[:,0,2,0] =(B0*(-(lp*sign_G*sign_psi*iota_N0*Y1c*Y1c*\
                                     torsion) + \
                                   lp*iota_N0*X1c*X1c*X1c*Y1s*\
                                   torsion + \
                                   X1c*X1c*(lp*lp*Y1s*Y1s*\
                                            torsion*torsion + \
                                            iota_N0*Y1s*d_Y1c_d_varphi - \
                                            iota_N0*Y1c*d_Y1s_d_varphi) + \
                                   sign_G*sign_psi*Y1c*\
                                   (iota_N0*d_X1c_d_varphi + \
                                    2*lp*torsion*d_Y1s_d_varphi) + \
                                   X1c*Y1s*\
                                   (2*lp*lp*sign_G*sign_psi*curvature*curvature - \
                                    lp*lp*sign_G*sign_psi*torsion*torsion - \
                                    iota_N0*Y1c*d_X1c_d_varphi + \
                                    lp*torsion*\
                                    (Y1s*d_Y1c_d_varphi - \
                                     Y1c*d_Y1s_d_varphi)) - \
                                   Y1s*(Y1s*\
                                        (lp*sign_G*sign_psi*iota_N0*torsion + \
                                         d_X1c_d_varphi*d_X1c_d_varphi) + \
                                        sign_G*sign_psi*(2*lp*torsion*\
                                                         d_Y1c_d_varphi - \
                                                         d2_X1c_d_varphi2))))/(lp*lp*sign_G)

    # Element 132
    grad_grad_B_alt[:,0,2,1] =(B0*(-(iota_N0*Y1c*Y1c*Y1s*\
                                     d_X1c_d_varphi) + \
                                   lp*X1c*X1c*Y1s*torsion*\
                                   (iota_N0*Y1c - d_Y1s_d_varphi) + \
                                   X1c*(-(iota_N0*Y1c*Y1c*\
                                          d_Y1s_d_varphi) + \
                                        Y1c*(lp*sign_G*sign_psi*iota_N0*\
                                             torsion + \
                                             iota_N0*Y1s*\
                                             d_Y1c_d_varphi + \
                                             d_Y1s_d_varphi*d_Y1s_d_varphi) - \
                                        Y1s*(lp*Y1s*torsion*\
                                             d_X1c_d_varphi + \
                                             d_Y1c_d_varphi*\
                                             d_Y1s_d_varphi - \
                                             lp*sign_G*sign_psi*d_torsion_d_varphi)) + \
                                   Y1s*(-(iota_N0*Y1s*Y1s*\
                                          d_X1c_d_varphi) - \
                                        Y1s*d_X1c_d_varphi*\
                                        d_Y1c_d_varphi + \
                                        sign_G*sign_psi*(2*lp*torsion*\
                                                         d_X1c_d_varphi + \
                                                         iota_N0*d_Y1s_d_varphi + \
                                                         d2_Y1c_d_varphi2)) + \
                                   Y1c*(sign_G*sign_psi*iota_N0*\
                                        d_Y1c_d_varphi + \
                                        Y1s*d_X1c_d_varphi*\
                                        d_Y1s_d_varphi - \
                                        sign_G*sign_psi*d2_Y1s_d_varphi2)))/\
                                        (lp*lp*sign_G)

    # Element 133
    grad_grad_B_alt[:,0,2,2] =-((B0*(Y1s*curvature*\
                                     d_X1c_d_varphi + \
                                     X1c*(iota_N0*Y1c*\
                                          curvature - \
                                          Y1s*d_curvature_d_varphi)))/\
                                (lp*sign_psi))

    # Element 211
    grad_grad_B_alt[:,1,0,0] =(-2*B0*X1c*(2*sign_G*sign_psi*iota_N0*X2c*\
                                          Y1s + iota_N0*X1c*X1c*\
                                          (Y1c*(Y20 - Y2c) + \
                                           sign_G*sign_psi*Y1s*curvature) - \
                                          X20*Y1c*Y1s*\
                                          d_X1c_d_varphi + \
                                          X2c*Y1c*Y1s*\
                                          d_X1c_d_varphi - \
                                          sign_G*sign_psi*Y20*d_X1c_d_varphi + \
                                          sign_G*sign_psi*Y2c*d_X1c_d_varphi + \
                                          X2s*(Y1c*\
                                               (-2*sign_G*sign_psi*iota_N0 + \
                                                iota_N0*X1c*Y1s) + \
                                               Y1s*Y1s*d_X1c_d_varphi) + \
                                          X1c*(-(iota_N0*X20*\
                                                 Y1c*Y1c) + \
                                               iota_N0*X2c*Y1c*Y1c - \
                                               sign_G*sign_psi*iota_N0*Y2s + \
                                               lp*sign_G*sign_psi*Y1s*Y1s*curvature*\
                                               torsion + \
                                               Y1s*Y20*\
                                               d_X1c_d_varphi - \
                                               Y1s*Y2c*\
                                               d_X1c_d_varphi) + \
                                          sign_G*sign_psi*Y1c*d_X20_d_varphi - \
                                          sign_G*sign_psi*Y1c*d_X2c_d_varphi - \
                                          sign_G*sign_psi*Y1s*d_X2s_d_varphi))/\
                                          (lp*sign_psi)

    # Element 212
    grad_grad_B_alt[:,1,0,1] =(-2*B0*X1c*(iota_N0*X2s*\
                                          Y1c*Y1c*Y1s + \
                                          iota_N0*X2s*Y1s*Y1s*Y1s + \
                                          iota_N0*X1c*Y1c*Y1c*\
                                          Y20 - sign_G*sign_psi*iota_N0*Y1s*\
                                          Y20 + iota_N0*X1c*\
                                          Y1s*Y1s*Y20 - \
                                          iota_N0*X1c*Y1c*Y1c*\
                                          Y2c + 3*sign_G*sign_psi*iota_N0*Y1s*\
                                          Y2c - iota_N0*X1c*\
                                          Y1s*Y1s*Y2c - \
                                          3*sign_G*sign_psi*iota_N0*Y1c*Y2s + \
                                          sign_G*sign_psi*iota_N0*X1c*Y1c*\
                                          Y1s*curvature - \
                                          lp*sign_G*sign_psi*X2s*Y1s*\
                                          torsion + \
                                          lp*X1c*X2s*Y1s*Y1s*\
                                          torsion - \
                                          lp*sign_G*sign_psi*X1c*Y20*\
                                          torsion + \
                                          lp*X1c*X1c*Y1s*Y20*\
                                          torsion + \
                                          lp*sign_G*sign_psi*X1c*Y2c*\
                                          torsion - \
                                          lp*X1c*X1c*Y1s*Y2c*\
                                          torsion + \
                                          X2s*Y1s*Y1s*\
                                          d_Y1c_d_varphi - \
                                          sign_G*sign_psi*Y20*d_Y1c_d_varphi + \
                                          X1c*Y1s*Y20*\
                                          d_Y1c_d_varphi + \
                                          sign_G*sign_psi*Y2c*d_Y1c_d_varphi - \
                                          X1c*Y1s*Y2c*\
                                          d_Y1c_d_varphi - \
                                          X2s*Y1c*Y1s*\
                                          d_Y1s_d_varphi - \
                                          X1c*Y1c*Y20*\
                                          d_Y1s_d_varphi + \
                                          X1c*Y1c*Y2c*\
                                          d_Y1s_d_varphi + \
                                          sign_G*sign_psi*Y2s*d_Y1s_d_varphi - \
                                          sign_G*sign_psi*X1c*Y1s*\
                                          curvature*d_Y1s_d_varphi - \
                                          X20*Y1c*\
                                          (iota_N0*Y1c*Y1c + iota_N0*Y1s*Y1s - \
                                           lp*sign_G*sign_psi*torsion + \
                                           Y1s*(lp*X1c*torsion + \
                                                d_Y1c_d_varphi) - \
                                           Y1c*d_Y1s_d_varphi) + \
                                          X2c*Y1c*\
                                          (iota_N0*Y1c*Y1c + iota_N0*Y1s*Y1s - \
                                           lp*sign_G*sign_psi*torsion + \
                                           Y1s*(lp*X1c*torsion + \
                                                d_Y1c_d_varphi) - \
                                           Y1c*d_Y1s_d_varphi) + \
                                          sign_G*sign_psi*Y1c*d_Y20_d_varphi - \
                                          sign_G*sign_psi*Y1c*d_Y2c_d_varphi - \
                                          sign_G*sign_psi*Y1s*d_Y2s_d_varphi))/\
                                          (lp*sign_psi)

    # Element 213
    grad_grad_B_alt[:,1,0,2] =(2*X1c*(2*lp*sign_G*sign_psi*B2s*\
                                      Y1s + Y1c*\
                                      (G2*sign_psi + I2*sign_psi*iota - \
                                       2*lp*sign_G*sign_psi*B20 + \
                                       2*lp*sign_G*sign_psi*B2c + \
                                       2*B0*sign_G*sign_psi*iota_N0*Z2s + \
                                       B0*lp*sign_G*sign_psi*X20*curvature - \
                                       B0*lp*sign_G*sign_psi*X2c*curvature + \
                                       B0*lp*X1c*X20*Y1s*\
                                       curvature - \
                                       B0*lp*X1c*X2c*Y1s*\
                                       curvature - \
                                       B0*sign_G*sign_psi*d_Z20_d_varphi + \
                                       B0*sign_G*sign_psi*d_Z2c_d_varphi) - \
                                      B0*(lp*X1c*X2s*Y1s*Y1s*\
                                          curvature + \
                                          lp*sign_G*sign_psi*X1c*\
                                          (-Y20 + Y2c)*\
                                          curvature + \
                                          Y1s*(2*sign_G*sign_psi*iota_N0*Z2c + \
                                               lp*sign_G*sign_psi*X2s*curvature + \
                                               lp*X1c*X1c*Y20*\
                                               curvature - \
                                               lp*X1c*X1c*Y2c*\
                                               curvature - \
                                               sign_G*sign_psi*d_Z2s_d_varphi))))/\
                                               (lp*sign_psi)

    # Element 221
    grad_grad_B_alt[:,1,1,0] =(2*B0*X1c*(iota_N0*X1c*X1c*X1c*\
                                         (Y20 - Y2c) + \
                                         X1c*X1c*(-(iota_N0*X20*\
                                                    Y1c) + \
                                                  iota_N0*X2c*Y1c + \
                                                  Y1s*(iota_N0*X2s + \
                                                       lp*(Y20 - Y2c)*\
                                                       torsion)) - \
                                         sign_G*sign_psi*(lp*X2s*Y1s*\
                                                          torsion + \
                                                          X2c*(lp*Y1c*\
                                                               torsion - d_X1c_d_varphi) + \
                                                          X20*(-(lp*Y1c*\
                                                                 torsion) + d_X1c_d_varphi)) \
                                         + X1c*(-(lp*X20*Y1c*\
                                                  Y1s*torsion) + \
                                                lp*X2c*Y1c*Y1s*\
                                                torsion - \
                                                lp*sign_G*sign_psi*Y20*torsion + \
                                                lp*sign_G*sign_psi*Y2c*torsion + \
                                                X2s*(-3*sign_G*sign_psi*iota_N0 + \
                                                     lp*Y1s*Y1s*torsion) + \
                                                sign_G*sign_psi*d_X20_d_varphi - \
                                                sign_G*sign_psi*d_X2c_d_varphi)))/\
                                                (lp*sign_psi)

    # Element 222
    grad_grad_B_alt[:,1,1,1] =(-2*B0*X1c*(sign_G*sign_psi*\
                                          (X20 - X2c)*\
                                          (iota_N0*Y1s + d_Y1c_d_varphi) + \
                                          X2s*(sign_G*sign_psi - \
                                               X1c*Y1s)*\
                                          (iota_N0*Y1c - d_Y1s_d_varphi) - \
                                          X1c*X1c*(Y20 - Y2c)*\
                                          (iota_N0*Y1c - d_Y1s_d_varphi) + \
                                          X1c*(X20*Y1c*\
                                               (iota_N0*Y1c - d_Y1s_d_varphi) \
                                               + X2c*Y1c*\
                                               (-(iota_N0*Y1c) + \
                                                d_Y1s_d_varphi) + \
                                               sign_G*sign_psi*(2*iota_N0*Y2s - \
                                                                d_Y20_d_varphi + \
                                                                d_Y2c_d_varphi))))/(lp*sign_psi)

    # Element 223
    grad_grad_B_alt[:,1,1,2] =(-2*X1c*X1c*(G2 + I2*iota - 2*lp*sign_G*B20 + \
                                           2*lp*sign_G*B2c + 2*B0*sign_G*iota_N0*Z2s + \
                                           2*B0*lp*sign_G*X20*curvature - \
                                           2*B0*lp*sign_G*X2c*curvature - \
                                           B0*sign_G*d_Z20_d_varphi + \
                                           B0*sign_G*d_Z2c_d_varphi))/(lp)

    # Element 231
    grad_grad_B_alt[:,1,2,0] =(B0*X1c*(lp*iota_N0*Y1c*\
                                       (sign_G*sign_psi + X1c*Y1s)*\
                                       torsion + \
                                       (-(sign_G*sign_psi*iota_N0) + \
                                        lp*Y1s*Y1s*torsion)*\
                                       d_X1c_d_varphi + \
                                       iota_N0*X1c*X1c*d_Y1s_d_varphi - \
                                       2*lp*sign_G*sign_psi*torsion*\
                                       d_Y1s_d_varphi + \
                                       lp*X1c*Y1s*torsion*\
                                       d_Y1s_d_varphi - \
                                       lp*sign_G*sign_psi*Y1s*d_torsion_d_varphi))/\
                                       (lp*lp*sign_G)

    # Element 232
    grad_grad_B_alt[:,1,2,1] =(B0*X1c*(lp*iota_N0*Y1c*Y1c*\
                                       Y1s*torsion + \
                                       lp*iota_N0*Y1s*Y1s*Y1s*torsion - \
                                       lp*lp*sign_G*sign_psi*Y1s*torsion*torsion - \
                                       sign_G*sign_psi*iota_N0*d_Y1c_d_varphi + \
                                       lp*Y1s*Y1s*torsion*\
                                       d_Y1c_d_varphi - \
                                       lp*Y1c*Y1s*torsion*\
                                       d_Y1s_d_varphi + \
                                       X1c*(-(lp*sign_G*sign_psi*iota_N0*\
                                              torsion) + \
                                            lp*lp*Y1s*Y1s*torsion*torsion + \
                                            (iota_N0*Y1c - d_Y1s_d_varphi)*\
                                            d_Y1s_d_varphi) + \
                                       sign_G*sign_psi*d2_Y1s_d_varphi2))/(lp*lp*sign_G)

    # Element 233
    grad_grad_B_alt[:,1,2,2] =(B0*X1c*curvature*\
                               (iota_N0*X1c + \
                                2*lp*Y1s*torsion))/(lp*sign_psi)

    # Element 311
    grad_grad_B_alt[:,2,0,0] =(B0*(2*lp*lp*sign_G*sign_psi*curvature*curvature + \
                                   lp*iota_N0*X1c*X1c*torsion - \
                                   lp*iota_N0*Y1c*Y1c*torsion - \
                                   lp*iota_N0*Y1s*Y1s*torsion + \
                                   iota_N0*Y1c*d_X1c_d_varphi + \
                                   iota_N0*X1c*d_Y1c_d_varphi - \
                                   lp*Y1s*torsion*\
                                   d_Y1c_d_varphi + \
                                   lp*Y1c*torsion*\
                                   d_Y1s_d_varphi + \
                                   d_X1c_d_varphi*d_Y1s_d_varphi + \
                                   Y1s*d2_X1c_d_varphi2))/\
                                   (lp*lp*sign_psi)

    # Element 312
    grad_grad_B_alt[:,2,0,1] =(B0*(lp*X1c*(2*iota_N0*Y1c*\
                                           torsion + \
                                           Y1s*d_torsion_d_varphi) + \
                                   Y1s*(2*lp*torsion*\
                                        d_X1c_d_varphi + \
                                        2*iota_N0*d_Y1s_d_varphi + \
                                        d2_Y1c_d_varphi2) + \
                                   Y1c*(2*iota_N0*d_Y1c_d_varphi - \
                                        d2_Y1s_d_varphi2)))/(lp*lp*sign_psi)

    # Element 313
    grad_grad_B_alt[:,2,0,2] =(B0*(-(iota_N0*X1c*Y1c*\
                                     curvature) - \
                                   Y1s*curvature*\
                                   d_X1c_d_varphi + \
                                   sign_G*sign_psi*d_curvature_d_varphi))/(lp*sign_psi)

    # Element 321
    grad_grad_B_alt[:,2,1,0] =(B0*X1c*(2*lp*iota_N0*Y1c*\
                                       torsion - \
                                       2*iota_N0*d_X1c_d_varphi - \
                                       lp*(2*torsion*d_Y1s_d_varphi + \
                                           Y1s*d_torsion_d_varphi)))/\
                                           (lp*lp*sign_psi)

    # Element 322
    grad_grad_B_alt[:,2,1,1] =-((B0*(iota_N0*Y1c*d_X1c_d_varphi + \
                                     iota_N0*X1c*d_Y1c_d_varphi - \
                                     d_X1c_d_varphi*\
                                     d_Y1s_d_varphi + \
                                     lp*torsion*\
                                     (iota_N0*X1c*X1c - \
                                      iota_N0*Y1c*Y1c - \
                                      Y1s*(iota_N0*Y1s + \
                                           d_Y1c_d_varphi) + \
                                      Y1c*d_Y1s_d_varphi) - \
                                     X1c*d2_Y1s_d_varphi2))/\
                                (lp*lp*sign_psi))

    # Element 323
    grad_grad_B_alt[:,2,1,2] =(B0*curvature*(iota_N0*X1c*X1c + \
                                             lp*sign_G*sign_psi*torsion + \
                                             lp*X1c*Y1s*torsion))/\
                                             (lp*sign_psi)

    # Element 331
    grad_grad_B_alt[:,2,2,0] =(B0*(-(iota_N0*X1c*Y1c*\
                                     curvature) - \
                                   Y1s*curvature*\
                                   d_X1c_d_varphi + \
                                   sign_G*sign_psi*d_curvature_d_varphi))/(lp*sign_psi)

    # Element 332
    grad_grad_B_alt[:,2,2,1] =-((B0*curvature*(iota_N0*Y1c*Y1c + \
                                               iota_N0*Y1s*Y1s - \
                                               lp*sign_G*sign_psi*torsion + \
                                               Y1s*(lp*X1c*torsion + \
                                                    d_Y1c_d_varphi) - \
                                               Y1c*d_Y1s_d_varphi))/\
                                (lp*sign_psi))

    # Element 333
    grad_grad_B_alt[:,2,2,2] =(-2*B0*curvature*curvature)/sign_G

    self.grad_grad_B_alt = grad_grad_B_alt
