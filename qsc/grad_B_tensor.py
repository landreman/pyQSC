#!/usr/bin/env python3

"""
Functions for computing the grad B tensor and grad grad B tensor.
"""

import logging
import numpy as np
from .util import Struct, fourier_minimum

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_grad_B_tensor(self):
    """
    Compute the components of the grad B tensor, and the scale
    length L grad B associated with the Frobenius norm of this
    tensor.
    The formula for the grad B tensor is eq (3.12) of
    Landreman (2021): Figures of merit for stellarators near the magnetic axis, JPP

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
    if hasattr(s.B0, "__len__"): # if B0 is an array (in quasisymmetry B0 is a scalar)
        tensor.tt = s.sG * np.matmul(s.d_d_varphi, s.B0) / s.d_l_d_varphi
    else:
        tensor.tt = 0

    self.grad_B_tensor = tensor
    
    t = s.tangent_cylindrical.transpose()
    n = s.normal_cylindrical.transpose()
    b = s.binormal_cylindrical.transpose()
    self.grad_B_tensor_cylindrical = np.array([[
                              tensor.nn * n[i] * n[j] \
                            + tensor.bn * b[i] * n[j] + tensor.nb * n[i] * b[j] \
                            + tensor.bb * b[i] * b[j] \
                            + tensor.tn * t[i] * n[j] + tensor.nt * n[i] * t[j] \
                            + tensor.tt * t[i] * t[j]
                        for i in range(3)] for j in range(3)])

    self.grad_B_colon_grad_B = tensor.tn * tensor.tn + tensor.nt * tensor.nt \
        + tensor.bb * tensor.bb + tensor.nn * tensor.nn \
        + tensor.nb * tensor.nb + tensor.bn * tensor.bn \
        + tensor.tt * tensor.tt

    self.L_grad_B = s.B0 * np.sqrt(2 / self.grad_B_colon_grad_B)
    self.inv_L_grad_B = 1.0 / self.L_grad_B
    self.min_L_grad_B = fourier_minimum(self.L_grad_B)
    
def calculate_grad_grad_B_tensor(self, two_ways=False):
    """
    Compute the components of the grad grad B tensor, and the scale
    length L grad grad B associated with the Frobenius norm of this
    tensor.
    self should be an instance of Qsc with X1c, Y1s etc populated.
    The grad grad B tensor in discussed around eq (3.13)
    Landreman (2021): Figures of merit for stellarators near the magnetic axis, JPP
    although an explicit formula is not given there.

    If ``two_ways`` is ``True``, an independent calculation of
    the tensor is also computed, to confirm the answer is the same.
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
    self.L_grad_grad_B = 1 / self.grad_grad_B_inverse_scale_length_vs_varphi
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


def Bfield_cylindrical(self, r=0, theta=0):
    '''
    Function to calculate the magnetic field vector B=(B_R,B_phi,B_Z) at
    every point along the axis (hence with nphi points) where R, phi and Z
    are the standard cylindrical coordinates for a given
    near-axis radius r and a Boozer poloidal angle vartheta (not theta).
    The formulae implemented here are eq (3.5) and (3.6) of
    Landreman (2021): Figures of merit for stellarators near the magnetic axis, JPP

    Args:
      r: the near-axis radius
      theta: the Boozer poloidal angle vartheta (= theta-N*phi)
    '''
    
    # Define auxiliary variables
    t = self.tangent_cylindrical.transpose()
    n = self.normal_cylindrical.transpose()
    b = self.binormal_cylindrical.transpose()
    B0 = self.B0
    sG = self.sG
    G0 = self.G0
    X1c = self.X1c
    X1s = self.X1s
    Y1c = self.Y1c
    Y1s = self.Y1s
    d_l_d_varphi = self.d_l_d_varphi
    curvature = self.curvature
    torsion = self.torsion
    iotaN = self.iotaN
    d_X1c_d_varphi = self.d_X1c_d_varphi
    d_X1s_d_varphi = self.d_X1s_d_varphi
    d_Y1s_d_varphi = self.d_Y1s_d_varphi
    d_Y1c_d_varphi = self.d_Y1c_d_varphi

    B0_vector = sG * B0 * t

    if r == 0:
        return B0_vector
    else:
        factor = B0 * B0 / G0
        B1_vector_t = factor * (X1c * np.cos(theta) + X1s * np.sin(theta)) * d_l_d_varphi * curvature
        B1_vector_n = factor * (np.cos(theta) * (d_X1c_d_varphi - Y1c * d_l_d_varphi * torsion + iotaN * X1s) \
                                + np.sin(theta) * (d_X1s_d_varphi - Y1s * d_l_d_varphi * torsion - iotaN * X1c))
        B1_vector_b = factor * (np.cos(theta) * (d_Y1c_d_varphi + X1c * d_l_d_varphi * torsion + iotaN * Y1s) \
                                + np.sin(theta) * (d_Y1s_d_varphi + X1s * d_l_d_varphi * torsion - iotaN * Y1c))

        B1_vector = B1_vector_t * t + B1_vector_n * n + B1_vector_b * b
        B_vector_cylindrical = B0_vector + r * B1_vector

        return B_vector_cylindrical

def Bfield_cartesian(self, r=0, theta=0):
    '''
    Function to calculate the magnetic field vector B=(B_x,B_y,B_z) at
    every point along the axis (hence with nphi points) where x, y and z
    are the standard cartesian coordinates for a given
    near-axis radius r and a Boozer poloidal angle vartheta (not theta).

    Args:
      r: the near-axis radius
      theta: the Boozer poloidal angle vartheta (= theta-N*phi)
    '''
    B_vector_cylindrical = self.Bfield_cylindrical(r,theta)
    phi = self.phi

    B_x = np.cos(phi) * B_vector_cylindrical[0] - np.sin(phi) * B_vector_cylindrical[1]
    B_y = np.sin(phi) * B_vector_cylindrical[0] + np.cos(phi) * B_vector_cylindrical[1]
    B_z = B_vector_cylindrical[2]

    B_vector_cartesian = np.array([B_x, B_y, B_z])

    return B_vector_cartesian

def grad_B_tensor_cartesian(self):
    '''
    Function to calculate the gradient of the magnetic field vector B=(B_x,B_y,B_z)
    at every point along the axis (hence with nphi points) where x, y and z
    are the standard cartesian coordinates.
    '''

    B0, B1, B2 = self.Bfield_cylindrical()
    nablaB = self.grad_B_tensor_cylindrical
    cosphi = np.cos(self.phi)
    sinphi = np.sin(self.phi)
    R0 = self.R0

    grad_B_vector_cartesian = np.array([
[cosphi**2*nablaB[0, 0] - cosphi*sinphi*(nablaB[0, 1] + nablaB[1, 0]) + 
   sinphi**2*nablaB[1, 1], cosphi**2*nablaB[0, 1] - sinphi**2*nablaB[1, 0] + 
   cosphi*sinphi*(nablaB[0, 0] - nablaB[1, 1]), cosphi*nablaB[0, 2] - 
   sinphi*nablaB[1, 2]], [-(sinphi**2*nablaB[0, 1]) + cosphi**2*nablaB[1, 0] + 
   cosphi*sinphi*(nablaB[0, 0] - nablaB[1, 1]), sinphi**2*nablaB[0, 0] + 
   cosphi*sinphi*(nablaB[0, 1] + nablaB[1, 0]) + cosphi**2*nablaB[1, 1], 
  sinphi*nablaB[0, 2] + cosphi*nablaB[1, 2]], 
 [cosphi*nablaB[2, 0] - sinphi*nablaB[2, 1], sinphi*nablaB[2, 0] + cosphi*nablaB[2, 1], 
  nablaB[2, 2]]
    ])

    return grad_B_vector_cartesian

def grad_grad_B_tensor_cylindrical(self):
    '''
    Function to calculate the gradient of of the gradient the magnetic field
    vector B=(B_R,B_phi,B_Z) at every point along the axis (hence with nphi points)
    where R, phi and Z are the standard cylindrical coordinates.
    '''
    return np.transpose(self.grad_grad_B,(1,2,3,0))

def grad_grad_B_tensor_cartesian(self):
    '''
    Function to calculate the gradient of of the gradient the magnetic field
    vector B=(B_x,B_y,B_z) at every point along the axis (hence with nphi points)
    where x, y and z are the standard cartesian coordinates.
    '''
    nablanablaB = self.grad_grad_B_tensor_cylindrical()
    cosphi = np.cos(self.phi)
    sinphi = np.sin(self.phi)

    grad_grad_B_vector_cartesian = np.array([[
[cosphi**3*nablanablaB[0, 0, 0] - cosphi**2*sinphi*(nablanablaB[0, 0, 1] + 
      nablanablaB[0, 1, 0] + nablanablaB[1, 0, 0]) + 
    cosphi*sinphi**2*(nablanablaB[0, 1, 1] + nablanablaB[1, 0, 1] + 
      nablanablaB[1, 1, 0]) - sinphi**3*nablanablaB[1, 1, 1], 
   cosphi**3*nablanablaB[0, 0, 1] + cosphi**2*sinphi*(nablanablaB[0, 0, 0] - 
      nablanablaB[0, 1, 1] - nablanablaB[1, 0, 1]) + sinphi**3*nablanablaB[1, 1, 0] - 
    cosphi*sinphi**2*(nablanablaB[0, 1, 0] + nablanablaB[1, 0, 0] - 
      nablanablaB[1, 1, 1]), cosphi**2*nablanablaB[0, 0, 2] - 
    cosphi*sinphi*(nablanablaB[0, 1, 2] + nablanablaB[1, 0, 2]) + 
    sinphi**2*nablanablaB[1, 1, 2]], [cosphi**3*nablanablaB[0, 1, 0] + 
    sinphi**3*nablanablaB[1, 0, 1] + cosphi**2*sinphi*(nablanablaB[0, 0, 0] - 
      nablanablaB[0, 1, 1] - nablanablaB[1, 1, 0]) - 
    cosphi*sinphi**2*(nablanablaB[0, 0, 1] + nablanablaB[1, 0, 0] - 
      nablanablaB[1, 1, 1]), cosphi**3*nablanablaB[0, 1, 1] - 
    sinphi**3*nablanablaB[1, 0, 0] + cosphi*sinphi**2*(nablanablaB[0, 0, 0] - 
      nablanablaB[1, 0, 1] - nablanablaB[1, 1, 0]) + 
    cosphi**2*sinphi*(nablanablaB[0, 0, 1] + nablanablaB[0, 1, 0] - 
      nablanablaB[1, 1, 1]), cosphi**2*nablanablaB[0, 1, 2] - 
    sinphi**2*nablanablaB[1, 0, 2] + cosphi*sinphi*(nablanablaB[0, 0, 2] - 
      nablanablaB[1, 1, 2])], [cosphi**2*nablanablaB[0, 2, 0] - 
    cosphi*sinphi*(nablanablaB[0, 2, 1] + nablanablaB[1, 2, 0]) + 
    sinphi**2*nablanablaB[1, 2, 1], cosphi**2*nablanablaB[0, 2, 1] - 
    sinphi**2*nablanablaB[1, 2, 0] + cosphi*sinphi*(nablanablaB[0, 2, 0] - 
      nablanablaB[1, 2, 1]), cosphi*nablanablaB[0, 2, 2] - 
    sinphi*nablanablaB[1, 2, 2]]], 
 [[sinphi**3*nablanablaB[0, 1, 1] + cosphi**3*nablanablaB[1, 0, 0] + 
    cosphi**2*sinphi*(nablanablaB[0, 0, 0] - nablanablaB[1, 0, 1] - 
      nablanablaB[1, 1, 0]) - cosphi*sinphi**2*(nablanablaB[0, 0, 1] + 
      nablanablaB[0, 1, 0] - nablanablaB[1, 1, 1]), -(sinphi**3*nablanablaB[0, 1, 0]) + 
    cosphi**3*nablanablaB[1, 0, 1] + cosphi*sinphi**2*(nablanablaB[0, 0, 0] - 
      nablanablaB[0, 1, 1] - nablanablaB[1, 1, 0]) + 
    cosphi**2*sinphi*(nablanablaB[0, 0, 1] + nablanablaB[1, 0, 0] - 
      nablanablaB[1, 1, 1]), -(sinphi**2*nablanablaB[0, 1, 2]) + 
    cosphi**2*nablanablaB[1, 0, 2] + cosphi*sinphi*(nablanablaB[0, 0, 2] - 
      nablanablaB[1, 1, 2])], [-(sinphi**3*nablanablaB[0, 0, 1]) + 
    cosphi*sinphi**2*(nablanablaB[0, 0, 0] - nablanablaB[0, 1, 1] - 
      nablanablaB[1, 0, 1]) + cosphi**3*nablanablaB[1, 1, 0] + 
    cosphi**2*sinphi*(nablanablaB[0, 1, 0] + nablanablaB[1, 0, 0] - 
      nablanablaB[1, 1, 1]), sinphi**3*nablanablaB[0, 0, 0] + 
    cosphi*sinphi**2*(nablanablaB[0, 0, 1] + nablanablaB[0, 1, 0] + 
      nablanablaB[1, 0, 0]) + cosphi**2*sinphi*(nablanablaB[0, 1, 1] + 
      nablanablaB[1, 0, 1] + nablanablaB[1, 1, 0]) + cosphi**3*nablanablaB[1, 1, 1], 
   sinphi**2*nablanablaB[0, 0, 2] + cosphi*sinphi*(nablanablaB[0, 1, 2] + 
      nablanablaB[1, 0, 2]) + cosphi**2*nablanablaB[1, 1, 2]], 
  [-(sinphi**2*nablanablaB[0, 2, 1]) + cosphi**2*nablanablaB[1, 2, 0] + 
    cosphi*sinphi*(nablanablaB[0, 2, 0] - nablanablaB[1, 2, 1]), 
   sinphi**2*nablanablaB[0, 2, 0] + cosphi*sinphi*(nablanablaB[0, 2, 1] + 
      nablanablaB[1, 2, 0]) + cosphi**2*nablanablaB[1, 2, 1], 
   sinphi*nablanablaB[0, 2, 2] + cosphi*nablanablaB[1, 2, 2]]], 
 [[cosphi**2*nablanablaB[2, 0, 0] - cosphi*sinphi*(nablanablaB[2, 0, 1] + 
      nablanablaB[2, 1, 0]) + sinphi**2*nablanablaB[2, 1, 1], 
   cosphi**2*nablanablaB[2, 0, 1] - sinphi**2*nablanablaB[2, 1, 0] + 
    cosphi*sinphi*(nablanablaB[2, 0, 0] - nablanablaB[2, 1, 1]), 
   cosphi*nablanablaB[2, 0, 2] - sinphi*nablanablaB[2, 1, 2]], 
  [-(sinphi**2*nablanablaB[2, 0, 1]) + cosphi**2*nablanablaB[2, 1, 0] + 
    cosphi*sinphi*(nablanablaB[2, 0, 0] - nablanablaB[2, 1, 1]), 
   sinphi**2*nablanablaB[2, 0, 0] + cosphi*sinphi*(nablanablaB[2, 0, 1] + 
      nablanablaB[2, 1, 0]) + cosphi**2*nablanablaB[2, 1, 1], 
   sinphi*nablanablaB[2, 0, 2] + cosphi*nablanablaB[2, 1, 2]], 
  [cosphi*nablanablaB[2, 2, 0] - sinphi*nablanablaB[2, 2, 1], 
   sinphi*nablanablaB[2, 2, 0] + cosphi*nablanablaB[2, 2, 1], nablanablaB[2, 2, 2]]
      ]])

    return grad_grad_B_vector_cartesian
