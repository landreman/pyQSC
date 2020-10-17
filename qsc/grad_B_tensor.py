#!/usr/bin/env python3

"""
Functions for computing the grad B tensor and grad grad B tensor.
"""

import numpy as np
import logging

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class grad_B_tensor():
    """
    Class representing the grad B tensor.
    """
    def __init__(self, s):
        """
        Compute the components of the grad B tensor, and the scale
        length L_{\nabla B} associated with the Frobenius norm of this
        tensor.

        s should be an instance of Qsc with X1c, Y1s etc populated.
        """
        factor = s.spsi * s.B0 / s.d_l_d_varphi
        self.tn = s.sG * s.B0 * s.curvature
        self.nt = self.tn
        self.bb = factor * (s.X1c * s.d_Y1s_d_varphi - s.iotaN * s.X1c * s.Y1c)
        self.nn = factor * (s.d_X1c_d_varphi * s.Y1s + s.iotaN * s.X1c * s.Y1c)
        self.bn = factor * (-s.sG * s.spsi * s.d_l_d_varphi * s.torsion \
                            - s.iotaN * s.X1c * s.X1c)
        self.nb = factor * (s.d_Y1c_d_varphi * s.Y1s - s.d_Y1s_d_varphi * s.Y1c \
                            + s.sG * s.spsi * s.d_l_d_varphi * s.torsion \
                            + s.iotaN * (s.Y1s * s.Y1s + s.Y1c * s.Y1c))
        
        self.grad_B_colon_grad_B = self.tn * self.tn + self.nt * self.nt \
            + self.bb * self.bb + self.nn * self.nn \
            + self.nb * self.nb + self.bn * self.bn

        self.L_grad_B = s.B0 * np.sqrt(2 / self.grad_B_colon_grad_B)
