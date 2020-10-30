"""
This module contains the routine for computing the terms in
Mercier's criterion.
"""

import numpy as np
from .util import mu0

def mercier(self):
    """
    Compute the terms in Mercier's criterion.
    """

    # See Overleaf note "Mercier criterion near the magnetic axis- detailed notes".
    # See also "20200604-02 Checking sign in Mercier DGeod near axis.docx"

    # Shorthand:
    d_l_d_phi = self.d_l_d_phi
    B0 = self.B0
    G0 = self.G0
    p2 = self.p2
    etabar = self.etabar
    curvature = self.curvature
    sigma = self.sigma
    iotaN = self.iotaN
    iota = self.iota
    pi = np.pi

    #integrand = d_l_d_phi * (Y1c * Y1c + X1c * (X1c + Y1s)) / (Y1c * Y1c + (X1c + Y1s) * (X1c + Y1s))
    integrand = d_l_d_phi * (etabar*etabar*etabar*etabar + curvature*curvature*curvature*curvature*sigma*sigma + etabar*etabar*curvature*curvature) \
        / (etabar*etabar*etabar*etabar + curvature*curvature*curvature*curvature*(1+sigma*sigma) + 2*etabar*etabar*curvature*curvature)

    integral = np.sum(integrand) * self.d_phi * self.nfp * 2 * pi / self.axis_length

    #DGeod_times_r2 = -(2 * sG * spsi * mu0 * mu0 * p2 * p2 * G0 * G0 * G0 * G0 * etabar * etabar &
    self.DGeod_times_r2 = -(2 * mu0 * mu0 * p2 * p2 * G0 * G0 * G0 * G0 * etabar * etabar \
                       / (pi * pi * pi * B0 * B0 * B0 * B0 * B0 * B0 * B0 * B0 * B0 * B0 * iotaN * iotaN)) \
                       * integral

    self.d2_volume_d_psi2 = 4*pi*pi*abs(G0)/(B0*B0*B0)*(3*etabar*etabar - 4*self.B20_mean/B0 + 2 * (self.G2 + iota * self.I2)/G0)

    self.DWell_times_r2 = (mu0 * p2 * abs(G0) / (8 * pi * pi * pi * pi * B0 * B0 * B0)) * \
        (self.d2_volume_d_psi2 - 8 * pi * pi * mu0 * p2 * abs(G0) / (B0 * B0 * B0 * B0 * B0))

    self.DMerc_times_r2 = self.DWell_times_r2 + self.DGeod_times_r2
