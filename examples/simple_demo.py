#!/usr/bin/env python3

import numpy as np
from qsc import Qsc

stel = Qsc(rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, order='r2', B2c=-0.7, p2=-600000.)

print(stel.iota) # Rotational transform on-axis for this configuration
print(stel.d2_volume_d_psi2) # Magnetic well V''(psi)
print(stel.DMerc_times_r2) # Mercier criterion parameter DMerc multiplied by r^2
print(stel.min_L_grad_B) # Scale length associated with the grad grad B tensor
print(stel.grad_grad_B_inverse_scale_length) # Scale length associated with the grad grad B tensor
stel.plot_boundary() # Plot the flux surface shape at the default radius r=1
stel.plot() # Plot relevant near axis parameters
