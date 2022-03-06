#!/usr/bin/env python3
import numpy as np
from qsc import Qsc
import matplotlib.pyplot as plt

###########################################################
########### READ FILES AND CONSTRUCT NAE MODEL ############
# Path of folder with VMEC and BOOZXFORM files
###########################################################
path = "C:\\Users\\erodrigu\\Documents\\MATLAB\\Stellerator\\NAE\\NAE_from_design\\PreciseQSs\\QA\\VMEC_BOOZXFORM\\"
nfp, psi, Raxis, Zaxis = Qsc.read_vmec(file_name = 'wout_mattQAgoodmany.00000.nc', path=path)
# Read the harmonic content of |B| from BOOZXFORM
b_cos2, b_cos, b_0 = Qsc.read_boozxform(file_name = 'boozmn_mattQAgoodmany.00000.nc', path=path)
# Compute the NAE parameters from |B|
B0 = b_0[0]
z = np.polynomial.polynomial.polyfit(np.sqrt(psi[1:]), b_cos, [1])
etabar = z[1]/np.sqrt(2*b_0[0])
z = np.polynomial.polynomial.polyfit(psi[1:], b_cos2, [1])
B2c = z[1]/2*b_0[0]
# Create the NAE stellarator out of these paramteres
stel = Qsc(rc=Raxis[0:10], zs=Zaxis[0:10], B0 = B0, nfp=nfp, etabar=etabar, I2=0, order='r2', B2c=B2c, p2=0)

####################################
# Plot some features of the problem
####################################
# plt.plot(stel.B20)
# plt.show()
# stel.plot_boundary(r=0.07)
# plt.show()
###########################################################
################### COMPARE EQN TERMS #####################
# Calculate the terms of the self-consistent quasisymmetric 
# equation when ordering it in small rotational transform
################### PARAMETER DEFIINTIONS #################
# (especially because the different treatment of B as 1/B^2)
###########################################################
eta = etabar*np.sqrt(2/B0)
B22c = -(B2c-0.75*etabar**2*B0)*4/B0**4
B0 = 1/B0**2
dldp = stel.abs_G0_over_B0
Ba0 = stel.G0
d_d_varphi = stel.d_d_varphi
k = stel.curvature
dk = np.matmul(d_d_varphi,k)
ddk = np.matmul(d_d_varphi,dk)
dddk = np.matmul(d_d_varphi,ddk)
t = stel.torsion
dt = np.matmul(d_d_varphi,t)
ddt = np.matmul(d_d_varphi,dt)
########################################################
################### EVALUATION TERMS ###################
########################################################
# Dm1 TERM - this is the Euler Elastica term
Dm1 = -16*k**2/Ba0/eta*np.matmul(d_d_varphi,1/k**2*np.matmul(d_d_varphi,t**2+(dk/k)**2/dldp**2+k**2/4))
# D0 TERM - compute the term independent of B2c and B20. One could use the others as well
D0 = 4*eta/k**4*(4*k**4*t-3*k**2*t**3-8*t*(dk/dldp)**2+k/dldp**2*(-9*dk*dt+5*t*ddk)+\
    2*k**2*ddt/dldp**2) + 4*stel.sigma/Ba0/eta/k**4*(-3*k**5*dk+24*k*dk**3/dldp**2-10*k**4*t*dt-\
    30*k**2*dk*ddk/dldp**2-2*k**3*(t**2*dk-2*dddk/dldp**2))

plt.plot(Dm1)
plt.plot(stel.iota*D0)
plt.show()
