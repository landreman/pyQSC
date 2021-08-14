#!/usr/bin/env python3

import numpy as np
from scipy.io import netcdf
import sys, os
import matplotlib.pyplot as plt

def fortran_plot_single(filename, ntheta=150, nphi = 4):

    f = netcdf.netcdf_file(filename,mode='r',mmap=False)
    r = f.variables['r'][()]
    nfp = f.variables['nfp'][()]
    mpol = f.variables['mpol'][()]
    ntor = f.variables['ntor'][()]
    RBC = f.variables['RBC'][()]
    RBS = f.variables['RBS'][()]
    ZBC = f.variables['ZBC'][()]
    ZBS = f.variables['ZBS'][()]
    R0c = f.variables['R0c'][()]
    R0s = f.variables['R0s'][()]
    Z0c = f.variables['Z0c'][()]
    Z0s = f.variables['Z0s'][()]

    theta1D = np.linspace(0,2*np.pi,ntheta)
    phi1D = np.linspace(0,2*np.pi/nfp,nphi,endpoint=False)
    phi2D,theta2D = np.meshgrid(phi1D,theta1D)

    R = np.zeros((ntheta,nphi))
    z = np.zeros((ntheta,nphi))
    for m in range(mpol+1):
        for jn in range(ntor*2+1):
            n = jn-ntor
            angle = m * theta2D - nfp * n * phi2D
            sinangle = np.sin(angle)
            cosangle = np.cos(angle)
            R += RBC[m,jn] * cosangle + RBS[m,jn] * sinangle
            z += ZBC[m,jn] * cosangle + ZBS[m,jn] * sinangle

    R0 = np.zeros(nphi)
    z0 = np.zeros(nphi)
    for n in range(len(R0c)):
        angle = nfp * n * phi1D
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        R0 += R0c[n] * cosangle + R0s[n] * sinangle
        z0 += Z0c[n] * cosangle + Z0s[n] * sinangle

    return R, z, R0, z0, r, mpol, ntor

if __name__ == "__main__":
    filename='quasisymmetry_out.LandremanSengupta2019_section5.4.nc'
    ntheta=150
    nphi = 4
    R, z, R0, z0, _, _, _ = fortran_plot_single(filename=filename, ntheta=ntheta, nphi=nphi)
    _, ax = plt.subplots(1,1,figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
    for jphi in range(nphi):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(R[:,jphi], z[:,jphi],color=color)
        plt.plot(R0[jphi],  z0[jphi],marker="x",linewidth=2,color=color)
    plt.xlabel('R (meters)')
    plt.ylabel('Z (meters)')
    plt.tight_layout()
    ax.set_aspect('equal')
    plt.show()
