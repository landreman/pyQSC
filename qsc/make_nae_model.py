"""
This module contains a function to read VMEC and BOOZXFORM to construct NAE.
"""
import os
import numpy as np
from scipy.io import netcdf
import matplotlib.pyplot as plt

def read_vmec(file_name, path = os.path.dirname(__file__)):
    """
    Read VMEC file to extract the shape of the magnetic axis to construct a NAE model
    out of the equilibrium file.

    Args:
        file_name: name of the VMEC file
        path: path to the VMEC file.
    """
    file_abs = os.path.join(path, file_name)
    f = netcdf.netcdf_file(file_abs, mode='r', mmap=False)
    Raxis = f.variables['raxis_cc'][()]
    Zaxis = f.variables['zaxis_cs'][()]
    nfp = f.variables['nfp'][()]
    psi = f.variables['phi'][()]/2/np.pi
    
    
    return nfp, psi, Raxis, Zaxis

def read_boozxform(file_name, path = os.path.dirname(__file__), helicity=0):
    """
    Read BOOZXFORM file to extract the eta parameter associated to the NAE model

    Args:
        file_name: name of the BOOZXFORM file
        path: path to the BOOZXFORM file.
    """
    file_abs = os.path.join(path, file_name)
    f = netcdf.netcdf_file(file_abs, mode='r', mmap=False)
    bmnc_b = f.variables['bmnc_b'][()]
    ixm_b = f.variables['ixm_b'][()]
    ixn_b = f.variables['ixn_b'][()]
    jlist = f.variables['jlist'][()]
    for i in range(np.size(jlist)):
        if ixm_b[i] == 1 and ixn_b[i]-ixm_b[i]*helicity==0:
            b_cos = bmnc_b[:,i]
        elif ixm_b[i] == 2 and ixn_b[i]-ixm_b[i]*helicity==0:
            b_cos2 = bmnc_b[:,i]
        elif ixm_b[i] == 0 and ixn_b[i] == 0:
            b_0 = bmnc_b[:,i]
    return b_cos2, b_cos, b_0