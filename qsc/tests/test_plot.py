#!/usr/bin/env python3

import unittest
import numpy as np
from qsc.qsc import Qsc
from scipy.io import netcdf
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fortran_plot_single(filename, ntheta=150, nphi = 4):
    """
    Function to extract boundary arrays from the fortran files
    """
    abs_filename = os.path.join(os.path.dirname(__file__), filename)
    f = netcdf.netcdf_file(abs_filename,mode='r',mmap=False)
    r = f.variables['r'][()]
    nfp = f.variables['nfp'][()]
    nphi_axis = f.variables['N_phi'][()]
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
    phi1D = np.linspace(0,2*np.pi,nphi)
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

    return R, z, R0, z0, r, mpol, ntor, nphi_axis

def compare_with_fortran(name, filename, ntheta = 60, nphi = 100, rtol=1e-7, atol=1e-7):
    """
    Compare output from pyQSC to the fortran code, for one
    of the example configurations from the papers.
    """
    logger.info('Extracting R, Z, R0 and Z0 from fortran output files')
    R_fortran, Z_fortran, R0_fortran, Z0_fortran, r, mpol, ntor, nphi_axis = fortran_plot_single(filename=filename, ntheta=ntheta, nphi=nphi)

    logger.info('Creating pyQSC configuration')
    order = 'r2' if name[1] == '2' else 'r1'
    stel = Qsc.from_paper(name, nphi=nphi_axis, order=order)
    logger.info('Creating R and Z arrays with get_boundary function')
    _, _, Z_qsc, R_qsc = stel.get_boundary(r=r, ntheta=ntheta, nphi=nphi, mpol=mpol, ntor=ntor, ntheta_fourier=2*mpol)
    logger.info('Creating R0 and Z0 arrays with R0_func and Z0_func functions')
    phi_array = np.linspace(0, 2*np.pi, nphi)
    R0_qsc = stel.R0_func(phi_array)
    Z0_qsc = stel.Z0_func(phi_array)
    logger.info('Asserting qsc and output fortran files are equal')
    np.testing.assert_allclose(R_fortran, R_qsc,   rtol=rtol, atol=atol)
    np.testing.assert_allclose(Z_fortran, Z_qsc,   rtol=rtol, atol=atol)
    np.testing.assert_allclose(R0_fortran, R0_qsc, rtol=rtol, atol=atol)
    np.testing.assert_allclose(Z0_fortran, Z0_qsc, rtol=rtol, atol=atol)

class PlotTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PlotTests, self).__init__(*args, **kwargs)
        logger = logging.getLogger('qsc.qsc')
        logger.setLevel(1)
        self.cases = ["r1 section 5.1","r1 section 5.2","r1 section 5.3",\
                      "r2 section 5.1","r2 section 5.2","r2 section 5.3","r2 section 5.4","r2 section 5.5"]
        self.fortran_names = ["quasisymmetry_out.LandremanSenguptaPlunk2019_section5.1_order_r1_finite_r_nonlinear.nc",
                              "quasisymmetry_out.LandremanSenguptaPlunk2019_section5.2_order_r1_finite_r_nonlinear.nc",
                              "quasisymmetry_out.LandremanSenguptaPlunk2019_section5.3_order_r1_finite_r_nonlinear.nc",
                              "quasisymmetry_out.LandremanSengupta2019_section5.1_order_r2_finite_r_nonlinear.nc",
                              "quasisymmetry_out.LandremanSengupta2019_section5.2_order_r2_finite_r_nonlinear.nc",
                              "quasisymmetry_out.LandremanSengupta2019_section5.3_order_r2_finite_r_nonlinear.nc",
                              "quasisymmetry_out.LandremanSengupta2019_section5.4_order_r2_finite_r_nonlinear.nc",
                              "quasisymmetry_out.LandremanSengupta2019_section5.5_order_r2_finite_r_nonlinear.nc"]

    def test_plot_call(self):
        """
        A call to plot() to check that it works with
        a first order case and a second order one
        """
        stel = Qsc.from_paper("r1 section 5.1")
        # stel.plot(fieldlines=True)
        stel.plot(show=False)
        stel.B_fieldline(show=False)
        stel.B_contour(show=False)
        stel = Qsc.from_paper(4)
        # stel.plot(fieldlines=True)
        stel.plot(show=False)
        stel.B_fieldline(show=False)
        stel.B_contour(show=False)

    def test_compare_with_fortran(self):
        """
        Compare the RBC/RBS/ZBC/ZBS values to those generated by the fortran version
        """
        for i in range(len(self.cases)):
            logger.info('Going through case '+self.cases[i])
            compare_with_fortran(self.cases[i], self.fortran_names[i])

    def test_axis(self):
        """
        Test call to plot axis shape
        """
        stel = Qsc.from_paper('r1 section 5.2')
        stel.plot_axis(frenet=False, show=False)
        stel = Qsc.from_paper(4)
        stel.plot_axis(frenet=False, show=False)
                
if __name__ == "__main__":
    unittest.main()
