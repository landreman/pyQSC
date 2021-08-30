#!/usr/bin/env python3

from qsc.util import to_Fourier
import unittest
import os
from scipy.io import netcdf
import numpy as np
import logging
from qsc.qsc import Qsc
from mpi4py import MPI
import vmec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_to_fortran(name, filename):
    """
    Compare output from pyQSC to the fortran code, for one
    of the example configurations from the papers.
    """
    # Add the directory of this file to the specified filename:
    abs_filename = os.path.join(os.path.dirname(__file__), filename)
    f = netcdf.netcdf_file(abs_filename, 'r')
    nphi = f.variables['N_phi'][()]
    r = f.variables['r'][()]
    mpol = f.variables['mpol'][()]
    ntor = f.variables['ntor'][()]

    logger.info('Creating pyQSC configuration')
    order = 'r2' if name[1] == '2' else 'r1'
    py = Qsc.from_paper(name, nphi=nphi, order=order)
    logger.info('Creating RBC, RBS, ZBC and ZBS arrays with to_vmec function')
    py.to_vmec(str("input."+name).replace(" ",""), r=r,params={'mpol': mpol, 'ntor': ntor})

    logger.info('Comparing to fortran file ' + abs_filename)
    def compare_field(fortran_name, py_field, rtol=1e-9, atol=1e-9):
        fortran_field = f.variables[fortran_name][()]
        logger.info('max difference in {}: {}'.format(fortran_name, np.max(np.abs(fortran_field - py_field))))
        np.testing.assert_allclose(fortran_field, py_field, rtol=rtol, atol=atol)

    compare_field('RBC', py.RBC)
    compare_field('RBS', py.RBS)
    compare_field('ZBC', py.ZBC)
    compare_field('ZBS', py.ZBS)
    f.close()

def compare_to_vmec(name, r=0.005, nphi=151):
    """
    Check that VMEC can run the input file outputed by pyQSC
    and check that the resulting VMEC output file has
    the expected parameters
    """
    # Add the directory of this file to the specified filename:
    inputFile="input."+str(name).replace(" ","")
    abs_filename = os.path.join(os.path.dirname(__file__), inputFile)
    # Run pyQsc and create a VMEC input file
    logger.info('Creating pyQSC configuration')
    order = 'r2' if name[1] == '2' else 'r1'
    py = Qsc.from_paper(name, nphi=nphi, order=order)
    logger.info('Outputing to VMEC')
    py.to_vmec(inputFile,r)
    # Run VMEC
    fcomm = MPI.COMM_WORLD.py2f()
    logger.info("Calling runvmec. comm={}".format(fcomm))
    ictrl=np.array([15,0,0,0,0], dtype=np.int32)
    vmec.runvmec(ictrl, inputFile, True, fcomm, '')
    # Check that VMEC converged
    assert ictrl[1] == 11
    # Open VMEC output file
    woutFile="wout_"+str(name).replace(" ","")+".nc"
    f = netcdf.netcdf_file(woutFile, 'r')
    # Compare the results
    logger.info('pyQSC iota on axis = '+str(py.iota))
    logger.info('VMEC iota on axis = '+str(-f.variables['iotaf'][()][0]))
    logger.info('pyQSC field on axis = '+str(py.B0))
    logger.info('VMEC bmnc[1][0] = '+str(f.variables['bmnc'][()][1][0]))
    assert np.isclose(py.iota,-f.variables['iotaf'][()][0],rtol=1e-2)
    assert np.isclose(py.B0,f.variables['bmnc'][()][1][0],rtol=1e-2)
    vmec.cleanup(True)
    f.close()

def Fourier_Inverse(name, r = 0.05, ntheta = 26, nphi = 51, mpol = 13, ntor = 25, atol=1e-9, rtol=1e-9):
    """
    Compute the Fourier transform of a boundary surface and then
    inverse Fourier transform it to find that it arrives
    at the same surface
    """
    logger.info('Creating pyQSC configuration')
    py = Qsc.from_paper(name, nphi=nphi)

    logger.info('Calculating old R_2D and Z_2D')
    R_2D, Z_2D, phi0_2D = py.Frenet_to_cylindrical(r, ntheta)

    logger.info('Calculating corresponding RBC, RBS, ZBC, ZBS')
    RBC, RBS, ZBC, ZBS = to_Fourier(R_2D, Z_2D, py.nfp, mpol, ntor, py.lasym)
    if not py.lasym:
        RBS = np.zeros((int(2*ntor+1),int(mpol+1)))
        ZBC = np.zeros((int(2*ntor+1),int(mpol+1)))

    logger.info('Inverse Fourier transform')
    nphi_conversion = py.nphi
    theta = np.linspace(0,2*np.pi,ntheta,endpoint=False)
    phi_conversion = np.linspace(0,2*np.pi/py.nfp,nphi_conversion,endpoint=False)
    R_2Dnew = np.zeros((ntheta,nphi_conversion))
    Z_2Dnew = np.zeros((ntheta,nphi_conversion))
    phi2d, theta2d = np.meshgrid(phi_conversion, theta)
    for m in range(mpol+1):
        for n in range(-ntor, ntor+1):
            angle = m * theta2d - n * py.nfp * phi2d
            R_2Dnew += RBC[n+ntor,m] * np.cos(angle) + RBS[n+ntor,m] * np.sin(angle)
            Z_2Dnew += ZBC[n+ntor,m] * np.cos(angle) + ZBS[n+ntor,m] * np.sin(angle)

    logger.info('Check the old and the new match')
    np.testing.assert_allclose(R_2D, R_2Dnew, atol=atol, rtol=rtol)
    np.testing.assert_allclose(Z_2D, Z_2Dnew, atol=atol, rtol=rtol)


class ToVmecTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ToVmecTests, self).__init__(*args, **kwargs)
        logger = logging.getLogger('qsc.qsc')
        logger.setLevel(1)
        self.cases=["r1 section 5.1","r1 section 5.2","r1 section 5.3",\
                    "r2 section 5.1","r2 section 5.2","r2 section 5.3","r2 section 5.4","r2 section 5.5"]
        self.fortran_names=["quasisymmetry_out.LandremanSenguptaPlunk2019_section5.1_order_r1_finite_r_nonlinear.nc",
                            "quasisymmetry_out.LandremanSenguptaPlunk2019_section5.2_order_r1_finite_r_nonlinear.nc",
                            "quasisymmetry_out.LandremanSenguptaPlunk2019_section5.3_order_r1_finite_r_nonlinear.nc",
                            "quasisymmetry_out.LandremanSengupta2019_section5.1_order_r2_finite_r_nonlinear.nc",
                            "quasisymmetry_out.LandremanSengupta2019_section5.2_order_r2_finite_r_nonlinear.nc",
                            "quasisymmetry_out.LandremanSengupta2019_section5.3_order_r2_finite_r_nonlinear.nc",
                            "quasisymmetry_out.LandremanSengupta2019_section5.4_order_r2_finite_r_nonlinear.nc",
                            "quasisymmetry_out.LandremanSengupta2019_section5.5_order_r2_finite_r_nonlinear.nc"]
        self.fortran_r3_names=["quasisymmetry_out.LandremanSengupta2019_section5.1.nc",
                               "quasisymmetry_out.LandremanSengupta2019_section5.2.nc",
                               "quasisymmetry_out.LandremanSengupta2019_section5.3.nc",
                               "quasisymmetry_out.LandremanSengupta2019_section5.4.nc",
                               "quasisymmetry_out.LandremanSengupta2019_section5.5.nc"]
    def test_boundary(self):
        """
        Compare the RBC/RBS/ZBC/ZBS values to those generated by the fortran version
        """
        for i in range(len(self.cases)):
            logger.info('Going through case '+self.cases[i])
            compare_to_fortran(self.cases[i], self.fortran_names[i])

    def test_vmec(self):
        """
        Verify that vmec can actually read the generated input files
        and that vmec's Bfield and iota on axis match the predicted values.
        """
        for case in self.cases:
            logger.info('Going through case '+case)
            compare_to_vmec(case)

    def test_Fourier(self):
        """
        Check that transforming with to_Fourier and then un-transforming gives the identity,
        for both even and odd ntheta and phi, and for lasym True or False.
        """
        for case in self.cases:
            logger.info('Going through case '+case)
            Fourier_Inverse(case, atol=1e-9, rtol=1e-9)

    def test_boundary_r3(self, ntheta=20):
        """
        Compare the RBC/RBS/ZBC/ZBS values to those generated by the fortran version with a 3rd order solve
        """
        # filename = 'quasisymmetry_out.LandremanSengupta2019_section5.1.nc'
        for filename in self.fortran_r3_names:
            logger.info('Comparing to fortran file ' + filename)

            logger.info('Reading from Fortran')
            abs_filename = os.path.join(os.path.dirname(__file__), filename)
            f    = netcdf.netcdf_file(abs_filename, 'r', mmap=False)
            r    = f.variables['r'][()]
            p2   = f.variables['p2'][()]
            rc   = f.variables['R0c'][()]
            zs   = f.variables['Z0s'][()]
            B2c  = f.variables['B2c'][()]
            B2s  = f.variables['B2s'][()]
            nfp  = f.variables['nfp'][()]
            B0   =  f.variables['B0'][()]
            mpol = f.variables['mpol'][()]
            ntor = f.variables['ntor'][()]
            nphi = f.variables['N_phi'][()]
            etabar = f.variables['eta_bar'][()]
            I2   =  B0*f.variables['I2_over_B0'][()]
            sigma0 = f.variables['sigma_initial'][()]

            logger.info('Creating pyQSC configuration')
            py = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=etabar, sigma0=sigma0, I2=I2, order='r3', B2c=B2c, B2s=B2s, p2=p2, nphi=nphi)

            logger.info('Creating RBC, RBS, ZBC and ZBS arrays')
            R_2D, Z_2D, phi0_2D = py.Frenet_to_cylindrical(r=r, ntheta=ntheta)
            RBC, RBS, ZBC, ZBS = to_Fourier(R_2D, Z_2D, py.nfp, mpol, ntor, py.lasym)

            RBC = RBC.transpose()
            ZBS = ZBS.transpose()
            if py.lasym:
                RBS = RBS.transpose()
                ZBC = ZBC.transpose()

            def compare_field(fortran_name, py_field, rtol=1e-9, atol=1e-9):
                fortran_field = f.variables[fortran_name][()]
                logger.info('max difference in {}: {}'.format(fortran_name, np.max(np.abs(fortran_field - py_field))))
                np.testing.assert_allclose(fortran_field, py_field, rtol=rtol, atol=atol)

            compare_field('RBC', RBC)
            compare_field('RBS', RBS)
            compare_field('ZBC', ZBC)
            compare_field('ZBS', ZBS)
            compare_field('X3c1', py.X3c1)
            compare_field('X3s1', py.X3s1)
            compare_field('Y3c1', py.Y3c1)
            compare_field('Y3s1', py.Y3s1)
            compare_field('Z3c1', py.Z3c1)
            compare_field('Z3s1', py.Z3s1)
            compare_field('X3c3', py.X3c3)
            compare_field('X3s3', py.X3s3)
            compare_field('Y3c3', py.Y3c3)
            compare_field('Y3s3', py.Y3s3)
            compare_field('Z3c3', py.Z3c3)
            compare_field('Z3s3', py.Z3s3)
            compare_field('B0_order_a_squared_to_cancel', py.B0_order_a_squared_to_cancel)

            f.close()
