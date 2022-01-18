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

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
