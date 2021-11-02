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

def compare_shear(name, r=0.03, nphi=201):
    """
    Check that the VMEC configuration obtained with
    the input file created by pyQSC has the same shear
    as the one predicted by pyQSC
    """
    # Add the directory of this file to the specified filename:
    inputFile="input."+str(name).replace(" ","")
    abs_filename = os.path.join(os.path.dirname(__file__), inputFile)
    # Run pyQsc and create a VMEC input file
    logger.info('Creating pyQSC configuration')
    py = Qsc.from_paper(name, nphi=nphi)
    logger.info('Outputing to VMEC')
    # py.to_vmec(abs_filename,r, 
    #     params={"ns_array": [16, 49, 101, 151, 201, 251],
    #             "ftol_array": [1e-17,1e-16,1e-15,1e-14,1e-14,1e-14],
    #             "niter_array": [2000,2000,2000,3000,4000,6000]})
    # Run VMEC
    fcomm = MPI.COMM_WORLD.py2f()
    logger.info("Calling runvmec. comm={}".format(fcomm))
    ictrl=np.array([15,0,0,0,0], dtype=np.int32)
    # vmec.runvmec(ictrl, abs_filename, True, fcomm, '')
    # Check that VMEC converged
    # assert ictrl[1] == 11
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
    phiEDGE=f.variables['phi'][()][-1]
    max_s_for_fit = 0.6
    iotaf = -f.variables['iotaf'][()]
    s_full = np.linspace(0,1,len(iotaf))
    mask = s_full < max_s_for_fit
    p = np.polyfit(s_full[mask], iotaf[mask], 2)
    # import matplotlib.pyplot as plt
    # plt.plot(np.sqrt(s_full), np.polyval(p, s_full),label='fit')
    # plt.plot(np.sqrt(s_full), iotaf, label='iotaf')
    # plt.legend()
    # plt.show()
    print(p[-2]/phiEDGE)
    py.calculate_r3()
    py.calculate_shear(B31c=-170)
    print(py.shear * py.Bbar / 2)
    vmec.cleanup(True)
    f.close()

class ToVmecTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ToVmecTests, self).__init__(*args, **kwargs)
        logger = logging.getLogger('qsc.qsc')
        logger.setLevel(1)
        self.cases = ["r2 section 5.1"]
        # self.cases=["r1 section 5.1","r1 section 5.2","r1 section 5.3",\
        #             "r2 section 5.1","r2 section 5.2","r2 section 5.3","r2 section 5.4","r2 section 5.5"]

    def test_shear(self):
        """
        Compare the shear from VMEC and from pyQSC for
        the second order cases with order='r3'
        """
        for case in self.cases:
            logger.info('Going through case '+case)
            compare_shear(case)
