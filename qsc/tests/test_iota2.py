#!/usr/bin/env python3

import unittest
import logging
import numpy as np
from qsc.qsc import Qsc
from mpi4py import MPI
import vmec
from scipy.io import netcdf
import os

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_shear(name, r=0.05, nphi=201):
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
    runVMEC=False
    if runVMEC:
        logger.info('Outputing to VMEC')
        py.to_vmec(abs_filename,r, 
            params={"ns_array": [16, 49, 101, 151, 201, 251],
                    "ftol_array": [1e-17,1e-16,1e-15,1e-14,1e-14,1e-14],
                    "niter_array": [2000,2000,2000,3000,4000,6000]})
        ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
        if ci:
            # Run VMEC on ci
            fcomm = MPI.COMM_WORLD.py2f()
            logger.info("Calling runvmec. comm={}".format(fcomm))
            ictrl=np.array([15,0,0,0,0], dtype=np.int32)
            vmec.runvmec(ictrl, abs_filename, True, fcomm, '')
            # Check that VMEC converged
            assert ictrl[1] == 11
            vmec.cleanup(True)
        else:
            # Run VMEC locally
            bashCommand = "/Users/rogeriojorge/local/STELLOPT/VMEC2000/Release/xvmec2000 input."+str(name).replace(" ","")
            from subprocess import run
            run(bashCommand.split())
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
    max_s_for_fit = 0.8
    iotaf = -f.variables['iotaf'][()]
    s_full = np.linspace(0,1,len(iotaf))
    mask = s_full < max_s_for_fit
    p = np.polyfit(s_full[mask], iotaf[mask], 3)
    f.close()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.sqrt(s_full), np.polyval(p, s_full),label='fit')
    plt.plot(np.sqrt(s_full), iotaf, label='VMEC iota')
    plt.plot(np.sqrt(s_full), [py.iota] * len(iotaf), label='pyQSC iota')
    plt.legend()
    plt.savefig('iota0 '+name)
    iota2VMEC = py.B0*p[-2]/phiEDGE/2
    py.calculate_r3()
    B31c_array = np.linspace(-30,30,1000)
    iota2_array = []
    for B31c in B31c_array:
        py.calculate_shear(B31c)
        iota2_array.append(py.iota2)
    plt.figure()
    plt.plot(B31c_array,iota2_array,label = 'iota2 from pyQSC')
    plt.plot(B31c_array,[-iota2VMEC] * len(B31c_array),label = 'MINUS iota2 from VMEC')
    plt.xlabel('B31c')
    plt.legend()
    plt.savefig('iota2 '+name)
    plt.show()

class Iota2Tests(unittest.TestCase):

    def test_iota_vs_VMEC(self):
        """
        Compare the shear from VMEC and from pyQSC for
        the second order cases with order='r3'
        """
        cases = ["LandremanPaul2021QA"]
        for case in cases:
            logger.info('Going through case '+case)
            compare_shear(case)

    def test_independent_of_nphi(self):
        """
        iota2 should be insensitive to nphi.
        """
        nphi = [51, 91, 151]
        nnphi = len(nphi)
        for sigma0 in [0, 0.3]:
            iota2 = np.zeros(nnphi)
            for j in range(nnphi):
                q = Qsc.from_paper('precise QA', sigma0=sigma0, nphi=nphi[j])
                q.calculate_shear()
                iota2[j] = q.iota2
            logging.info(f'For sigma0={sigma0}, iota2 for various nphi is {iota2}, '
                         f'rel diff={(max(iota2) - min(iota2)) / iota2[0]}')
            np.testing.assert_allclose(iota2, iota2[0], rtol=1e-3)

    def test_independent_of_nfp(self):
        """
        We are free to represent any configuration with nfp=1, in which
        case iota2 should come out to the same value.
        """
        for sigma0 in [0, 0.4]:
            q1 = Qsc(rc=[1., 0., 0.1], zs=[0., 0., 0.1], I2=1., order='r3', nfp=1, sigma0=sigma0, nphi=101)
            q2 = Qsc(rc=[1., 0.1], zs=[0., 0.1], I2=1., order='r3', nfp=2, sigma0=sigma0, nphi=101)
            q1.calculate_shear()
            q2.calculate_shear()
            logging.info(f'For sigma0={sigma0}, q1.iota2={q1.iota2} q2.iota2={q2.iota2}, '
                         f'rel diff={(q1.iota2 - q2.iota2) / q1.iota2}')
            self.assertAlmostEqual(q1.iota2, q2.iota2, delta=abs(q1.iota2) / 1.0e-4)

    def test_stellarator_symmetry(self):
        """
        Breaking stellarator symmetry by a negligible amount should cause
        iota2 to change a little, since a different integration method
        is used, but iota2 should not change by much.
        """

        q1 = Qsc.from_paper('precise QA')
        q1.calculate_shear()
        
        q2 = Qsc.from_paper('precise QA', sigma0=1e-100)
        q2.calculate_shear()
        logging.info(f'iota2 sym={q1.iota2} sigma0 nonsym={q2.iota2} diff={q1.iota2 - q2.iota2}')
        self.assertGreater(np.abs((q1.iota2 - q2.iota2) / q1.iota2), 1e-10)
        self.assertLess(np.abs((q1.iota2 - q2.iota2) / q1.iota2), 1e-2)
        
        q3 = Qsc.from_paper('precise QA', rs=[1.0e-100])
        q3.calculate_shear()
        logging.info(f'iota2 sym={q1.iota2} rs nonsym={q3.iota2} diff={q1.iota2 - q3.iota2}')
        self.assertGreater(np.abs((q1.iota2 - q3.iota2) / q1.iota2), 1e-10)
        self.assertLess(np.abs((q1.iota2 - q3.iota2) / q1.iota2), 1e-2)
        
        q4 = Qsc.from_paper('precise QA', zc=[1.0e-100])
        q4.calculate_shear()
        logging.info(f'iota2 sym={q1.iota2} zc nonsym={q4.iota2} diff={q1.iota2 - q4.iota2}')
        self.assertGreater(np.abs((q1.iota2 - q4.iota2) / q1.iota2), 1e-10)
        self.assertLess(np.abs((q1.iota2 - q4.iota2) / q1.iota2), 1e-2)
        
        self.assertAlmostEqual(q2.iota2, q3.iota2, delta=abs(q2.iota2) * 1.0e-10)
        self.assertAlmostEqual(q2.iota2, q4.iota2, delta=abs(q2.iota2) * 1.0e-10)

if __name__ == "__main__":
    unittest.main()
