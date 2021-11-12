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
import booz_xform as bx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_from_boozxform(name, r=0.008, nphi=151):
    # Add the directory of this file to the specified filename:
    inputFile="input."+str(name).replace(" ","")
    abs_filename = os.path.join(os.path.dirname(__file__), inputFile)
    woutFile="wout_"+str(name).replace(" ","")+".nc"
    boozFile="boozmn_"+str(name).replace(" ","")+".nc"
    # Run pyQsc and create a VMEC input file
    logger.info('Creating pyQSC configuration')
    py = Qsc.from_paper(name, nphi=nphi)
    logger.info('Outputing to VMEC')

    py.to_vmec(inputFile,r,params={"ftol_array": [2e-17,5e-17,3e-16,5e-16], "ns_array": [16,49,101,151], "niter_array": [2000,2000,2000,2000]})

    # Run VMEC
    fcomm = MPI.COMM_WORLD.py2f()
    logger.info("Calling runvmec. comm={}".format(fcomm))
    ictrl=np.array([15,0,0,0,0], dtype=np.int32)
    vmec.runvmec(ictrl, inputFile, True, fcomm, '')
    # Check that VMEC converged
    assert ictrl[1] == 11
    vmec.cleanup(True)

    # Run VMEC locally
    # bashCommand = "./xvmec2000 input."+str(name).replace(" ","")
    # from subprocess import run
    # run(bashCommand.split())

    # Run BOOZ_XFORM
    b1 = bx.Booz_xform()

    b1.read_wout(woutFile)
    b1.compute_surfs = [0,5,10,15,20,25,30,40,50,60,75,90,105,125,149]
    b1.mboz = 120
    b1.nboz = 40
    b1.run()
    b1.write_boozmn(boozFile)

    # Read local BOOZ_XFORM file
    # b1.read_boozmn(boozFile)
    # Check results
    stel = Qsc.from_boozxform(boozFile, vmec_file=woutFile, order=py.order, sigma0=py.sigma0, I2=py.I2)
    logger.info('Initial iota on axis = '+str(py.iota)+'for case '+name)
    logger.info('Final   iota on axis = '+str(stel.iota)+'for case '+name)
    assert np.isclose(py.iota,stel.iota,rtol=1e-2)
    logger.info('Initial B0 = '+str(py.B0)+'for case '+name)
    logger.info('Final   B0 = '+str(stel.B0)+'for case '+name)
    assert np.isclose(py.B0,stel.B0,rtol=1e-2)
    logger.info('Initial etabar = '+str(py.etabar)+'for case '+name)
    logger.info('Final   etabar = '+str(stel.etabar)+'for case '+name)
    assert np.isclose(py.etabar,stel.etabar,rtol=2e-2)
    if stel.order != 'r1':
        logger.info('Initial B2c = '+str(py.B2c)+'for case '+name)
        logger.info('Final   B2c = '+str(stel.B2c)+'for case '+name)
        logger.info('Initial B2s = '+str(py.B2s)+'for case '+name)
        logger.info('Final   B2s = '+str(stel.B2s)+'for case '+name)
        assert np.isclose(py.B2c,stel.B2c,rtol=2e-2,atol=5e-3)
        assert np.isclose(py.B2s,stel.B2s,rtol=1e-3,atol=1e-3)

class FromBoozxformTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(FromBoozxformTests, self).__init__(*args, **kwargs)
        logger = logging.getLogger('qsc.qsc')
        logger.setLevel(1)
        self.cases=["r1 section 5.1","r1 section 5.2","r1 section 5.3",\
                    "r2 section 5.1","r2 section 5.2","r2 section 5.3","r2 section 5.4","r2 section 5.5"]

    def test_from_boozxform(self):
        """
        Verify that the we can read the Booz_Xform and VMEC files and
        that iota on axis match the predicted values when using the to_vmec
        and from_boozxform functions one after the other.
        """
        for case in self.cases:
            logger.info('Going through case '+case)
            compare_from_boozxform(case)
