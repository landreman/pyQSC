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

def compare_from_boozxform(name, r=0.005, nphi=151):
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
    woutFile="wout_"+str(name).replace(" ","")+".nc"
    # Check that VMEC converged
    assert ictrl[1] == 11
    vmec.cleanup(True)
    # Run BOOZ_XFORM
    b1 = bx.Booz_xform()
    b1.read_wout(woutFile)
    b1.compute_surfs = [5,10,15,20,30,40,50,60,80,100]
    b1.mboz = 120
    b1.nboz = 40
    b1.run()
    boozFile="boozmn_"+str(name).replace(" ","")+".nc"
    b1.write_boozmn(boozFile)
    # Check results
    stel = Qsc.from_boozxform(boozFile, vmec_file=woutFile)
    logger.info('Initial iota on axis = '+str(py.iota)+'for case '+name)
    logger.info('Final iota on axis = '+str(stel.iota)+'for case '+name)
    assert np.isclose(py.iota,stel.iota,rtol=1e-2)
    logger.info('Initial B0 = '+str(py.B0)+'for case '+name)
    logger.info('Final B0 = '+str(stel.B0)+'for case '+name)
    assert np.isclose(py.B0,stel.B0,rtol=1e-2)
    logger.info('Initial B1c on axis = '+str(py.B1c)+'for case '+name)
    logger.info('Final B1c on axis = '+str(stel.B1c)+'for case '+name)
    assert np.isclose(py.B1c,stel.B1c,rtol=1e-2)
    logger.info('Initial B2c on axis = '+str(py.B2c)+'for case '+name)
    logger.info('Final B2c on axis = '+str(stel.B2c)+'for case '+name)
    assert np.isclose(py.B2c,stel.B2c,rtol=1e-2)

class FromBoozxformTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(FromBoozxformTests, self).__init__(*args, **kwargs)
        logger = logging.getLogger('qsc.qsc')
        logger.setLevel(1)
        self.cases=["r1 section 5.1","r1 section 5.2","r1 section 5.3",\
                    "r2 section 5.1","r2 section 5.2","r2 section 5.3","r2 section 5.4","r2 section 5.5"]
        # self.cases=["r1 section 5.1"]

    def test_from_boozxform(self):
        """
        Verify that the we can read the Booz_Xform and VMEC files and
        that iota on axis match the predicted values.
        """
        for case in self.cases:
            logger.info('Going through case '+case)
            compare_from_boozxform(case)
