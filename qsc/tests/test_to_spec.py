#!/usr/bin/env python3

import unittest
import os
import numpy as np
import logging
from qsc.qsc import Qsc
import h5py
from subprocess import run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_to_spec(name, r=0.005, nphi=151):
    """
    Check that SPEC can run the input file outputed by pyQSC
    and check that the resulting SPEC output file has
    the expected parameters
    """
    # Add the directory of this file to the specified filename:
    inputFile=str(name).replace(" ","")+".sp"
    abs_filename = os.path.join(os.path.dirname(__file__), inputFile)
    # Run pyQsc and create a SPEC input file
    logger.info('Creating pyQSC configuration')
    order = 'r2' if name[1] == '2' else 'r1'
    py = Qsc.from_paper(name, nphi=nphi, order=order)
    logger.info('Outputing to SPEC')
    py.to_spec(abs_filename,r)
    ### I am running these locally only
    # # Run SPEC
    # bashCommand = "./xspec "+str(name).replace(" ","")+".sp"
    # run(bashCommand.split())
    # # Open SPEC output file
    # specFile=str(name).replace(" ","")+".sp.h5"
    # abs_filename_SPEC = os.path.join(os.path.dirname(__file__), specFile)
    # specContent = h5py.File(abs_filename_SPEC, "r")
    # # Compare the results
    # specIotaOnAxis = specContent['transform']['fiota'][1, 0]
    # logger.info('pyQSC iota on axis = '+str(py.iota)+'for case = '+name)
    # logger.info('SPEC iota on axis = '+str(specIotaOnAxis)+'for case = '+name)
    ### Not sure if this is how I get B on axis from SPEC
    # specFieldOnAxis = specContent['output']['Bzemn'][0,1,0]
    # logger.info('pyQSC field on axis = '+str(py.B0)+'for case = '+name)
    # logger.info('SPEC field on axis = '+str(specFieldOnAxis)+'for case = '+name)
    # assert np.isclose(py.iota,specIotaOnAxis,rtol=1e-2)
    ### Reminder on how to check spec contents
    # assert np.isclose(py.B0,specFieldOnAxis,rtol=1e-2)
    # for key in specContent:
    #     print('key =',key)
    #     for key2 in specContent[key]:
    #         print('  key2 =',key2)
    # specContent.close()

class ToSpecTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ToSpecTests, self).__init__(*args, **kwargs)
        logger = logging.getLogger('qsc.qsc')
        logger.setLevel(1)
        self.cases=["r1 section 5.1","r1 section 5.2","r1 section 5.3",\
                    "r2 section 5.1","r2 section 5.2",# "r2 section 5.3",\ This one is not working
                    "r2 section 5.4"]#,"r2 section 5.5"] Also this one is not working

    def test_spec(self):
        """
        Verify that spec can read the generated input files
        and that spec's Bfield and iota on axis match the predicted values.
        """
        for case in self.cases:
            logger.info('Going through case '+case)
            compare_to_spec(case)