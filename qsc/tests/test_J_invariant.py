#!/usr/bin/env python3

import unittest
import logging
from qsc.qsc import Qsc
import subprocess
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JInvariantTests(unittest.TestCase):

    def tes_J_order1(self):
        """
        Compare with Mathematica script for the first order examples
        """
        r_test = [0.01,0.05,0.1,0.2]
        Lambda_test = [0.8, 0.9, 1.0, 1.1]
        cases = ['r1 section 5.1', 'r1 section 5.2', 'r1 section 5.3']
        # Result from local runs with mathematica
        J_Mathematica_array = [0.0, 0.0, 0.5520175851832482, 0.0, 0.0, 0.0, 1.2680418920617738,\
              0.0, 0.0, 0.0, 1.857160467252043, 0.0, 0.0, 4.501771267478283,\
              2.8310692330539338, 1.4488896969754543, 0.0, 0.0, 0.24527461529726705,\
              0.0, 0.0, 1.2453777786731264, 0.5882403049125543, 0.11413048114632884,\
              0.0, 1.3236913155055972, 0.9169301477926776, 0.5697621801610244, 2.247061923551934,\
              1.9279774339389866, 1.650584519794647, 1.4001404539190385, 0.0, 0.0,\
              0.8217596057315822, 0.0, 0.0, 0.0, 1.8993562363511547, 0.0, 0.0, 0.0, 2.805302967218362,\
              0.4915725507729421, 0.0, 6.353290484030179, 4.361634115014241, 2.6663917522779927]
        i=0
        for case in cases:
            for r0 in r_test:
                for Lambda0 in Lambda_test:
                    logger.info('Comparing to Mathematica with r='+str(r0)+', lambda='+str(Lambda0)+', case ='+case)
                    # Compute J with pyQSC
                    stel = Qsc.from_paper(case)
                    J_pyQSC=stel.J_invariant(r=r0, Lambda=Lambda0, plot=False)
                    # Compute J with Mathematica script in tests folder, use known values from local runs
                    mathematica_file = os.path.join(os.path.dirname(__file__), 'J_solver.wls')
                    # J_Mathematica = float(subprocess.Popen([mathematica_file+' %s %s %s %s %s %s %s %s %s %s %s %s'\
                    #                         %(stel.B0, 0, stel.B0*stel.etabar, 0, 0, r0, 0, Lambda0, stel.iotaN, stel.nfp, stel.G0, stel.I2)],\
                    #                         shell = True, stdout=subprocess.PIPE).communicate()[0].strip().decode('ascii'))
                    J_Mathematica = J_Mathematica_array[i]
                    logger.info('J_pyQSC ='+str(J_pyQSC))
                    logger.info('J_Mathematica ='+str(J_Mathematica))
                    # Assert that they're close to each other
                    np.testing.assert_allclose(J_pyQSC,J_Mathematica,rtol=1e-6)
                    i+=1

    def test_J_plot(self):
        """
        Check that it is able to call the plot command
        """
        cases = ['r1 section 5.1',4]
        for case in cases:
            stel=Qsc.from_paper(case,p2=-5e6)
            stel.J_invariant(Lambda=1.0,plot=True)
            stel.J_contour()
        
if __name__ == "__main__":
    unittest.main()
