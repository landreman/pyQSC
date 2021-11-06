#!/usr/bin/env python3

import unittest
import logging
import numpy as np
from qsc.qsc import Qsc

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Iota2Tests(unittest.TestCase):

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
