#!/usr/bin/env python3

import unittest
import numpy as np
import logging
from qsc.qsc import Qsc

logger = logging.getLogger(__name__)

class GradBTensorTests(unittest.TestCase):

    def test_axisymmetry(self):
        """
        Test that the grad B tensor is correct for an axisymmetric vacuum field
        """
        for sG in [-1, 1]:
            for spsi in [-1, 1]:
                for nfp in range(1, 4):
                    for j in range(3):
                        R0 = np.random.rand() * 2 + 0.3
                        B0 = np.random.rand() * 2 + 0.3
                        nphi = int(np.random.rand() * 10 + 3)
                        etabar = (np.random.rand() - 0.5) * 4
                        stel = Qsc(rc=[R0], zs=[0], nfp=nfp, nphi=nphi, sG=sG, \
                                   spsi=spsi, B0=B0, etabar=etabar)
                        val = sG * B0 / R0
                        if np.mod(nphi, 2) == 0:
                            nphi += 1

                        rtol = 1e-13
                        atol = 1e-13
                        np.testing.assert_allclose(stel.grad_B_tensor.nt, np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stel.grad_B_tensor.tn, np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stel.grad_B_tensor.nn, np.zeros(nphi), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stel.grad_B_tensor.bb, np.zeros(nphi), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stel.grad_B_tensor.nb, np.zeros(nphi), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stel.grad_B_tensor.bn, np.zeros(nphi), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stel.L_grad_B, np.full(nphi, R0), rtol=rtol, atol=atol)
                        
class GradGradBTensorTests(unittest.TestCase):

    def test_alternate_derivation_and_symmetry(self):
        """
        Compare Rogerio's derivation to mine to ensure the results
        coincide.  Also verify symmetry in the first two components.
        """
        for sG in [-1, 1]:
            for spsi in [-1, 1]:
                for config in [1, 2, 3, 4, 5]:
                    B0 = np.random.rand() * 0.4 + 0.8
                    nphi = int(np.random.rand() * 50 + 61)
                    s = Qsc.from_paper(config, sG=sG, spsi=spsi, B0=B0, nphi=65)
                    s.calculate_grad_grad_B_tensor(two_ways=True)
                    logger.info("Max difference between Matt and Rogerio's derivation for config {} is {}".format(config, np.max(np.abs(s.grad_grad_B - s.grad_grad_B_alt))))
                    np.testing.assert_allclose(s.grad_grad_B, s.grad_grad_B_alt, rtol=1e-8, atol=1e-8)
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                # For all configs, the tensor should be symmetric in the 1st 2 indices:
                                np.testing.assert_allclose(s.grad_grad_B[:, i, j, k], s.grad_grad_B[:, j, i, k], rtol=1e-8, atol=1e-8)
                                # For curl-free fields, the tensor should also be symmetric in the last 2 indices:
                                if config in {1, 2, 4}:
                                    np.testing.assert_allclose(s.grad_grad_B[:, i, j, k], s.grad_grad_B[:, i, k, j], rtol=1e-8, atol=1e-8)
            
    def test_axisymmetry(self):
        """
        Test that the grad grad B tensor is correct for an axisymmetric vacuum field
        """
        for sG in [-1, 1]:
            for spsi in [-1, 1]:
                for nfp in range(1, 4):
                    for j in range(3):
                        R0 = np.random.rand() * 2 + 0.3
                        B0 = np.random.rand() * 2 + 0.3
                        nphi = int(np.random.rand() * 10 + 3)
                        etabar = (np.random.rand() - 0.5) * 4
                        s = Qsc(rc=[R0], zs=[0], nfp=nfp, nphi=nphi, sG=sG, \
                                   spsi=spsi, B0=B0, etabar=etabar, I2=1.e-8, order='r2')
                        # The O(r^2) solve fails if iota is exactly 0,
                        # so above we set a tiny nonzero current to
                        # create a tiny nonzero iota.
                        if np.mod(nphi, 2) == 0:
                            nphi += 1

                        s.calculate_grad_grad_B_tensor(two_ways=True)
                        # Check all components other than {tnn, ntn, nnt, ttt}. These should be 0.
                        rtol = 1e-6
                        atol = 1e-6
                        for i in range(3):
                            for j in range(3):
                                for k in range(3):
                                    if not ((i==2 and j==2 and k==2) or (i==2 and j==0 and k==0) or (i==0 and j==2 and k==0) or (i==i and j==0 and k==2)):
                                        np.testing.assert_allclose(s.grad_grad_B[:,i,j,k], np.zeros(nphi), rtol=rtol, atol=atol)
                                        np.testing.assert_allclose(s.grad_grad_B_alt[:,i,j,k], np.zeros(nphi), rtol=rtol, atol=atol)
                                        
                        # Check ttt component
                        val = -2 * sG * B0 / (R0 * R0)
                        np.testing.assert_allclose(s.grad_grad_B[:,2,2,2], np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(s.grad_grad_B_alt[:,2,2,2], np.full(nphi, val), rtol=rtol, atol=atol)

                        # Check tnn, ntn, and nnt components:
                        val = 2 * sG * B0 / (R0 * R0)
                        np.testing.assert_allclose(s.grad_grad_B[:,2,0,0], np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(s.grad_grad_B[:,0,2,0], np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(s.grad_grad_B[:,0,0,2], np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(s.grad_grad_B_alt[:,2,0,0], np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(s.grad_grad_B_alt[:,0,2,0], np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(s.grad_grad_B_alt[:,0,0,2], np.full(nphi, val), rtol=rtol, atol=atol)
                        
if __name__ == "__main__":
    unittest.main()
