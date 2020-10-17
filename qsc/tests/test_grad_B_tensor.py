#!/usr/bin/env python3

import unittest
import numpy as np
from qsc.qsc import Qsc

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
                        
            
if __name__ == "__main__":
    unittest.main()
