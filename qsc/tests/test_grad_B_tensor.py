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
                    np.testing.assert_allclose(s.grad_grad_B, s.grad_grad_B_alt, rtol=1e-7, atol=1e-7)
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
        B0_vals = [0.9, 1.1]
        R0_vals = [0.85, 1.0, 1.2]
        nphi_vals = [3, 5, 7, 9, 11]
        etabar_vals = [0.55, 0.65, 0.95, 1.0, 1.09, 1.3, 1.4]
        index = 0
        for sG in [-1, 1]:
            for spsi in [-1, 1]:
                for nfp in range(1, 4):
                    for j in range(3):
                        """
                        R0 = np.random.rand() * 2 + 0.3
                        B0 = np.random.rand() * 2 + 0.3
                        nphi = int(np.random.rand() * 10 + 3)
                        etabar = (np.random.rand() - 0.5) * 4
                        """
                        # Try many combinations of values, in a deterministic way:
                        R0 = R0_vals[np.mod(index, len(R0_vals))]
                        B0 = B0_vals[np.mod(index, len(B0_vals))]
                        nphi = nphi_vals[np.mod(index, len(nphi_vals))]
                        etabar = etabar_vals[np.mod(index, len(etabar_vals))]
                        index += 1
                        
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
                        
class DirectionalDerivativeTests(unittest.TestCase):
    """
    This test computes the B(\phi), B'(\phi), and B''(\phi) on the magnetic
    axis in two ways, where \phi is the standard cylindrical angle.  
    
    The first way uses the fact that B(\phi) = sG*B0*T(\phi), where T(\phi) is 
    the tangent vector of the Frenet frame associated to the magnetic axis. Since
    we have analytical expressions for the Frenet frame, we can differentiate them
    with respect to \phi.

    The second way relies on the gradB, and gradgradB tensors in pyqsc associated
    to the NAE.  We know that hatB(\phi) = B(\Gamma(\phi)) where Bhat is the magnetic
    field B evaluated on the magnetic axis, given by \Gamma(\phi).  Differentiating
    this expression with respect to \phi and applying the chain rule, we obtain
    a formula for hatB(\phi) that depends on derivative tensors of B.

    With these two independent formulas for B, B', and B'', we can verify that
    the two results are the same.
    """
    def test_B_Bprime_Bprimeprime(self):
        rtol = 1e-8
        atol = 1e-8
        
        np.random.seed(0)
        for sG in [-1, 1]:
            for spsi in [-1, 1]:
                for config in [1, 2, 3, 4, 5]:
                    print(sG, spsi, config)
                    B0 = np.random.rand() * 0.4 + 0.8
                    nphi = int(np.random.rand() * 50 + 61)
                    s = Qsc.from_paper(config, sG=sG, spsi=spsi, B0=B0, nphi=nphi)
                    s.calculate()

                    # this test checks that the derivatives B'(\phi) and B''(\phi) 
                    # when calculated by from the gradB and gradgradB tensors are correct
                    axis_B0 = s.Bfield_cartesian(r=0, theta=0.)
                    axis_B1 = s.grad_B_tensor_cartesian()
                    axis_B2 = s.grad_grad_B_tensor_cartesian()
                    
                    from simsopt.geo import CurveRZFourier
                    nfp=s.nfp
                    G0 = CurveRZFourier(np.linspace(0, 1/nfp, s.phi.size, endpoint=False), s.rc.size-1, nfp, True)
                    G0.rc[:] = s.rc[:]
                    G0.zs[:] = s.zs[1:]
                    G0.x = G0.get_dofs()
                    
                    dG0_dphi = G0.gammadash()
                    d2G0_dphi2 = G0.gammadashdash()
                    
                    ds_dphi = np.linalg.norm(dG0_dphi, axis=1)
                    d2s_dphi2 = np.sum(dG0_dphi*d2G0_dphi2, axis=1)/ds_dphi
                    
                    T, N, B = G0.frenet_frame()
                    kappa = G0.kappa()
                    dkappa_dphi = G0.kappadash()
                    dkappa_ds = dkappa_dphi * (1/ds_dphi)
                    torsion = G0.torsion()
                    
                    dT_ds = kappa[:, None] * N
                    dN_ds = -kappa[:, None] * T + torsion[:, None]*B
                    d2T_ds2 = dkappa_ds[:, None] * N + dN_ds*kappa[:, None]
                    
                    dT_dphi = dT_ds * ds_dphi[:, None]
                    d2T_dphi2 = d2T_ds2*ds_dphi[:, None]**2 + d2s_dphi2[:, None]*dT_ds
                    
                    Bexact = np.transpose(axis_B0, axes=(1, 0))
                    Bqsc = sG*B0*T
                    
                    #check B(\phi)
                    np.testing.assert_allclose(Bexact, Bqsc, rtol=rtol, atol=atol)
                    
                    dG0_dphi = np.transpose(dG0_dphi, axes=(1, 0))
                    dBqsc_dphi = np.einsum('ijk,ik->jk', axis_B1, dG0_dphi)
                    dBexact_dphi = sG*B0*dT_dphi
                    dBexact_dphi = np.transpose(dBexact_dphi, axes=(1, 0))
                    #check B'(\phi)
                    np.testing.assert_allclose(dBexact_dphi, dBqsc_dphi, rtol=rtol, atol=atol)
                    
                    d2G0_dphi2 = np.transpose(d2G0_dphi2, axes=(1, 0))
                    d2Bqsc_dphi2 = np.einsum('ijkl,il,jl->kl', axis_B2, dG0_dphi, dG0_dphi)+np.einsum('ijl,jl->il', axis_B1, d2G0_dphi2)
                    d2Bexact_dphi2 = sG*B0*d2T_dphi2
                    d2Bexact_dphi2 = np.transpose(d2Bexact_dphi2, axes=(1, 0))
                    #check B''(\phi)
                    np.testing.assert_allclose(d2Bexact_dphi2, d2Bqsc_dphi2, rtol=rtol, atol=atol)

class CylindricalCartesianTensorsTests(unittest.TestCase):

    def test_cylindrical_cartesian_tensors(self):
        """
        Test the accuracy and the symmetry in the the cylindrical and Cartesian
        versions of the grad B and grad grad B tensors.
        """
        # Test the grad B tensor in axisymmetry, including symmetry in both indices
        Rmajor = np.random.rand() * 2 + 0.3
        B0 = np.random.rand() * 2 + 1.0
        stel = Qsc(rc=[Rmajor, 0.], zs=[0, 0.], etabar=1.0, B0=B0)
        factor = stel.B0 / Rmajor
        dBdx_cylindrical = stel.grad_B_tensor_cylindrical
        np.testing.assert_almost_equal(dBdx_cylindrical[0, 0], np.zeros(stel.nphi))
        np.testing.assert_almost_equal(dBdx_cylindrical[0, 1], np.full(stel.nphi, -factor))
        np.testing.assert_almost_equal(dBdx_cylindrical[1, 1], np.zeros(stel.nphi))
        np.testing.assert_almost_equal(dBdx_cylindrical, dBdx_cylindrical.transpose(1, 0, 2))
        dBdx_cartesian = stel.grad_B_tensor_cartesian()
        np.testing.assert_almost_equal(dBdx_cartesian[0, 0], np.full(stel.nphi, factor * np.sin(2 * stel.phi)))
        np.testing.assert_almost_equal(dBdx_cartesian[0, 1], np.full(stel.nphi, -factor * np.cos(2 * stel.phi)))
        np.testing.assert_almost_equal(dBdx_cartesian[1, 1], np.full(stel.nphi, -factor * np.sin(2 * stel.phi)))
        np.testing.assert_almost_equal(dBdx_cartesian, dBdx_cartesian.transpose(1, 0, 2))

        # Test the grad B tensor for another configuration, including symmetry in both indices
        stel = Qsc.from_paper(1)
        dBdx_cylindrical = stel.grad_B_tensor_cylindrical
        np.testing.assert_almost_equal(dBdx_cylindrical, dBdx_cylindrical.transpose(1, 0, 2))
        dBdx_cartesian = stel.grad_B_tensor_cartesian()
        np.testing.assert_almost_equal(dBdx_cartesian, dBdx_cartesian.transpose(1, 0, 2))

        # Test the grad grad B tensor, including symmetry in three indices
        dBdxdx_cylindrical = stel.grad_grad_B_tensor_cylindrical()
        np.testing.assert_almost_equal(np.transpose(stel.grad_grad_B, (1, 2, 3, 0)), dBdxdx_cylindrical)
        dBdxdx_cylindrical_transpose_1 = dBdxdx_cylindrical.transpose(0, 2, 1, 3)
        dBdxdx_cylindrical_transpose_2 = dBdxdx_cylindrical.transpose(1, 0, 2, 3)
        dBdxdx_cylindrical_transpose_3 = dBdxdx_cylindrical.transpose(1, 2, 0, 3)
        dBdxdx_cylindrical_transpose_4 = dBdxdx_cylindrical.transpose(2, 0, 1, 3)
        dBdxdx_cylindrical_transpose_5 = dBdxdx_cylindrical.transpose(2, 1, 0, 3)
        np.testing.assert_almost_equal(dBdxdx_cylindrical_transpose_1, dBdxdx_cylindrical)
        np.testing.assert_almost_equal(dBdxdx_cylindrical_transpose_2, dBdxdx_cylindrical)
        np.testing.assert_almost_equal(dBdxdx_cylindrical_transpose_3, dBdxdx_cylindrical)
        np.testing.assert_almost_equal(dBdxdx_cylindrical_transpose_4, dBdxdx_cylindrical)
        np.testing.assert_almost_equal(dBdxdx_cylindrical_transpose_5, dBdxdx_cylindrical)

        dBdxdx_cartesian = stel.grad_grad_B_tensor_cartesian()
        dBdxdx_cartesian_transpose_1 = dBdxdx_cartesian.transpose(0, 2, 1, 3)
        dBdxdx_cartesian_transpose_2 = dBdxdx_cartesian.transpose(1, 0, 2, 3)
        dBdxdx_cartesian_transpose_3 = dBdxdx_cartesian.transpose(1, 2, 0, 3)
        dBdxdx_cartesian_transpose_4 = dBdxdx_cartesian.transpose(2, 0, 1, 3)
        dBdxdx_cartesian_transpose_5 = dBdxdx_cartesian.transpose(2, 1, 0, 3)
        np.testing.assert_almost_equal(dBdxdx_cartesian_transpose_1, dBdxdx_cartesian)
        np.testing.assert_almost_equal(dBdxdx_cartesian_transpose_2, dBdxdx_cartesian)
        np.testing.assert_almost_equal(dBdxdx_cartesian_transpose_3, dBdxdx_cartesian)
        np.testing.assert_almost_equal(dBdxdx_cartesian_transpose_4, dBdxdx_cartesian)
        np.testing.assert_almost_equal(dBdxdx_cartesian_transpose_5, dBdxdx_cartesian)
                
if __name__ == "__main__":
    unittest.main()
