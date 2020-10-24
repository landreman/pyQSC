#!/usr/bin/env python3

import unittest
import os
from scipy.io import netcdf
import numpy as np
import logging
from qsc.qsc import Qsc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_r2(s):
    """
    Verify that the O(r^2) equations have been solved, using a
    different method than the method used to solve them originally.
    """
    B0_over_abs_G0 = s.B0 / np.abs(s.G0)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    X1c = s.X1c
    Y1s = s.Y1s
    Y1c = s.Y1c
    sigma = s.sigma
    d_d_varphi = s.d_d_varphi
    iota_N = s.iotaN
    curvature = s.curvature
    torsion = s.torsion
    etabar = s.etabar
    B0 = s.B0
    B2s = s.B2s
    B2c = s.B2c
    p2 = s.p2
    sG = s.sG
    spsi = s.spsi
    I2_over_B0 = s.I2 / s.B0
    X20 = s.X20
    X2s = s.X2s
    X2c = s.X2c
    Y20 = s.Y20
    Y2s = s.Y2s
    Y2c = s.Y2c
    Z20 = s.Z20
    Z2s = s.Z2s
    Z2c = s.Z2c
    beta_1s = s.beta_1s
    
    fX0 = np.matmul(d_d_varphi, X20) - torsion * abs_G0_over_B0 * Y20 + curvature * abs_G0_over_B0 * Z20 \
        -4*sG*spsi*abs_G0_over_B0*(Y2c * Z2s - Y2s * Z2c) \
        - spsi * I2_over_B0 * (curvature/2 * X1c * Y1c - 2 * Y20) * abs_G0_over_B0 + abs_G0_over_B0 * beta_1s * Y1c / 2

    fXs = np.matmul(d_d_varphi, X2s) - 2 * iota_N * X2c - torsion * abs_G0_over_B0 * Y2s + curvature * abs_G0_over_B0 * Z2s \
        -4*sG*spsi*abs_G0_over_B0*(-Y20 * Z2c + Y2c * Z20) \
        -spsi * I2_over_B0 * (curvature/2 * X1c * Y1s - 2 * Y2s) * abs_G0_over_B0 - abs_G0_over_B0 * beta_1s * Y1s / 2

    fXc = np.matmul(d_d_varphi, X2c) + 2 * iota_N * X2s - torsion * abs_G0_over_B0 * Y2c + curvature * abs_G0_over_B0 * Z2c \
        -4*sG*spsi*abs_G0_over_B0*(Y20 * Z2s - Y2s * Z20) \
        -spsi * I2_over_B0 * (curvature/2 * X1c * Y1c - 2 * Y2c) * abs_G0_over_B0 - abs_G0_over_B0 * beta_1s * Y1c / 2

    fY0 = np.matmul(d_d_varphi, Y20) + torsion * abs_G0_over_B0 * X20 - 4*sG*spsi*abs_G0_over_B0*(X2s * Z2c - X2c * Z2s) \
        -spsi * I2_over_B0 * (-curvature/2*X1c*X1c + 2*X20) * abs_G0_over_B0 - abs_G0_over_B0 * beta_1s * X1c / 2

    fYs = np.matmul(d_d_varphi, Y2s) - 2 * iota_N * Y2c + torsion * abs_G0_over_B0 * X2s \
        -4*sG*spsi*abs_G0_over_B0*(X20 * Z2c - X2c * Z20) - 2*spsi* I2_over_B0 * X2s * abs_G0_over_B0

    fYc = np.matmul(d_d_varphi, Y2c) + 2 * iota_N * Y2s + torsion * abs_G0_over_B0 * X2c \
        -4*sG*spsi*abs_G0_over_B0*(X2s * Z20 - X20 * Z2s) \
        -spsi * I2_over_B0 * (-curvature/2 * X1c * X1c + 2 * X2c) * abs_G0_over_B0 + abs_G0_over_B0 * beta_1s * X1c / 2

    eq1residual = X1c * fXs - Y1s * fY0 + Y1c * fYs - Y1s * fYc
    eq2residual = -X1c * fX0 + X1c * fXc - Y1c * fY0 + Y1s * fYs + Y1c * fYc
    # Now check the two equations that were used to determine Y2s and Y2c:
    eq3residual = -X1c * Y2c + X1c * Y20 + X2s * Y1s + X2c * Y1c - X20 * Y1c
    eq4residual = X1c * Y2s + X2c * Y1s - X2s * Y1c + X20 * Y1s + sG * spsi * X1c * curvature / 2

    logger.info("max(abs(eq1residual)): {}".format(np.max(np.abs(eq1residual))))
    logger.info("max(abs(eq2residual)): {}".format(np.max(np.abs(eq2residual))))
    logger.info("max(abs(eq3residual)): {}".format(np.max(np.abs(eq3residual))))
    logger.info("max(abs(eq4residual)): {}".format(np.max(np.abs(eq4residual))))

    atol = 1e-8
    np.testing.assert_allclose(eq1residual, np.zeros(s.nphi), atol=atol)
    np.testing.assert_allclose(eq2residual, np.zeros(s.nphi), atol=atol)
    np.testing.assert_allclose(eq3residual, np.zeros(s.nphi), atol=atol)
    np.testing.assert_allclose(eq4residual, np.zeros(s.nphi), atol=atol)

def compare_to_fortran(name, filename):
    """
    Compare output from this python code to the fortran code, for one
    of the example configurations from the papers.
    """
    # Add the directory of this file to the specified filename:
    abs_filename = os.path.join(os.path.dirname(__file__), filename)
    f = netcdf.netcdf_file(abs_filename, 'r')
    nphi = f.variables['N_phi'][()]

    py = Qsc.from_paper(name, nphi=nphi)
    logger.info('Comparing to fortran file ' + abs_filename)

    def compare_field(fortran_name, py_field, rtol=1e-9, atol=1e-9):
        fortran_field = f.variables[fortran_name][()]
        logger.info('max difference in {}: {}'.format(fortran_name, np.max(np.abs(fortran_field - py_field))))
        np.testing.assert_allclose(fortran_field, py_field, rtol=rtol, atol=atol)

    compare_field('iota', py.iota)
    compare_field('curvature', py.curvature)
    compare_field('torsion', py.torsion)
    compare_field('sigma', py.sigma)
    compare_field('modBinv_sqrt_half_grad_B_colon_grad_B', 1 / py.L_grad_B)
    if hasattr(py, 'X20'):
        compare_field('X20', py.X20)
        compare_field('X2s', py.X2s)
        compare_field('X2c', py.X2c)
        compare_field('Y20', py.Y20)
        compare_field('Y2s', py.Y2s)
        compare_field('Y2c', py.Y2c)
        compare_field('Z20', py.Z20)
        compare_field('Z2s', py.Z2s)
        compare_field('Z2c', py.Z2c)
        compare_field('B20', py.B20)
        compare_field('d2_volume_d_psi2', py.d2_volume_d_psi2)
        compare_field('DWell_times_r2', py.DWell_times_r2)
        compare_field('DGeod_times_r2', py.DGeod_times_r2)
        compare_field('DMerc_times_r2', py.DMerc_times_r2)
        compare_field('grad_grad_B_inverse_scale_length_vs_zeta', py.grad_grad_B_inverse_scale_length_vs_varphi)
        compare_field('grad_grad_B_inverse_scale_length', py.grad_grad_B_inverse_scale_length)
    
    f.close()
    
class QscTests(unittest.TestCase):

    def test_curvature_torsion(self):
        """
        Test that the curvature and torsion match an independent
        calculation using the fortran code.
        """
        
        # Stellarator-symmetric case:
        stel = Qsc(rc=[1.3, 0.3, 0.01, -0.001],
                   zs=[0, 0.4, -0.02, -0.003], nfp=5, nphi=15)
        
        curvature_fortran = [1.74354628565018, 1.61776632275718, 1.5167042487094, 
                             1.9179603622369, 2.95373444883134, 3.01448808361584, 1.7714523990583, 
                             1.02055493647363, 1.02055493647363, 1.77145239905828, 3.01448808361582, 
                             2.95373444883135, 1.91796036223691, 1.5167042487094, 1.61776632275717]
        
        torsion_fortran = [0.257226801231061, -0.131225053326418, -1.12989287766591, 
                           -1.72727988032403, -1.48973327005739, -1.34398161921833, 
                           -1.76040161697108, -2.96573007082039, -2.96573007082041, 
                           -1.7604016169711, -1.34398161921833, -1.48973327005739, 
                           -1.72727988032403, -1.12989287766593, -0.13122505332643]

        varphi_fortran = [0, 0.0909479184372571, 0.181828299105257, 
                          0.268782689120682, 0.347551637441381, 0.42101745128188, 
                          0.498195826255542, 0.583626271820683, 0.673010789615233, 
                          0.758441235180374, 0.835619610154036, 0.909085423994535, 
                          0.987854372315234, 1.07480876233066, 1.16568914299866]

        rtol = 1e-13
        atol = 1e-13
        np.testing.assert_allclose(stel.curvature, curvature_fortran, rtol=rtol, atol=atol)
        np.testing.assert_allclose(stel.torsion, torsion_fortran, rtol=rtol, atol=atol)
        np.testing.assert_allclose(stel.varphi, varphi_fortran, rtol=rtol, atol=atol)

        # Non-stellarator-symmetric case:
        stel = Qsc(rc=[1.3, 0.3, 0.01, -0.001],
                   zs=[0, 0.4, -0.02, -0.003],
                   rs=[0, -0.1, -0.03, 0.002],
                   zc=[0.3, 0.2, 0.04, 0.004], nfp=5, nphi=15)
        
        curvature_fortran = [2.10743037699653, 2.33190181686696, 1.83273654023051, 
                             1.81062232906827, 2.28640008392347, 1.76919841474321, 0.919988560478029, 
                             0.741327470169023, 1.37147330126897, 2.64680884158075, 3.39786486424852, 
                             2.47005615416209, 1.50865425515356, 1.18136509189105, 1.42042418970102]
        
        torsion_fortran = [-0.167822738386845, -0.0785778346620885, -1.02205137493593, 
                           -2.05213528002946, -0.964613202459108, -0.593496282035916, 
                           -2.15852857178204, -3.72911055219339, -1.9330792779459, 
                           -1.53882290974916, -1.42156496444929, -1.11381642382793, 
                           -0.92608309386204, -0.868339812017432, -0.57696266498748]

        varphi_fortran = [0, 0.084185130335249, 0.160931495903817, 
                          0.232881563535092, 0.300551168190665, 0.368933497012765, 
                          0.444686439112853, 0.528001290336008, 0.612254611059372, 
                          0.691096975269652, 0.765820243301147, 0.846373713025902, 
                          0.941973362938683, 1.05053459351092, 1.15941650366667]
        rtol = 1e-13
        atol = 1e-13
        np.testing.assert_allclose(stel.curvature, curvature_fortran, rtol=rtol, atol=atol)
        np.testing.assert_allclose(stel.torsion, torsion_fortran, rtol=rtol, atol=atol)
        np.testing.assert_allclose(stel.varphi, varphi_fortran, rtol=rtol, atol=atol)
            
    def test_published_cases(self):
        """
        Solve the sigma equation and verify that the resulting iota matches
        published examples.
        """
        places = 7
        places2 = 3 # For max/min quantities that are less accurate
        for nphi in [50, 63]:
            # Landreman, Sengupta, Plunk (2019), section 5.1:
            stel = Qsc.from_paper('r1 section 5.1', nphi=nphi)
            self.assertEqual(stel.helicity, 0)
            self.assertAlmostEqual(stel.iota, 0.418306910215178, places=places)
            self.assertAlmostEqual(stel.max_elongation, 2.41373705531443, places=places2)
            self.assertAlmostEqual(stel.min_L_grad_B, 1 / 1.52948586064743, places=places2)
            
            # Landreman, Sengupta, Plunk (2019), section 5.2:
            stel = Qsc.from_paper('r1 section 5.2', nphi=nphi)
            self.assertEqual(stel.helicity, -1)
            self.assertAlmostEqual(stel.iota, 1.93109725535729, places=places)
            self.assertAlmostEqual(stel.max_elongation, 3.08125973323805, places=places2)
            self.assertAlmostEqual(stel.min_L_grad_B, 1 / 4.73234243198959, places=places2)
            
            # Landreman, Sengupta, Plunk (2019), section 5.3:
            stel = Qsc.from_paper('r1 section 5.3', nphi=nphi)
            self.assertEqual(stel.helicity, 0)
            self.assertAlmostEqual(stel.iota, 0.311181373123728, places=places)
            self.assertAlmostEqual(stel.max_elongation, 3.30480616121377, places=places2)
            self.assertAlmostEqual(stel.min_L_grad_B, 1 / 1.7014044379421, places=places2)
        
            # Landreman & Sengupta (2019), section 5.1:
            stel = Qsc.from_paper('r2 section 5.1', nphi=nphi)
            self.assertEqual(stel.helicity, 0)
            self.assertAlmostEqual(stel.iota, -0.420473351810416 , places=places)
            self.assertAlmostEqual(stel.max_elongation, 4.38384260252044, places=places2)
            self.assertAlmostEqual(stel.min_L_grad_B, 1 / 1.39153088147691, places=places2)
            check_r2(stel)
        
            # Landreman & Sengupta (2019), section 5.2:
            stel = Qsc.from_paper('r2 section 5.2', nphi=nphi)
            self.assertEqual(stel.helicity, 0)
            self.assertAlmostEqual(stel.iota, -0.423723995700502, places=places)
            self.assertAlmostEqual(stel.max_elongation, 4.86202324600918, places=places2)
            self.assertAlmostEqual(stel.min_L_grad_B, 1 / 1.47675199709439, places=places2)
            check_r2(stel)
            
            # Landreman & Sengupta (2019), section 5.3:
            stel = Qsc.from_paper('r2 section 5.3', nphi=nphi)
            self.assertEqual(stel.helicity, 0)
            self.assertAlmostEqual(stel.iota, 0.959698159859113, places=places)
            self.assertAlmostEqual(stel.max_elongation, 2.20914173760329, places=places2)
            self.assertAlmostEqual(stel.min_L_grad_B, 1 / 1.4922510395338, places=places2)
            check_r2(stel)
            
            # Landreman & Sengupta (2019), section 5.4:
            stel = Qsc.from_paper('r2 section 5.4', nphi=nphi)
            self.assertEqual(stel.helicity, 1)
            self.assertAlmostEqual(stel.iota, -1.14413695118515, places=places)
            self.assertAlmostEqual(stel.max_elongation, 2.98649978627541, places=places2)
            self.assertAlmostEqual(stel.min_L_grad_B, 1 / 2.64098280647292, places=places2)
            check_r2(stel)
            
            # Landreman & Sengupta (2019), section 5.5:
            stel = Qsc.from_paper('r2 section 5.5', nphi=nphi)
            self.assertEqual(stel.helicity, 1)
            self.assertAlmostEqual(stel.iota, -0.828885267089981, places=places)
            self.assertAlmostEqual(stel.max_elongation, 3.6226360623368, places=places2)
            self.assertAlmostEqual(stel.min_L_grad_B, 1 / 4.85287603883526, places=places2)
            check_r2(stel)

    def test_compare_to_fortran(self):
        """
        Compare the output of this python code to the fortran code.
        """
        compare_to_fortran("r2 section 5.1", "quasisymmetry_out.LandremanSengupta2019_section5.1.nc")
        compare_to_fortran("r2 section 5.2", "quasisymmetry_out.LandremanSengupta2019_section5.2.nc")
        compare_to_fortran("r2 section 5.3", "quasisymmetry_out.LandremanSengupta2019_section5.3.nc")
        compare_to_fortran("r2 section 5.4", "quasisymmetry_out.LandremanSengupta2019_section5.4.nc")
        compare_to_fortran("r2 section 5.5", "quasisymmetry_out.LandremanSengupta2019_section5.5.nc")

    def test_change_nfourier(self):
        """
        Test the change_nfourier() method.
        """
        rtol = 1e-13
        atol = 1e-13
        s1 = Qsc.from_paper('r2 section 5.2')
        m = s1.nfourier
        for n in range(1, 7):
            s2 = Qsc.from_paper('r2 section 5.2')
            s2.change_nfourier(n)
            if n <= m:
                # We lowered nfourier
                np.testing.assert_allclose(s1.rc[:n], s2.rc, rtol=rtol, atol=atol)
                np.testing.assert_allclose(s1.rs[:n], s2.rs, rtol=rtol, atol=atol)
                np.testing.assert_allclose(s1.zc[:n], s2.zc, rtol=rtol, atol=atol)
                np.testing.assert_allclose(s1.zs[:n], s2.zs, rtol=rtol, atol=atol)
            else:
                # We increased nfourier
                np.testing.assert_allclose(s1.rc, s2.rc[:m], rtol=rtol, atol=atol)
                np.testing.assert_allclose(s1.rs, s2.rs[:m], rtol=rtol, atol=atol)
                np.testing.assert_allclose(s1.zc, s2.zc[:m], rtol=rtol, atol=atol)
                np.testing.assert_allclose(s1.zs, s2.zs[:m], rtol=rtol, atol=atol)
                z = np.zeros(n - s1.nfourier)
                np.testing.assert_allclose(z, s2.rc[m:], rtol=rtol, atol=atol)
                np.testing.assert_allclose(z, s2.rs[m:], rtol=rtol, atol=atol)
                np.testing.assert_allclose(z, s2.zc[m:], rtol=rtol, atol=atol)
                np.testing.assert_allclose(z, s2.zs[m:], rtol=rtol, atol=atol)
                
if __name__ == "__main__":
    unittest.main()
