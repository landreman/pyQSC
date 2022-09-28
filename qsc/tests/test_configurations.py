#!/usr/bin/env python3

import unittest
import numpy as np
import logging
from qsc.qsc import Qsc

#logging.basicConfig(level=logging.INFO)
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


class ConfigurationsTests(unittest.TestCase):
    def test_configurations_list(self):
        """
        Check the list of available configurations.
        """
        self.assertEqual(len(Qsc.configurations), 20)
        for config in Qsc.configurations:
            # Just test that nothing crashes
            stel = Qsc.from_paper(config)

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

            # Landreman, arXiv:2209.11849 (2022), section 5.1:
            stel = Qsc.from_paper("2022 QA", nphi=nphi)
            self.assertEqual(stel.helicity, 0)
            self.assertAlmostEqual(stel.iota, 0.41900572366804, places=places)
            self.assertAlmostEqual(stel.d2_volume_d_psi2, -19.9522218635808, places=places)
            self.assertAlmostEqual(stel.r_singularity, 0.372988558216609, places=places2)
            check_r2(stel)
            
            # Landreman, arXiv:2209.11849 (2022), section 5.2:
            stel = Qsc.from_paper("2022 QH nfp2", nphi=nphi)
            self.assertEqual(stel.helicity, 1)
            self.assertAlmostEqual(stel.iota, -0.954236580482566, places=places)
            self.assertAlmostEqual(stel.d2_volume_d_psi2, 125.832934212705, places=places)
            self.assertAlmostEqual(stel.r_singularity, 0.57380081585628, places=places2)
            check_r2(stel)
            
            # Landreman, arXiv:2209.11849 (2022), section 5.3:
            stel = Qsc.from_paper("2022 QH nfp3 vacuum", nphi=nphi)
            self.assertEqual(stel.helicity, 1)
            self.assertAlmostEqual(stel.iota, -1.25402177409397, places=places)
            self.assertAlmostEqual(stel.d2_volume_d_psi2, 256.615192741512, places=places)
            self.assertAlmostEqual(stel.r_singularity, 1.19017150226105, places=2)
            check_r2(stel)
            
            # Landreman, arXiv:2209.11849 (2022), section 5.3:
            stel = Qsc.from_paper("2022 QH nfp3 beta", nphi=nphi)
            self.assertEqual(stel.helicity, 1)
            self.assertAlmostEqual(stel.iota, -1.08767867125642, places=places)
            self.assertAlmostEqual(stel.d2_volume_d_psi2, -373.876056716947, places=places)
            self.assertAlmostEqual(stel.r_singularity, 0.330516938190007, places=places2)
            check_r2(stel)
            
            # Landreman, arXiv:2209.11849 (2022), section 5.4:
            stel = Qsc.from_paper("2022 QH nfp4 long axis", nphi=nphi)
            self.assertEqual(stel.helicity, 1)
            self.assertAlmostEqual(stel.iota, -1.77538223796474, places=places)
            self.assertAlmostEqual(stel.d2_volume_d_psi2, 297.749457179557, places=places)
            self.assertAlmostEqual(stel.r_singularity, 0.957566705852123, places=2)
            check_r2(stel)
            
            # Landreman, arXiv:2209.11849 (2022), section 5.4:
            stel = Qsc.from_paper("2022 QH nfp4 well", nphi=nphi)
            self.assertEqual(stel.helicity, 1)
            self.assertAlmostEqual(stel.iota, -1.13766926093388, places=places)
            self.assertAlmostEqual(stel.d2_volume_d_psi2, -39.9505399355026, places=places)
            self.assertAlmostEqual(stel.r_singularity, 0.433497572286833, places=places2)
            check_r2(stel)
            
            # Landreman, arXiv:2209.11849 (2022), section 5.4:
            stel = Qsc.from_paper("2022 QH nfp4 Mercier", nphi=nphi)
            self.assertEqual(stel.helicity, 1)
            self.assertAlmostEqual(stel.iota, -1.60108981435759, places=places)
            self.assertAlmostEqual(stel.d2_volume_d_psi2, -1211.82381025175, places=places)
            self.assertAlmostEqual(stel.r_singularity, 0.0999925405632508, places=places2)
            check_r2(stel)
            
            # Landreman, arXiv:2209.11849 (2022), section 5.5:
            stel = Qsc.from_paper("2022 QH nfp7", nphi=nphi)
            self.assertEqual(stel.helicity, 1)
            self.assertAlmostEqual(stel.iota, -3.65196588712095, places=2)
            self.assertAlmostEqual(stel.d2_volume_d_psi2, 1281.33320214232, places=places)
            self.assertGreater(stel.r_singularity, 100)
            check_r2(stel)
            

                
if __name__ == "__main__":
    unittest.main()
