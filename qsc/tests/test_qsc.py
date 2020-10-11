#!/usr/bin/env python3

import unittest
import numpy as np
from qsc.qsc import Qsc

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

    def test_helicity(self):
        """
        Check that the quasisymmetry helicity number N is correct for
        published examples.
        """
        for nphi in [15, 20]:
            # Landreman, Sengupta, Plunk (2019), section 5.1:
            stel = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=3, nphi=nphi)
            self.assertEqual(stel.helicity, 0)
            
            # Landreman, Sengupta, Plunk (2019), section 5.2:
            stel = Qsc(rc=[1, 0.265], zs=[0, -0.21], nfp=4, nphi=nphi)
            self.assertEqual(stel.helicity, -1)
            
            # Landreman, Sengupta, Plunk (2019), section 5.3:
            stel = Qsc(rc=[1, 0.042], zs=[0, -0.042], zc=[0, -0.025], nfp=3, nphi=nphi)
            self.assertEqual(stel.helicity, 0)
        
            # Landreman & Sengupta (2019), section 5.1:
            stel = Qsc(rc=[1, 0.155, 0.0102], zs=[0, 0.154, 0.0111], nfp=2, nphi=nphi)
            self.assertEqual(stel.helicity, 0)
        
            # Landreman & Sengupta (2019), section 5.2:
            stel = Qsc(rc=[1, 0.173, 0.0168, 0.00101], zs=[0, 0.159, 0.0165, 0.000985], nfp=2, nphi=nphi)
            self.assertEqual(stel.helicity, 0)
            
            # Landreman & Sengupta (2019), section 5.3:
            stel = Qsc(rc=[1, 0.09], zs=[0, -0.09], nfp=2, nphi=nphi)
            self.assertEqual(stel.helicity, 0)
            
            # Landreman & Sengupta (2019), section 5.4:
            stel = Qsc(rc=[1, 0.17, 0.01804, 0.001409, 5.877e-05],
                       zs=[0, 0.1581, 0.01820, 0.001548, 7.772e-05], nfp=4, nphi=nphi)
            self.assertEqual(stel.helicity, 1)
            
            # Landreman & Sengupta (2019), section 5.5:
            stel = Qsc(rc=[1, 0.3], zs=[0, 0.3], nfp=5, nphi=nphi)
            self.assertEqual(stel.helicity, 1)
            
    def test_iota(self):
        """
        Solve the sigma equation and verify that the resulting iota matches
        published examples.
        """
        places = 7
        for nphi in [63]:
            # Landreman, Sengupta, Plunk (2019), section 5.1:
            stel = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=3, nphi=nphi, etabar=-0.9)
            self.assertAlmostEqual(stel.iota, 0.418306910215178, places=places)
            self.assertAlmostEqual(stel.max_elongation, 2.41373705531443, places=places)
            
            # Landreman, Sengupta, Plunk (2019), section 5.2:
            stel = Qsc(rc=[1, 0.265], zs=[0, -0.21], nfp=4, nphi=nphi, etabar=-2.25)
            self.assertAlmostEqual(stel.iota, 1.93109725535729, places=places)
            self.assertAlmostEqual(stel.max_elongation, 3.08125973323805, places=places)
            
            # Landreman, Sengupta, Plunk (2019), section 5.3:
            stel = Qsc(rc=[1, 0.042], zs=[0, -0.042], zc=[0, -0.025], nfp=3, nphi=nphi, etabar=-1.1, sigma0=-0.6)
            self.assertAlmostEqual(stel.iota, 0.311181373123728, places=places)
            self.assertAlmostEqual(stel.max_elongation, 3.30480616121377, places=places)
        
            # Landreman & Sengupta (2019), section 5.1:
            stel = Qsc(rc=[1, 0.155, 0.0102], zs=[0, 0.154, 0.0111], nfp=2, nphi=nphi, etabar=0.64)
            self.assertAlmostEqual(stel.iota, -0.420473351810416 , places=places)
            self.assertAlmostEqual(stel.max_elongation, 4.38384260252044, places=places)
        
            # Landreman & Sengupta (2019), section 5.2:
            stel = Qsc(rc=[1, 0.173, 0.0168, 0.00101], zs=[0, 0.159, 0.0165, 0.000985], nfp=2, nphi=nphi, etabar=0.632)
            self.assertAlmostEqual(stel.iota, -0.423723995700502, places=places)
            self.assertAlmostEqual(stel.max_elongation, 4.86202324600918, places=places)
            
            # Landreman & Sengupta (2019), section 5.3:
            stel = Qsc(rc=[1, 0.09], zs=[0, -0.09], nfp=2, nphi=nphi, etabar=0.95, I2=0.9)
            self.assertAlmostEqual(stel.iota, 0.959698159859113, places=places)
            self.assertAlmostEqual(stel.max_elongation, 2.20914173760329, places=places)
            
            # Landreman & Sengupta (2019), section 5.4:
            stel = Qsc(rc=[1, 0.17, 0.01804, 0.001409, 5.877e-05],
                       zs=[0, 0.1581, 0.01820, 0.001548, 7.772e-05], nfp=4, nphi=nphi, etabar=1.569)
            self.assertAlmostEqual(stel.iota, -1.14413695118515, places=places)
            self.assertAlmostEqual(stel.max_elongation, 2.98649978627541, places=places)
            
            # Landreman & Sengupta (2019), section 5.5:
            stel = Qsc(rc=[1, 0.3], zs=[0, 0.3], nfp=5, nphi=nphi, etabar=2.5, sigma0=0.3, I2=1.6)
            self.assertAlmostEqual(stel.iota, -0.828885267089981, places=places)
            self.assertAlmostEqual(stel.max_elongation, 3.6226360623368, places=places)
            
if __name__ == "__main__":
    unittest.main()
