#!/usr/bin/env python3

import unittest
import numpy as np
from qsc.util import *

class FourierMinimizeTests(unittest.TestCase):

    def test_sin(self):
        """
        If data represents a single Fourier mode, we know the minimum analytically
        """
        for ndata in [4, 11]:
            for const in [-0.4, 1.3]:
                for amplitude in [2.2, 3.7]:
                    x = np.linspace(0, 2 * np.pi, ndata, endpoint=False)
                    y = const + amplitude * np.sin(x - 1.7)
                    min = fourier_minimum(y)
                    self.assertAlmostEqual(min, const - amplitude, places=14)

if __name__ == "__main__":
    unittest.main()
