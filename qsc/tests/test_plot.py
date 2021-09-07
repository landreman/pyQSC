#!/usr/bin/env python3

import unittest
import numpy as np
from qsc.qsc import Qsc
from scipy.io import netcdf
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlotTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PlotTests, self).__init__(*args, **kwargs)
        logger = logging.getLogger('qsc.qsc')
        logger.setLevel(1)

    def test_plot_call(self):
        """
        A call to plot() to check that it works with
        a first order case and a second order one
        """
        stel = Qsc.from_paper("r1 section 5.1")
        # stel.plot(fieldlines=True)
        stel.plot(show=False)
        stel.B_fieldline(show=False)
        stel.B_contour(show=False)
        stel = Qsc.from_paper(4)
        # stel.plot(fieldlines=True)
        stel.plot(show=False)
        stel.B_fieldline(show=False)
        stel.B_contour(show=False)

    def test_axis(self):
        """
        Test call to plot axis shape
        """
        stel = Qsc.from_paper('r1 section 5.2')
        stel.plot_axis(frenet=False, show=False)
        stel = Qsc.from_paper(4)
        stel.plot_axis(frenet=False, show=False)
                
if __name__ == "__main__":
    unittest.main()
