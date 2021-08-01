#!/usr/bin/env python3

import unittest

from matplotlib.pyplot import savefig
from qsc.qsc import Qsc

class PlotTests(unittest.TestCase):

    def test_plot(self):
        """
        A call to plot() to check that it works
        """
        stel=Qsc.from_paper("r1 section 5.1")
        # stel.plot(fieldlines=True)
        stel.plot()
        # stel=Qsc.from_paper("r1 section 5.2")
        # stel.plot(r=0.03, fieldlines=True)
        # stel.plot(r=0.03)
        # stel=Qsc.from_paper("r1 section 5.3")
        # stel.plot(fieldlines=True)
        # stel.plot()
        # stel=Qsc.from_paper(2)
        # stel.plot(fieldlines=True)
        # stel.plot()
        # stel=Qsc.from_paper(4)
        # stel.plot(fieldlines=True)
        # stel.plot()
                
if __name__ == "__main__":
    unittest.main()
