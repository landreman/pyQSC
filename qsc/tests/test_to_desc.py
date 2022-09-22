#!/usr/bin/env python3

import unittest
import logging
import numpy as np
from qsc.qsc import Qsc
from qsc.to_desc import ptolemy_identity

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from desc.io import InputReader
    desc_loaded = True
except ImportError as e:
    logger.debug(str(e))
    desc_loaded = False

def compare_desc_to_vmec(name, r=0.005, nphi=151):
    """
    Check that qsc.to_desc gives the same inputs as qsc.to_vmec
    when read by the DESC input reader.
    """
    # Add the directory of this file to the specified filename:
    inputFile_vmec = "input." + str(name).replace(" ", "") + "_vmec"
    inputFile_desc = "input." + str(name).replace(" ", "") + "_desc"
    # Run pyQsc and create a VMEC input file
    logger.info("Creating pyQSC configuration")
    order = "r2" if name[1] == "2" else "r1"
    py = Qsc.from_paper(name, nphi=nphi, order=order)
    logger.info("Outputing to VMEC")
    py.to_vmec(inputFile_vmec, r)
    py.to_desc(inputFile_desc, r)
    inputs_vmec = InputReader(inputFile_vmec).inputs
    inputs_desc = InputReader(inputFile_desc).inputs
    for arg in ["sym", "NFP", "Psi", "pressure", "iota", "current", "surface", "axis"]:
        np.testing.assert_allclose(inputs_desc[arg], inputs_vmec[arg])


class ToDescTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ToDescTests, self).__init__(*args, **kwargs)
        logger = logging.getLogger("qsc.qsc")
        logger.setLevel(1)
        self.cases = [
            "r1 section 5.1",
            "r1 section 5.2",
            "r1 section 5.3",
            "r2 section 5.1",
            "r2 section 5.2",
            "r2 section 5.3",
            "r2 section 5.4",
            "r2 section 5.5",
        ]

    def test_desc(self):
        """
        Verify that DESC can actually read the generated input files
        and it gives the same input data as to VMEC.
        """
        if desc_loaded:
            for case in self.cases:
                logger.info("Going through case " + case)
                compare_desc_to_vmec(case)

    def test_ptolemy_identity(self):
        """
        Test the conversion from double-angle to double-Fourier form
        using Ptolemy's identity.

        sin test:
        = a0*sin(-z) + a1*sin(t+z) + a3*sin(t)
        = -a0*sin(z) + a1*sin(t)*cos(z) + a1*cos(t)*sin(z) + a3*sin(t)

        cos test:
        = a0 + a2*cos(t+z) + a3*cos(t-z)
        = a0 + (a2+a3)*cos(t)*cos(z) + (a3-a2)*sin(t)*sin(z)
        """
        a0 = 3
        a1 = -1
        a2 = 1
        a3 = 2

        s = np.array([[0, a1], [0, a3], [a0, 0]])
        c = np.array([[0, a2], [a0, 0], [0, a3]])

        m_correct = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        n_correct = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        x_correct = np.array([a3 - a2, a3, a1, -a0, a0, 0, a1, 0, a2 + a3])

        m, n, x = ptolemy_identity(s, c)

        np.testing.assert_allclose(m, m_correct, atol=1e-8)
        np.testing.assert_allclose(n, n_correct, atol=1e-8)
        np.testing.assert_allclose(x, x_correct, atol=1e-8)
