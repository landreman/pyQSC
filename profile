#!/usr/bin/env python3

import cProfile
from qsc import Qsc

cProfile.run("""
for j in range(10):
    s = Qsc(rc=[1, 0.045], zs=[0, 0.045], etabar=0.9, nphi=31, order="r2")
""", sort='tottime')
