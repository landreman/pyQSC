"""
This module contains the top-level routines for the quasisymmetric
stellarator construction.
"""

import numpy as np
import logging
#from numba import jit

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Qsc():
    """
    This is the main class for representing the quasisymmetric
    stellarator construction.
    """
    
    # Import methods that are defined in separate files:
    from .init_axis import init_axis
    from .calculate_r1 import _residual, _jacobian, solve_sigma_equation, \
        _determine_helicity, r1_diagnostics
    from .grad_B_tensor import calculate_grad_B_tensor, calculate_grad_grad_B_tensor
    from .calculate_r2 import calculate_r2
    from .mercier import mercier
    
    def __init__(self, rc, zs, rs=[], zc=[], nfp=1, etabar=1., sigma0=0., B0=1.,
                 I2=0., sG=1, spsi=1, nphi=31, B2s=0., B2c=0., p2=0., order="r1"):
        """
        Create a quasisymmetric stellarator.
        """
        # First, force {rc, zs, rs, zc} to have the same length, for
        # simplicity.
        nfourier = np.max([len(rc), len(zs), len(rs), len(zc)])
        self.nfourier = nfourier
        self.rc = np.zeros(nfourier)
        self.zs = np.zeros(nfourier)
        self.rs = np.zeros(nfourier)
        self.zc = np.zeros(nfourier)
        self.rc[:len(rc)] = rc
        self.zs[:len(zs)] = zs
        self.rs[:len(rs)] = rs
        self.zc[:len(zc)] = zc

        # Force nphi to be odd:
        if np.mod(nphi, 2) == 0:
            nphi += 1

        if sG != 1 and sG != -1:
            raise ValueError('sG must be +1 or -1')
        
        if spsi != 1 and spsi != -1:
            raise ValueError('spsi must be +1 or -1')

        self.nfp = nfp
        self.etabar = etabar
        self.sigma0 = sigma0
        self.B0 = B0
        self.I2 = I2
        self.sG = sG
        self.spsi = spsi
        self.nphi = nphi
        self.B2s = B2s
        self.B2c = B2c
        self.p2 = p2
        self.order = order
        self.min_R0_threshold = 0.3
        self._set_names()

        self.calculate()

    def change_nfourier(self, nfourier_new):
        """
        Resize the arrays of Fourier amplitudes. You can either increase
        or decrease nfourier.
        """
        rc_old = self.rc
        rs_old = self.rs
        zc_old = self.zc
        zs_old = self.zs
        index = np.min((self.nfourier, nfourier_new))
        self.rc = np.zeros(nfourier_new)
        self.rs = np.zeros(nfourier_new)
        self.zc = np.zeros(nfourier_new)
        self.zs = np.zeros(nfourier_new)
        self.rc[:index] = rc_old[:index]
        self.rs[:index] = rs_old[:index]
        self.zc[:index] = zc_old[:index]
        self.zs[:index] = zs_old[:index]
        nfourier_old = self.nfourier
        self.nfourier = nfourier_new
        self._set_names()
        # No need to recalculate if we increased the Fourier
        # resolution, only if we decreased it.
        if nfourier_new < nfourier_old:
            self.calculate()

    def calculate(self):
        """
        Driver for the main calculations.
        """
        self.init_axis()
        self.solve_sigma_equation()
        self.r1_diagnostics()
        if self.order == 'r2':
            self.calculate_r2()
       
    def get_dofs(self):
        """
        Return a 1D numpy vector of all possible optimizable
        degrees-of-freedom, for simsopt.
        """
        return np.concatenate((self.rc, self.zs, self.rs, self.zc,
                               np.array([self.etabar, self.sigma0, self.B2s, self.B2c, self.p2, self.I2, self.B0])))

    def set_dofs(self, x):
        """
        For interaction with simsopt, set the optimizable degrees of
        freedom from a 1D numpy vector.
        """
        assert len(x) == self.nfourier * 4 + 7
        self.rc = x[self.nfourier * 0 : self.nfourier * 1]
        self.zs = x[self.nfourier * 1 : self.nfourier * 2]
        self.rs = x[self.nfourier * 2 : self.nfourier * 3]
        self.zc = x[self.nfourier * 3 : self.nfourier * 4]
        self.etabar = x[self.nfourier * 4 + 0]
        self.sigma0 = x[self.nfourier * 4 + 1]
        self.B2s = x[self.nfourier * 4 + 2]
        self.B2c = x[self.nfourier * 4 + 3]
        self.p2 = x[self.nfourier * 4 + 4]
        self.I2 = x[self.nfourier * 4 + 5]
        self.B0 = x[self.nfourier * 4 + 6]
        self.calculate()
        logger.info('set_dofs called with x={}. Now iota={}, elongation={}'.format(x, self.iota, self.max_elongation))
        
    def _set_names(self):
        """
        For simsopt, sets the list of names for each degree of freedom.
        """
        names = []
        names += ['rc({})'.format(j) for j in range(self.nfourier)]
        names += ['zs({})'.format(j) for j in range(self.nfourier)]
        names += ['rs({})'.format(j) for j in range(self.nfourier)]
        names += ['zc({})'.format(j) for j in range(self.nfourier)]
        names += ['etabar', 'sigma0', 'B2s', 'B2c', 'p2', 'I2', 'B0']
        self.names = names

    @classmethod
    def from_paper(cls, name, **kwargs):
        """
        Get one of the configurations that has been used in our papers.
        """
        if name == "r1 section 5.1":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.1 """
            return cls(rc=[1, 0.045], zs=[0, -0.045], nfp=3, etabar=-0.9, **kwargs)
        elif name == "r1 section 5.2":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.2 """
            return cls(rc=[1, 0.265], zs=[0, -0.21], nfp=4, etabar=-2.25, **kwargs)
        elif name == "r1 section 5.3":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.3 """
            return cls(rc=[1, 0.042], zs=[0, -0.042], zc=[0, -0.025], nfp=3, etabar=-1.1, sigma0=-0.6, **kwargs)
        elif name == "r2 section 5.1" or name == '5.1' or name == 1:
            """ The configuration from Landreman & Sengupta (2019), section 5.1 """
            return cls(rc=[1, 0.155, 0.0102], zs=[0, 0.154, 0.0111], nfp=2, etabar=0.64, order='r2', B2c=-0.00322, **kwargs)
        elif name == "r2 section 5.2" or name == '5.2' or name == 2:
            """ The configuration from Landreman & Sengupta (2019), section 5.2 """
            return cls(rc=[1, 0.173, 0.0168, 0.00101], zs=[0, 0.159, 0.0165, 0.000985], nfp=2, etabar=0.632, order='r2', B2c=-0.158, **kwargs)
        elif name == "r2 section 5.3" or name == '5.3' or name == 3:
            """ The configuration from Landreman & Sengupta (2019), section 5.3 """
            return cls(rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, order='r2', B2c=-0.7, p2=-600000., **kwargs)
        elif name == "r2 section 5.4" or name == '5.4' or name == 4:
            """ The configuration from Landreman & Sengupta (2019), section 5.4 """
            return cls(rc=[1, 0.17, 0.01804, 0.001409, 5.877e-05],
                       zs=[0, 0.1581, 0.01820, 0.001548, 7.772e-05], nfp=4, etabar=1.569, order='r2', B2c=0.1348, **kwargs)
        elif name == "r2 section 5.5" or name == '5.5' or name == 5:
            """ The configuration from Landreman & Sengupta (2019), section 5.5 """
            return cls(rc=[1, 0.3], zs=[0, 0.3], nfp=5, etabar=2.5, sigma0=0.3, I2=1.6, order='r2', B2c=1., B2s=3., p2=-0.5e7, **kwargs)
        else:
            raise ValueError('Unrecognized configuration name')

    def min_R0_penalty(self):
        """
        This function can be used in optimization to penalize situations
        in which min(R0) < min_R0_constraint.
        """
        return np.max((0, self.min_R0_threshold - self.min_R0)) ** 2
        
