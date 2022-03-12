"""
This module contains the top-level routines for the quasisymmetric
stellarator construction.
"""

import logging
import numpy as np
from scipy.io import netcdf
import matplotlib.pyplot as plt
from sympy import im

#from numba import jit

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Qsc():
    """
    This is the main class for representing the quasisymmetric
    stellarator construction.
    """
    
    # Import methods that are defined in separate files:
    from .init_axis import init_axis, convert_to_spline
    from .calculate_r1 import _residual, _jacobian, solve_sigma_equation, \
        _determine_helicity, r1_diagnostics
    from .grad_B_tensor import calculate_grad_B_tensor, calculate_grad_grad_B_tensor, \
        Bfield_cylindrical, Bfield_cartesian, grad_B_tensor_cartesian, \
        grad_grad_B_tensor_cylindrical, grad_grad_B_tensor_cartesian
    from .calculate_r2 import calculate_r2
    from .calculate_r3 import calculate_r3, calculate_shear
    from .mercier import mercier
    from .r_singularity import calculate_r_singularity
    from .plot import plot, plot_boundary, get_boundary, get_boundary_vmec, B_fieldline, B_contour, plot_axis, flux_tube
    from .Frenet_to_cylindrical import Frenet_to_cylindrical
    from .make_nae_model import read_vmec, read_boozxform
    from .to_vmec import to_vmec
    from .util import B_mag
    
    def __init__(self, rc, zs, rs=[], zc=[], nfp=1, etabar=1., sigma0=0., B0=1.,
                 I2=0., sG=1, spsi=1, nphi=61, B2s=0., B2c=0., p2=0., order="r1"):
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
        if self.order != 'r0':
            self.solve_sigma_equation()
            self.r1_diagnostics()
            if self.order != 'r1':
                self.calculate_r2()
                if self.order == 'r3':
                    self.calculate_r3()
    
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
        Available values for ``name`` are
        ``"r1 section 5.1"``,
        ``"r1 section 5.2"``,
        ``"r1 section 5.3"``,
        ``"r2 section 5.1"``,
        ``"r2 section 5.2"``,
        ``"r2 section 5.3"``,
        ``"r2 section 5.4"``, and
        ``"r2 section 5.5"``.
        These last 5 configurations can also be obtained by specifying an integer 1-5 for ``name``.
        The configurations that begin with ``"r1"`` refer to sections in 
        Landreman, Sengupta, and Plunk, Journal of Plasma Physics 85, 905850103 (2019).
        The configurations that begin with ``"r2"`` refer to sections in 
        Landreman and Sengupta, Journal of Plasma Physics 85, 815850601 (2019).

        You can specify any other arguments of the ``Qsc`` constructor
        in ``kwargs``. You can also use ``kwargs`` to override any of
        the properties of the configurations from the papers. For
        instance, you can modify the value of ``etabar`` in the first
        example using

        .. code-block::

          q = qsc.Qsc.from_paper('r1 section 5.1', etabar=1.1)
        """

        def add_default_args(kwargs_old, **kwargs_new):
            """
            Take any key-value arguments in ``kwargs_new`` and treat them as
            defaults, adding them to the dict ``kwargs_old`` only if
            they are not specified there.
            """
            for key in kwargs_new:
                if key not in kwargs_old:
                    kwargs_old[key] = kwargs_new[key]

                    
        if name == "r1 section 5.1":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.1 """
            add_default_args(kwargs, rc=[1, 0.045], zs=[0, -0.045], nfp=3, etabar=-0.9)
                
        elif name == "r1 section 5.2":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.2 """
            add_default_args(kwargs, rc=[1, 0.265], zs=[0, -0.21], nfp=4, etabar=-2.25)
                
        elif name == "r1 section 5.3":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.3 """
            add_default_args(kwargs, rc=[1, 0.042], zs=[0, -0.042], zc=[0, -0.025], nfp=3, etabar=-1.1, sigma0=-0.6)
                
        elif name == "r2 section 5.1" or name == '5.1' or name == 1:
            """ The configuration from Landreman & Sengupta (2019), section 5.1 """
            add_default_args(kwargs, rc=[1, 0.155, 0.0102], zs=[0, 0.154, 0.0111], nfp=2, etabar=0.64, order='r3', B2c=-0.00322)
            
        elif name == "r2 section 5.2" or name == '5.2' or name == 2:
            """ The configuration from Landreman & Sengupta (2019), section 5.2 """
            add_default_args(kwargs, rc=[1, 0.173, 0.0168, 0.00101], zs=[0, 0.159, 0.0165, 0.000985], nfp=2, etabar=0.632, order='r3', B2c=-0.158)
                             
        elif name == "r2 section 5.3" or name == '5.3' or name == 3:
            """ The configuration from Landreman & Sengupta (2019), section 5.3 """
            add_default_args(kwargs, rc=[1, 0.09], zs=[0, -0.09], nfp=2, etabar=0.95, I2=0.9, order='r3', B2c=-0.7, p2=-600000.)
                             
        elif name == "r2 section 5.4" or name == '5.4' or name == 4:
            """ The configuration from Landreman & Sengupta (2019), section 5.4 """
            add_default_args(kwargs, rc=[1, 0.17, 0.01804, 0.001409, 5.877e-05],
                       zs=[0, 0.1581, 0.01820, 0.001548, 7.772e-05], nfp=4, etabar=1.569, order='r3', B2c=0.1348)
                             
        elif name == "r2 section 5.5" or name == '5.5' or name == 5:
            """ The configuration from Landreman & Sengupta (2019), section 5.5 """
            add_default_args(kwargs, rc=[1, 0.3], zs=[0, 0.3], nfp=5, etabar=2.5, sigma0=0.3, I2=1.6, order='r3', B2c=1., B2s=3., p2=-0.5e7)

        elif name == "LandremanPaul2021QA" or name == "precise QA":
            """
            A fit of the near-axis model to the quasi-axisymmetric
            configuration in Landreman & Paul, arXiv:2108.03711 (2021).

            The fit was performed to the boozmn data using the script
            20200621-01-Extract_B0_B1_B2_from_boozxform
            """
            add_default_args(kwargs,
                             nfp=2,
                             rc=[1.0038581971135636, 0.18400998741139907, 0.021723381370503204, 0.0025968236014410812, 0.00030601568477064874, 3.5540509760304384e-05, 4.102693907398271e-06, 5.154300428457222e-07, 4.8802742243232844e-08, 7.3011320375259876e-09],
                             zs=[0.0, -0.1581148860568176, -0.02060702320552523, -0.002558840496952667, -0.0003061368667524159, -3.600111450532304e-05, -4.174376962124085e-06, -4.557462755956434e-07, -8.173481495049928e-08, -3.732477282851326e-09],
                             B0=1.006541121335688,
                             etabar=-0.6783912804454629,
                             B2c=0.26859318908803137,
                             nphi=99,
                             order='r3')

        elif name == "precise QA+well":
            """
            A fit of the near-axis model to the precise quasi-axisymmetric
            configuration from SIMSOPT with magnetic well.

            The fit was performed to the boozmn data using the script
            20200621-01-Extract_B0_B1_B2_from_boozxform
            """
            add_default_args(kwargs,
                             nfp=2,
                             rc=[1.0145598919163676, 0.2106377247598754, 0.025469267136340394, 0.0026773601516136727, 0.00021104172568911153, 7.891887175655046e-06, -8.216044358250985e-07, -2.379942694112007e-07, -2.5495108673798585e-08, 1.1679227114962395e-08, 8.961288962248274e-09],
                             zs=[0.0, -0.14607192982551795, -0.021340448470388084, -0.002558983303282255, -0.0002355043952788449, -1.2752278964149462e-05, 3.673356209179739e-07, 9.261098628194352e-08, -7.976283362938471e-09, -4.4204430633540756e-08, -1.6019372369445714e-08],
                             B0=1.0117071561808106,
                             etabar=-0.5064143402495729,
                             B2c=-0.2749140163639202,
                             nphi=99,
                             order='r3')
            
        elif name == "LandremanPaul2021QH" or name == "precise QH":
            """
            A fit of the near-axis model to the quasi-helically symmetric
            configuration in Landreman & Paul, arXiv:2108.03711 (2021).

            The fit was performed to the boozmn data using the script
            20211001-02-Extract_B0_B1_B2_from_boozxform
            """
            add_default_args(kwargs,
                             nfp=4,
                             rc=[1.0033608429348413, 0.19993025252481125, 0.03142704185268144, 0.004672593645851904, 0.0005589954792333977, 3.298415996551805e-05, -7.337736061708705e-06, -2.8829857667619663e-06, -4.51059545517434e-07],
                             zs=[0.0, 0.1788824025525348, 0.028597666614604524, 0.004302393796260442, 0.0005283708386982674, 3.5146899855826326e-05, -5.907671188908183e-06, -2.3945326611145963e-06, -6.87509350019021e-07],
                             B0=1.003244143729638,
                             etabar=-1.5002839921360023,
                             B2c=0.37896407142157423,
                             nphi=99,
                             order='r3')

        elif name == "precise QH+well":
            """
            A fit of the near-axis model to the precise quasi-helically symmetric
            configuration from SIMSOPT with magnetic well.

            The fit was performed to the boozmn data using the script
            20211001-02-Extract_B0_B1_B2_from_boozxform
            """
            add_default_args(kwargs,
                             nfp=4,
                             rc=[1.000474932581454, 0.16345392520298313, 0.02176330066615466, 0.0023779201451133163, 0.00014141976024376502, -1.0595894482659743e-05, -2.9989267970578764e-06, 3.464574408947338e-08],
                             zs=[0.0, 0.12501739099323073, 0.019051257169780858, 0.0023674771227236587, 0.0001865909743321566, -2.2659053455802824e-06, -2.368335337174369e-06, -1.8521248561490157e-08],
                             B0=0.999440074325872,
                             etabar=-1.2115187546668142,
                             B2c=0.6916862277166693,
                             nphi=99,
                             order='r3')
            
        else:
            raise ValueError('Unrecognized configuration name')

        return cls(**kwargs)

    @classmethod
    def from_cxx(cls, filename):
        """
        Load a configuration from a ``qsc_out.<extension>.nc`` output file
        that was generated by the C++ version of QSC. Almost all the
        data will be taken from the output file, over-writing any
        calculations done in python when the new Qsc object is
        created.
        """
        def to_string(nc_str):
            """ Convert a string from the netcdf binary format to a python string. """
            temp = [c.decode('UTF-8') for c in nc_str]
            return (''.join(temp)).strip()
        
        f = netcdf.netcdf_file(filename, mmap=False)
        nfp = f.variables['nfp'][()]
        nphi = f.variables['nphi'][()]
        rc = f.variables['R0c'][()]
        rs = f.variables['R0s'][()]
        zc = f.variables['Z0c'][()]
        zs = f.variables['Z0s'][()]
        I2 = f.variables['I2'][()]
        B0 = f.variables['B0'][()]
        spsi = f.variables['spsi'][()]
        sG = f.variables['sG'][()]
        etabar = f.variables['eta_bar'][()]
        sigma0 = f.variables['sigma0'][()]
        order_r_option = to_string(f.variables['order_r_option'][()])
        if order_r_option == 'r2.1':
            order_r_option = 'r3'
        if order_r_option == 'r1':
            p2 = 0.0
            B2c = 0.0
            B2s = 0.0
        else:
            p2 = f.variables['p2'][()]
            B2c = f.variables['B2c'][()]
            B2s = f.variables['B2s'][()]

        q = cls(nfp=nfp, nphi=nphi, rc=rc, rs=rs, zc=zc, zs=zs,
                B0=B0, sG=sG, spsi=spsi,
                etabar=etabar, sigma0=sigma0, I2=I2, p2=p2, B2c=B2c, B2s=B2s, order=order_r_option)
        
        def read(name, cxx_name=None):
            if cxx_name is None: cxx_name = name
            setattr(q, name, f.variables[cxx_name][()])

        [read(v) for v in ['R0', 'Z0', 'R0p', 'Z0p', 'R0pp', 'Z0pp', 'R0ppp', 'Z0ppp',
                        'sigma', 'curvature', 'torsion', 'X1c', 'Y1c', 'Y1s', 'elongation']]
        if order_r_option != 'r1':
            [read(v) for v in ['X20', 'X2c', 'X2s', 'Y20', 'Y2c', 'Y2s', 'Z20', 'Z2c', 'Z2s', 'B20']]
            if order_r_option != 'r2':
                [read(v) for v in ['X3c1', 'Y3c1', 'Y3s1']]
                    
        f.close()
        return q
        
    def min_R0_penalty(self):
        """
        This function can be used in optimization to penalize situations
        in which min(R0) < min_R0_constraint.
        """
        return np.max((0, self.min_R0_threshold - self.min_R0)) ** 2
        
    @classmethod
    def from_boozxform(cls, vmec_file, booz_xform_file, order='r2', max_s_for_fit = 0.4, N_phi = [], N_axis = [],
                        rc=[], rs=[], zc=[], zs=[], sigma0=0, I2=0, p2=0):
        """
        Load a configuration from a VMEC and a BOOZ_XFORM output files
        """
        # Read properties of BOOZ_XFORM output file
        f = netcdf.netcdf_file(booz_xform_file,'r',mmap=False)
        bmnc = f.variables['bmnc_b'][()]
        ixm = f.variables['ixm_b'][()]
        ixn = f.variables['ixn_b'][()]
        jlist = f.variables['jlist'][()]
        nfp = f.variables['nfp_b'][()]
        f.close()

        # Read axis-shape from VMEC output file
        f = netcdf.netcdf_file(vmec_file,'r',mmap=False)
        am = f.variables['am'][()] # Pressure profile polynomial
        rc = f.variables['raxis_cc'][()]
        zs = f.variables['zaxis_cs'][()]
        cls.s_n = rc*(1+nfp**2*np.arange(0,np.size(rc),1)**2)/rc[0]
        if N_axis:
            rc = rc[0:N_axis-1]
            zs = zs[0:N_axis-1]
        psi = f.variables['phi'][()]/2/np.pi
        psi_edge = np.abs(psi[-1])
        bsubumnc = f.variables['bsubumnc'][()]   
        bsubvmnc = f.variables['bsubvmnc'][()] 
        rmnc = f.variables['rmnc'][()]
        zmns = -f.variables['zmns'][()] 
        xm_vmec = f.variables['xm'][()]
        xn_vmec = f.variables['xn'][()]
        iota_vmec = f.variables['iotas'][()] 
        try:
            rs = -f.variables['raxis_cs'][()]
            zc = f.variables['zaxis_cc'][()]
            if N_axis:
                rs = rs[0:N_axis-1]
                zc = zc[0:N_axis-1]
            logger.info('Non stellarator symmetric configuration')
        except:
            rs=[]
            zc=[]
            logger.info('Stellarator symmetric configuration')
        f.close()

        # Calculate nNormal
        stel = Qsc(rc=rc, rs=rs, zc=zc, zs=zs, nfp=nfp)
        helicity = stel.iotaN - stel.iota

        # Prepare coordinates for fit
        psi_booz = np.abs(psi[jlist-1])
        sqrt_psi_booz = np.sqrt(psi_booz)
        mask = psi_booz/psi_edge < max_s_for_fit
        psi_booz_vmec = np.abs(psi)
        mask_vmec = psi_booz_vmec/psi_edge < max_s_for_fit
        # s_fine = np.linspace(0,1,400)
        # sqrts_fine = s_fine
        if N_phi:
            phi = np.linspace(0,2*np.pi / nfp, N_phi)
            B0_phi  = np.zeros(N_phi)
            B1s_phi = np.zeros(N_phi)
            B1c_phi = np.zeros(N_phi)
            B20_phi = np.zeros(N_phi)
            B2s_phi = np.zeros(N_phi)
            B2c_phi = np.zeros(N_phi)
            chck_phi = 1
        else:
            chck_phi = 0
            N_phi = 200
        ### PARAMETER FIT ####
        # Perform fit of parameters for NAE
        for jmn in range(len(ixm)):
            m = ixm[jmn]
            n = ixn[jmn]
            if m>2:
                continue
            if m==0:
                # For m=0, fit a polynomial in s (not sqrt(s)) that does not need to go through the origin.
                if n==0:
                    b_0 = bmnc[mask,jmn]
                    z = np.polynomial.polynomial.polyfit(psi_booz[mask], b_0, [2, 1, 0])
                    B0 = z[0]
                    B20 = z[1]/2*B0
                if chck_phi==1:
                    z = np.polynomial.polynomial.polyfit(psi_booz[mask], bmnc[mask,jmn], [2, 1, 0])
                    B0_phi += z[0] * np.cos(n*phi)
                    B20_phi += z[1] * np.cos(n*phi)/2*b_0[0]
            if m==1:
                if ixn[jmn]-ixm[jmn]*helicity==0:
                    b_cos = bmnc[mask,jmn]
                    z = np.polynomial.polynomial.polyfit(sqrt_psi_booz[mask], b_cos, [3, 1])
                    etabar = z[1]/np.sqrt(2*B0)
                    B31cp = z[3]
                if chck_phi==1:
                    z = np.polynomial.polynomial.polyfit(sqrt_psi_booz[mask],bmnc[mask,jmn], [3, 1])
                    B1c_phi += z[1] * np.cos((n-helicity)*phi)*np.sqrt(B0/2)
                    B1s_phi += z[1] * np.sin((n-helicity)*phi)*np.sqrt(B0/2)
            if m==2:
                # For m=2, fit a polynomial in s (not sqrt(s)) that does need to go through the origin.
                if ixn[jmn]-ixm[jmn]*helicity==0:
                    z = np.polynomial.polynomial.polyfit(psi_booz[mask], bmnc[mask,jmn], [2,1])
                    B2c = z[1]/2*B0
                if chck_phi==1:
                    z = np.polynomial.polynomial.polyfit(psi_booz[mask], bmnc[mask,jmn], [2,1])
                    B2c_phi += z[1] * np.cos((n-2*helicity)*phi)/2*B0
                    B2s_phi += z[1] * np.sin((n-2*helicity)*phi)/2*B0

        # Compute B31c: note that we want the 1/B**2 B31c component and not that of B (the shear
        # expression was obtained using the Jacobian form of |B|). Very sensitive, does not appear
        # to be too reliable (perhaps need a larger s_max for the fit). Often better to take B31c=0 for shear
        B20c = 4*B0**4*(0.75*B0*etabar**2-B20)
        B22c = 4*B0**4*(0.75*B0*etabar**2-B2c)
        eta = etabar*np.sqrt(2/B0)
        B31c = -2/B0**2*(B31cp/B0+1.5*eta*(B20c*B0**2+B22c/2*B0**2)-15*eta**3/8)
        # print(B31c)

        # Read I2 from VMEC
        if I2==0:
            for jmn in range(len(ixm)):
                m = ixm[jmn]
                n = ixn[jmn]
                if m==0 and n==0:
                    G_psi = bsubvmnc[mask_vmec,jmn]
                    I_psi = bsubumnc[mask_vmec,jmn]
                    z = np.polynomial.polynomial.polyfit(psi_booz_vmec[mask_vmec], I_psi, [2,1,0])
                    I2 = z[1]*B0/2
                    if I2<1e-10:
                        I2=0
        # Read p2 from VMEC
        if p2==0:
            r  = np.sqrt(2*psi_edge/B0)
            p2 = am[1]/r/r

        ### CONSTRUCT NAE MODEL ####
        if order=='r1':
            q = cls(rc=rc,rs=rs,zc=zc,zs=zs,etabar=etabar,nphi=N_phi,nfp=nfp,B0=B0,sigma0=sigma0, I2=I2)
        else:
            q = cls(rc=rc,rs=rs,zc=zc,zs=zs,etabar=etabar,nphi=N_phi,nfp=nfp,B0=B0,sigma0=sigma0, I2=I2, B2c=B2c, order=order)
        q.rmnc_vmec = rmnc
        q.zmns_vmec = zmns
        q.xm_vmec = xm_vmec
        q.xn_vmec = xn_vmec
        q.psi_vmec = psi_booz_vmec
        q.iota_vmec = iota_vmec
        if chck_phi==1:
            q.B0_boozxform_array=B0_phi
            q.B1c_boozxform_array=B1c_phi
            q.B1s_boozxform_array=B1s_phi
            q.B20_boozxform_array=B20_phi
            q.B2c_boozxform_array=B2c_phi
            q.B2s_boozxform_array=B2s_phi
        q.B31c = B31c
        return q