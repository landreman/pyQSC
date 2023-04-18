"""
This module contains the top-level routines for the quasisymmetric
stellarator construction.
"""

import numpy as np

@classmethod
def from_paper(cls, name, **kwargs):
    """
    Get one of the configurations that has been used in papers.
    Available values for ``name`` are::

       "r1 section 5.1"
       "r1 section 5.2"
       "r1 section 5.3"
       "r2 section 5.1"
       "r2 section 5.2"
       "r2 section 5.3"
       "r2 section 5.4"
       "r2 section 5.5"
       "precise QA"
       "precise QA+well"
       "precise QH"
       "precise QH+well"
       "2022 QA"
       "2022 QH nfp2"
       "2022 QH nfp3 vacuum"
       "2022 QH nfp3 beta"
       "2022 QH nfp4 long axis"
       "2022 QH nfp4 well"
       "2022 QH nfp4 Mercier"
       "2022 QH nfp7"

    The list of available configurations is also available as
    :obj:`Qsc.configurations`.  The configurations that begin with
    ``"r1"`` refer to sections in Landreman, Sengupta, and Plunk,
    "Direct construction of optimized stellarator shapes. Part
    2. Numerical quasisymmetric solutions", Journal of Plasma Physics
    85, 905850103 (2019).  The configurations that begin with ``"r2"``
    refer to sections in Landreman and Sengupta, "Constructing
    stellarators with quasisymmetry to high order", Journal of Plasma
    Physics 85, 815850601 (2019).  The configurations that begin with
    ``"precise"`` are fits to the configurations in Landreman and
    Paul, "Magnetic Fields with Precise Quasisymmetry for Plasma
    Confinement", Physical Review Letters 128, 035001 (2022).  The
    configurations that begin with ``"2022"`` refer to Landreman,
    "Mapping the space of quasisymmetric stellarators using optimized
    near-axis expansion", J. Plasma Phys. 88, 905880616 (2022).

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

    elif name == "LandremanPaul2022QA" or name == "precise QA":
        """
        A fit of the near-axis model to the quasi-axisymmetric
        configuration in
        Landreman & Paul, Physical Review Letters 128, 035001 (2022)

        The fit was performed to the boozmn data using the script
        20200621-01-Extract_B0_B1_B2_from_boozxform
        """
        add_default_args(
            kwargs,
            nfp=2,
            rc=[1.0038581971135636, 0.18400998741139907, 0.021723381370503204,
                0.0025968236014410812, 0.00030601568477064874, 3.5540509760304384e-05,
                4.102693907398271e-06, 5.154300428457222e-07, 4.8802742243232844e-08,
                7.3011320375259876e-09],
            zs=[0.0, -0.1581148860568176, -0.02060702320552523, -0.002558840496952667,
                -0.0003061368667524159, -3.600111450532304e-05, -4.174376962124085e-06,
                -4.557462755956434e-07, -8.173481495049928e-08, -3.732477282851326e-09],
            B0=1.006541121335688,
            etabar=-0.6783912804454629,
            B2c=0.26859318908803137,
            nphi=99,
            order='r3',
        )

    elif name == "precise QA+well":
        """
        A fit of the near-axis model to the precise quasi-axisymmetric
        configuration with magnetic well from
        Landreman & Paul, Physical Review Letters 128, 035001 (2022)

        The fit was performed to the boozmn data using the script
        20200621-01-Extract_B0_B1_B2_from_boozxform
        """
        add_default_args(
            kwargs,
            nfp=2,
            rc=[1.0145598919163676, 0.2106377247598754, 0.025469267136340394, 0.0026773601516136727,
                0.00021104172568911153, 7.891887175655046e-06, -8.216044358250985e-07,
                -2.379942694112007e-07, -2.5495108673798585e-08, 1.1679227114962395e-08,
                8.961288962248274e-09],
            zs=[0.0, -0.14607192982551795, -0.021340448470388084, -0.002558983303282255,
                -0.0002355043952788449, -1.2752278964149462e-05, 3.673356209179739e-07,
                9.261098628194352e-08, -7.976283362938471e-09, -4.4204430633540756e-08,
                -1.6019372369445714e-08],
            B0=1.0117071561808106,
            etabar=-0.5064143402495729,
            B2c=-0.2749140163639202,
            nphi=99,
            order='r3',
        )

    elif name == "LandremanPaul2022QH" or name == "precise QH":
        """
        A fit of the near-axis model to the quasi-helically symmetric
        configuration in
        Landreman & Paul, Physical Review Letters 128, 035001 (2022)

        The fit was performed to the boozmn data using the script
        20211001-02-Extract_B0_B1_B2_from_boozxform
        """
        add_default_args(
            kwargs,
            nfp=4,
            rc=[1.0033608429348413, 0.19993025252481125, 0.03142704185268144, 0.004672593645851904,
                0.0005589954792333977, 3.298415996551805e-05, -7.337736061708705e-06,
                -2.8829857667619663e-06, -4.51059545517434e-07],
            zs=[0.0, 0.1788824025525348, 0.028597666614604524, 0.004302393796260442,
                0.0005283708386982674, 3.5146899855826326e-05, -5.907671188908183e-06,
                -2.3945326611145963e-06, -6.87509350019021e-07],
            B0=1.003244143729638,
            etabar=-1.5002839921360023,
            B2c=0.37896407142157423,
            nphi=99,
            order='r3',
        )

    elif name == "precise QH+well":
        """
        A fit of the near-axis model to the precise quasi-helically symmetric
        configuration with magnetic well from
        Landreman & Paul, Physical Review Letters 128, 035001 (2022)

        The fit was performed to the boozmn data using the script
        20211001-02-Extract_B0_B1_B2_from_boozxform
        """
        add_default_args(
            kwargs,
            nfp=4,
            rc=[1.000474932581454, 0.16345392520298313, 0.02176330066615466,
                0.0023779201451133163, 0.00014141976024376502, -1.0595894482659743e-05,
                -2.9989267970578764e-06, 3.464574408947338e-08],
            zs=[0.0, 0.12501739099323073, 0.019051257169780858, 0.0023674771227236587,
                0.0001865909743321566, -2.2659053455802824e-06, -2.368335337174369e-06,
                -1.8521248561490157e-08],
            B0=0.999440074325872,
            etabar=-1.2115187546668142,
            B2c=0.6916862277166693,
            nphi=99,
            order='r3',
        )

    elif name == "2022 QA":
        """ The QA nfp=2 configuration from section 5.1 of Landreman, J. Plasma Phys. 88, 905880616 (2022) """
        add_default_args(
            kwargs,
            nfp=2,
            rc=[1, -0.199449520320017, 0.0239839546865877, -0.00267077266433249, 
                0.000263369906075079, -2.28252686940861e-05, 1.77481423558342e-06, 
                -1.11886947533483e-07],
            zs=[0, 0.153090987614971, -0.0220380957634702, 0.00273207449905532, 
                -0.000289902600946716, 2.60032185367434e-05, -1.93900596618347e-06, 
                1.07177057081779e-07],
            etabar=-0.546960261227405,
            B2c=-0.226693190121799,
            order='r3',
        )

    elif name == "2022 QH nfp2":
        """ The QH nfp=2 configuration from section 5.2 of Landreman, J. Plasma Phys. 88, 905880616 (2022) """
        add_default_args(
            kwargs,
            nfp=2,
            rc=[1,0.6995680628446487, 0.23502036382115418, 0.061503864369157564,
                0.010419139882799225, -5.311696004759487e-08,
                -0.0007331779959884904, -0.0002900010988343009,
                -6.617198558802484e-05, -9.241481219213564e-06,
                -6.284956172067802e-07],
            zs=[0, 0.6214598287182819, 0.23371749756309024, 0.06541788070010997,
                0.011645099864023048, -7.5568378122204045e-06, -0.0008931603464766644,
                -0.00036651597175926245, -8.685195584634676e-05, -1.2617030747711465e-05,
                -8.945854983981342e-07],
            etabar=-0.7598964639478568,
            B2c=-0.09169914960557547,
            order='r3',
        )

    elif name == "2022 QH nfp3 vacuum":
        """ The QA nfp=3 vacuum configuration from section 5.3 of Landreman, J. Plasma Phys. 88, 905880616 (2022) """
        add_default_args(
            kwargs,
            nfp=3,
            rc=[1, 0.44342438066028106, 0.1309928804381408, 0.036826101497868344,
                0.009472569725432308, 0.0021825892707486904, 0.00043801313411164704,
                7.270090423024292e-05, 8.847711104877492e-06, 4.863820022333069e-07,
                -6.48572267338807e-08, -1.309512199798216e-08],
            zs=[0, 0.40118483156012347, 0.1245296767597972, 0.0359252575240197,
                0.009413071841635272, 0.002202227882755186, 0.00044793727963748345,
                7.513385401132283e-05, 9.092986418282475e-06, 3.993637202113794e-07,
                -1.1523282290069935e-07, -2.3010157353892155e-08],
            etabar=1.253110036546191,
            B2c=0.1426420204102797,
            order='r3',
        )

    elif name == "2022 QH nfp3 beta":
        """ The QA nfp=3 configuration with beta > 0 from section 5.3 of Landreman, J. Plasma Phys. 88, 905880616 (2022) """
        add_default_args(
            kwargs,
            nfp=3,
            rc=[1, 0.35202226158037475, 0.07950774007599863, 0.01491931003455014,
                0.0019035177304995063, 2.974489668543068e-05, -5.7768875975485955e-05,
                -1.4029165878029966e-05, -3.636566770484427e-07, 7.14616952513107e-07,
                2.1991450219049712e-07, 2.602997321736813e-08],
            zs=[0, 0.2933368717265116, 0.07312772496167881, 0.014677291769133093,
                0.002032497421621057, 6.751908932231852e-05, -5.485713404214329e-05,
                -1.5321940269647778e-05, -8.529635395421784e-07, 6.820412266134571e-07,
                2.4768295839385676e-07, 3.428344210929051e-08],
            etabar=1.1722273002245573,
            B2c=0.04095882972842455,
            p2=-2000000.0,
            order='r3',
        )

    elif name == "2022 QH nfp4 long axis":
        """ The QA nfp=4 vacuum configuration with long magnetic axis from section 5.4 of Landreman, J. Plasma Phys. 88, 905880616 (2022) """
        add_default_args(
            kwargs,
            nfp=4,
            rc=[1, 0.364526157493978, 0.121118955282543, 0.0442501956803706,
                0.0167330353750722, 0.00642771673694224, 0.00248894981601539,
                0.000968018513866967, 0.00037746342020491, 0.000147428591967393,
                5.76454424945328e-05, 2.25531946406253e-05, 8.82168816589983e-06,
                3.44372305427649e-06, 1.33656083525841e-06, 5.11778503697701e-07,
                1.90503825097649e-07, 6.7112150192589e-08, 2.13264982947534e-08,
                5.58167149815304e-09, 9.68106325338102e-10, 2.05080481895135e-11,
                -3.01196911646619e-11, -1.24160079888634e-12],
            zs=[0, 0.348452925506333, 0.118208201403097, 0.0435300780241838,
                0.0165374758530089, 0.00637325522789471, 0.00247384835222063,
                0.000963898894229973, 0.000376342323515169, 0.000147097870171498,
                5.75176913109258e-05, 2.24818089823666e-05, 8.77306129195814e-06,
                3.4101895486024e-06, 1.31510010810783e-06, 4.99667861692231e-07,
                1.84913811184955e-07, 6.54248703443968e-08, 2.14617350652655e-08,
                6.18264134083988e-09, 1.39965848798875e-09, 1.80608129195069e-10,
                -9.21974743938235e-12, -4.43329488732321e-12],
            etabar=1.44414649103253,
            B2c=0.398731902154125,
            order='r3',
        )

    elif name == "2022 QH nfp4 well":
        """ The QA nfp=4 vacuum configuration with magnetic well from section 5.4 of Landreman, J. Plasma Phys. 88, 905880616 (2022) """
        add_default_args(
            kwargs,
            nfp=4,
            rc=[1, 0.146058715426138, 0.015392595644143, 0.000635861018673727, -0.000182708084736453,
                -4.19903763762926e-05, -9.67542241621129e-07, 1.28914480335549e-06,
                2.62211480710386e-07, 2.1800341254237e-10, -8.42417594606744e-09,
                -1.15051050955081e-09, 1.12882841570496e-11, -1.02361915094644e-12,
                1.06847707789217e-12, 2.6540060912065e-12, -6.41390701236282e-14,
                -2.53410665876572e-13, 4.88649173594043e-14],
            zs=[0, 0.0944567013421266, 0.0124314733883564, 0.000870479159752327,
                -9.20779986777308e-05, -3.53758456786842e-05, -2.95739225483239e-06,
                6.84445244254255e-07, 2.22277038153942e-07, 1.43648174908159e-08,
                -3.79406884758791e-09, 1.16393592133432e-10, 5.40884472454975e-10,
                9.78511899449798e-11, -1.28030361642495e-11, -1.56552634173487e-12,
                1.4116353201481e-12, 5.65879693563372e-14, 2.08417037461475e-13],
            etabar=1.10047627852273,
            B2c=0.369119914393565,
            order='r3',
        )

    elif name == "2022 QH nfp4 Mercier":
        """ The QA nfp=4 finite-beta configuration with Mercier stability from section 5.4 of Landreman, J. Plasma Phys. 88, 905880616 (2022) """
        add_default_args(
            kwargs,
            nfp=4,
            rc=[1, 0.2129392535673467, 0.015441107613823592, -0.002203653896196156,
                -0.0008518709398790752, -0.00010150954708573669, 3.014820573209385e-06,
                2.5369987051312065e-06, 2.4360891263250144e-07, -8.05847285028083e-09],
            zs=[0, 0.1939647576102168, 0.01497505771701485, -0.0020519752655822378,
                -0.0008327023600275835, -0.00010346032332812117, 2.349757901177408e-06,
                2.5571742536309522e-06, 2.6606204989469905e-07, -1.1550694494402663e-08],
            etabar=2.477004356118308,
            B2c=-1.4201479609742513,
            p2=-1018591.6352336337,
            order='r3',
        )

    elif name == "2022 QH nfp7":
        """ The QA nfp=7 vacuum configuration from section 5.5 of Landreman, J. Plasma Phys. 88, 905880616 (2022) """
        add_default_args(
            kwargs,
            nfp=7,
            rc=[1, 0.258042492428667, 0.0737980849415119, 0.0245235419692282,
                0.0087489425800056, 0.00325710762056782, 0.00124798398776684,
                0.00048824282997784, 0.000194039834892702, 7.80531836428887e-05,
                3.16850344178095e-05, 1.29427362598943e-05, 5.30054625293292e-06,
                2.16386280302807e-06, 8.71522369664416e-07, 3.39805766815203e-07,
                1.2389287673191e-07, 3.9648686370117e-08, 9.84425205161371e-09,
                1.39986408835728e-09],
            zs=[0, 0.259299681361503, 0.0744954847634618, 0.0247675058258839,
                0.00883445611523715, 0.00328774095300022, 0.00125909923475415,
                0.000492254404917503, 0.000195431061105875, 7.84791099248868e-05,
                3.17656436049729e-05, 1.29116163099803e-05, 5.24481359745833e-06,
                2.11430136671029e-06, 8.36820327603084e-07, 3.19761744590898e-07,
                1.14811054736367e-07, 3.69561282325895e-08, 9.69489045054884e-09,
                1.59629224344766e-09],
            etabar=2.63694707090359,
            B2c=1.89378085087321,
            order='r3',
        )

    else:
        raise ValueError('Unrecognized configuration name')

    return cls(**kwargs)

# Extract the available configurations from the docstring:
configurations = []
docstring = from_paper.__func__.__doc__.split("\n")
startline = '       "'  # Grab lines that start with this string.
for line in docstring:
    if line[:len(startline)] == startline:
        # Remove the whitespace and quote marks:
        configurations.append(line[len(startline) : -1])
