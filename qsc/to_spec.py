"""
This module contains the routines to output a
near-axis boundary to a VMEC input file
"""
from datetime import datetime
import numpy as np
from .Frenet_to_cylindrical import Frenet_to_cylindrical
from .util import mu0, to_Fourier

def to_spec(self, filename, r=0.1, params=dict(), ntheta=20, ntorMax=14):
    """
    Outputs the near-axis configuration calculated with pyQSC to
    a text file that is able to be read by SPEC.

    Args:
        filename: name of the text file to be created
        r:  near-axis radius r of the desired boundary surface
        params: a Python dict() instance containing one/several of the following parameters: mpol,
          delt, nstep, tcon0, ns_array, ftol_array, niter_array
        ntheta: resolution in the poloidal angle theta for the Frenet_to_cylindrical and VMEC calculations
        ntorMax: maximum number of NTOR in the resulting SPEC input file
    """
    if "mpol" not in params.keys():
        mpol1d = 100
        mpol = int(np.floor(min(ntheta / 2, mpol1d)))
    else:
        mpol = int(params["mpol"])
    if "ntor" not in params.keys():
        ntord = 100
        ntor = int(min(self.nphi / 2, ntord))
    else:
        ntor = int(params["ntor"])
    if "delt" not in params.keys():
        params["Lrad"] = 10
    if "nmodes" not in params.keys():
        params["nmodes"] = 25
    if "nPpts" not in params.keys():
        params["nPpts"] = 100
    if "nPtrj" not in params.keys():
        params["nPtrj"] = 5

    phiedge = np.pi * r * r * self.spsi * self.Bbar

    ## Where to use am and ac?
    # Set pressure Profile
    temp = - self.p2 * r * r
    am = [temp,-temp]
    pmass_type='power_series'
    pres_scale=1

    # Set current profile:
    ncurr = 1
    pcurr_type = 'power_series'
    ac = [1]
    curtor = 2 * np.pi / mu0 * self.I2 * r * r

    # Get surface shape at fixed off-axis toroidal angle phi
    R_2D, Z_2D, phi0_2D = self.Frenet_to_cylindrical(r, ntheta)
    
    # Fourier transform the result.
    # This is not a rate-limiting step, so for clarity of code, we don't bother with an FFT.
    RBC, RBS, ZBC, ZBS = to_Fourier(R_2D, Z_2D, self.nfp, mpol, ntor, self.lasym)

    self.RBC = RBC.transpose()
    self.ZBS = ZBS.transpose()
    if self.lasym:
        self.RBS = RBS.transpose()
        self.ZBC = ZBC.transpose()
    else:
        self.RBS = RBS
        self.ZBC = ZBC
    RBC = self.RBC
    RBS = self.RBS
    ZBC = self.ZBC
    ZBS = self.ZBS

    # Write to SPEC file
    file_object = open(filename,"w+")
    file_object.write("! This &phisicslist namelist was generated by pyQSC: github.com/landreman/pyQSC\n")
    file_object.write("! Date: "+datetime.now().strftime("%B %d, %Y")+", Time: "+datetime.now().strftime("%H:%M:%S")+" UTC"+datetime.now().astimezone().strftime("%z")+"\n")
    file_object.write('! Near-axis parameters:  radius r = '+str(r)+', etabar = '+str(self.etabar)+'\n')
    file_object.write('! nphi = '+str(self.nphi)+', order = '+self.order+', sigma0 = '+str(self.sigma0)+', I2 = '+str(self.I2)+', B0 = '+str(self.B0)+'\n')
    file_object.write('! Resolution parameters: ntheta = '+str(ntheta)+', mpol = '+str(mpol)+', ntor = '+str(ntor)+'\n')
    file_object.write('&physicslist\n')
    file_object.write('  Igeometry   = 3\n')
    file_object.write('  Istellsym   = '+str(0 if self.lasym else 1)+'\n')
    file_object.write('  Lfreebound  = 0\n')
    file_object.write('  phiedge     = '+str(phiedge)+'\n')
    file_object.write('  curtor      = '+str(curtor)+'\n')
    file_object.write('  curpol      = '+str(0)+'\n') # when is it not zero?
    file_object.write('  gamma       = '+str(0)+'\n') # when is it not zero?
    file_object.write('  Nfp         = '+str(self.nfp)+'\n')
    file_object.write('  Nvol        = '+str(1)+'\n')
    file_object.write('  Mpol        = '+str(mpol)+'\n')
    file_object.write('  Ntor        = '+str(min(ntor,ntorMax))+'\n')
    file_object.write('  Lrad        = '+str(params["Lrad"])+'\n') # ask user
    # from here down, I am not sure how to change these parameters
    file_object.write('  tflux       = 1.0\n')
    file_object.write('  pflux       = 0.0\n')
    file_object.write('  helicity    = -0.1\n')
    file_object.write('  pscale      = 0.0\n')
    file_object.write('  Ladiabatic  = 0\n')
    file_object.write('  pressure    = 0\n') # how does this relate to p2?
    file_object.write('  adiabatic   = 1.0  0.0\n')
    file_object.write('  mu          = 0.0\n')
    file_object.write('  Lconstraint = 0\n')
    file_object.write('  pl          =                       0                      0                      0\n')
    file_object.write('  ql          =                       0                      0                      0\n')
    file_object.write('  pr          =                       0                      0                      0\n')
    file_object.write('  qr          =                       0                      0                      0\n')
    file_object.write('  iota        =                       0.000000000000000E+00  2.809417939338480E-01  3.050000000000000E-01\n')
    file_object.write('  lp          =                       0                      0                      0\n')
    file_object.write('  lq          =                       0                      0                      0\n')
    file_object.write('  rp          =                       0                      0                      0\n')
    file_object.write('  rq          =                       0                      0                      0\n')
    file_object.write('  oita        =                       0.000000000000000E+00  2.809417939338480E-01  3.050000000000000E-01\n')
    file_object.write('  mupftol     =   1.000000000000000E-12\n')
    file_object.write('  mupfits     =         128\n\n')
    # Until here
    file_object.write('!----- Axis Shape -----\n')
    file_object.write("rac(0:"+str(len(self.rc)-1)+")="+", ".join([str(elem) for elem in self.rc])+"\n")
    file_object.write("zas(0:"+str(len(self.zs)-1)+")="+", ".join([str(-elem) for elem in self.zs])+"\n\n")
    file_object.write('!----- Boundary Parameters -----\n')
    text = ""
    nmodes = params["nmodes"]
    rbc=[0 for i in range(len(RBC))]
    zbs=[0 for i in range(len(RBC))]
    nmodesTot=len(RBC[1])
    for count in range(len(RBC)):
        if count==0:
            val = next((index for index,value in enumerate(RBC[0]) if value != 0), None)
            rbc[count]=RBC[count][val:val+2*nmodes]
            val = next((index for index,value in enumerate(ZBS[0]) if value != 0), None)
            zbs[count]=ZBS[count][val-1:val-1+2*nmodes]
        else:
            rbc[count]=RBC[count][int((nmodesTot-1)/2-nmodes):int((nmodesTot-1)/2+nmodes)]
            zbs[count]=ZBS[count][int((nmodesTot-1)/2-nmodes):int((nmodesTot-1)/2+nmodes)]
    for countn in range(len(rbc)-1):
        if countn==0:
            for countm in range(nmodes):
                text=text+"rbc("+str(countm)+","+str(countn)+")= "+str(rbc[countn][countm])+", zbs("+str(countm)+","+str(countn)+")= "+str(zbs[countn][countm])+",\n"
        elif countn==len(rbc)-2:
            for countm in range(2*nmodes):
                if countm==2*nmodes-1:
                    text=text+"rbc("+str(countm-nmodes)+","+str(countn)+")= "+str(rbc[countn][countm])+", zbs("+str(countm-nmodes)+","+str(countn)+")= "+str(zbs[countn][countm])+"\n"
                else:
                    text=text+"rbc("+str(countm-nmodes)+","+str(countn)+")= "+str(rbc[countn][countm])+", zbs("+str(countm-nmodes)+","+str(countn)+")= "+str(zbs[countn][countm])+",\n"
        else:
            for countm in range(2*nmodes):
                text=text+"rbc("+str(countm-nmodes)+","+str(countn)+")= "+str(rbc[countn][countm])+", zbs("+str(countm-nmodes)+","+str(countn)+")= "+str(zbs[countn][countm])+",\n"
    file_object.write(text)
    file_object.write('Rwc(0,0)    =  0.000000000000000E+00 Zws(0,0)    =  0.000000000000000E+00 Rws(0,0)    =  0.000000000000000E+00 Zwc(0,0)    =  0.000000000000000E+00 Bns(0,0)    =  0.000000000000000E+00 Bnc(0,0)    =  0.000000000000000E+00\n')
    file_object.write('Rwc(1,0)    =  0.000000000000000E+00 Zws(1,0)    =  0.000000000000000E+00 Rws(1,0)    =  0.000000000000000E+00 Zwc(1,0)    =  0.000000000000000E+00 Bns(1,0)    =  0.000000000000000E+00 Bnc(1,0)    =  0.000000000000000E+00\n')
    file_object.write('/\n')
    file_object.write('&numericlist\n')
    file_object.write(' Linitialize =   1\n')
    file_object.write(' Ndiscrete   =   2\n')
    file_object.write(' Nquad       =  -1\n')
    file_object.write(' iMpol       =  -4\n')
    file_object.write(' iNtor       =  -4\n')
    file_object.write(' Lsparse     =   0\n')
    file_object.write(' Lsvdiota    =   0\n')
    file_object.write(' imethod     =   3\n')
    file_object.write(' iorder      =   2\n')
    file_object.write(' iprecon     =   1\n')
    file_object.write(' iotatol     =  -1.0\n')
    file_object.write(' lrzaxis = 2\n')
    file_object.write('/\n')
    file_object.write('&locallist\n')
    file_object.write(' LBeltrami   =   4\n')
    file_object.write(' Linitgues   =   1\n')
    file_object.write('/\n')
    file_object.write('&globallist\n')
    file_object.write(' Lfindzero   =   2\n')
    file_object.write(' escale      =   0.0\n')
    file_object.write(' pcondense   =   4.0\n')
    file_object.write(' forcetol    =   1.0E-12\n')
    file_object.write(' c05xtol     =   1.0E-12\n')
    file_object.write(' c05factor   =   1.0E-04\n')
    file_object.write(' LreadGF     =   F\n')
    file_object.write(' opsilon     =   1.0E+00\n')
    file_object.write(' epsilon     =   1.0\n')
    file_object.write(' upsilon     =   1.0\n')
    file_object.write('/\n')
    file_object.write('&diagnosticslist\n')
    file_object.write(' odetol      =   1.0E-05\n')
    file_object.write(' absreq      =   1.0-08\n')
    file_object.write(' relreq      =   1.0E-08\n')
    file_object.write(' absacc      =   1.0E-04\n')
    file_object.write(' epsr        =   1.0E-08\n')
    file_object.write(' ! Increase nPpts and nPtrj if you want to make a Poincare plot.\n')
    file_object.write(' nPpts       =   '+str(params["nPpts"])+'\n')
    file_object.write(' nPtrj       =   '+str(params["nPtrj"])+'\n')
    file_object.write(' LHevalues   =   F\n')
    file_object.write(' LHevectors  =   F\n')
    file_object.write('/\n')
    file_object.write('&screenlist\n')
    file_object.write(' Wpp00aa = T\n')
    file_object.write('/\n')
    file_object.close()
