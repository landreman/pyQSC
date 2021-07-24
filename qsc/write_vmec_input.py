"""
This module contains the routines to output a
near-axis boundary to a VMEC input file
"""
import numpy as np
from .util import mu0
from scipy.interpolate import UnivariateSpline as spline
from scipy.optimize import root_scalar

def to_vmec(self, filename, rVMEC=0.1, vmec_template_filename=None):
    """
    Output a near-axis boundary to a VMEC input file
    """
    if vmec_template_filename==None:
        delt=0.9
        niter=2500
        nstep=200
        tcon0=2.
        mpol=10
        ntorMax=14
        ns_array=[16,49,101]
        ftol_array=[1e-13,1e-12,1e-11]
        niter_array=[1000,1000,1500]

    r = rVMEC
    phiedge = np.pi * r * r * self.spsi * self.Bbar

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

    # The output is not stellarator-symmetric if (1) R0s is nonzero, (2) Z0c is nonzero, or (3) sigma_initial is nonzero
    lasym = np.max(np.abs(self.rs))>0 or np.max(np.abs(self.zc))>0 or np.abs(self.sigma0)>0

    # We should be able to resolve (N_phi-1)/2 modes (note integer division!), but in case N_phi is very large, don't attempt more than the vmec arrays can handle.
    ntord = 100 # maximum number of mode numbers VMEC can handle
    ntor = min((self.nphi - 1) / 2, ntord)
    mpold = 101
    mpol1d = mpold - 1
    ntor1d = 1 + ntord

    # Set axis shape
    # To convert sin(...) modes to vmec, we introduce a minus sign. This is because in vmec,
    # R and Z ~ sin(m theta - n phi), which for m=0 is sin(-n phi) = -sin(n phi).
    raxis_cc = self.rc
    raxis_cs = -self.rs
    zaxis_cc = self.zc
    zaxis_cs = -self.zs

    # Compute RBC, ZBS
    finite_r_nonlinear_N_theta = 20
    N_theta = finite_r_nonlinear_N_theta
    N_phi_conversion = self.nphi
    theta = np.linspace(0,2*np.pi,N_theta,endpoint=False)
    phi_conversion = np.linspace(0,2*np.pi/self.nfp,N_phi_conversion,endpoint=False)

    def R0_spline(phi): return sum([self.rc[i]*np.cos(i*self.nfp*phi) for i in range(len(self.rc))])
    def z0_spline(phi): return sum([self.zs[i]*np.sin(i*self.nfp*phi) for i in range(len(self.zs))])
    def normal_R_spline(phi):
        sp=spline(self.phi, self.normal_cylindrical[:,0], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def normal_phi_spline(phi):
        sp=spline(self.phi, self.normal_cylindrical[:,1], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def normal_z_spline(phi):
        sp=spline(self.phi, self.normal_cylindrical[:,2], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def binormal_R_spline(phi):
        sp=spline(self.phi, self.binormal_cylindrical[:,0], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def binormal_phi_spline(phi):
        sp=spline(self.phi, self.binormal_cylindrical[:,1], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def binormal_z_spline(phi):
        sp=spline(self.phi, self.binormal_cylindrical[:,2], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))

    rootSolve_abserr = 1.0e-30
    rootSolve_relerr = 1.0e-30

    # If helicity is nonzero, then the original X1s/X1c/Y1s/Y1c variables are defined with respect to a "poloidal" angle that
    # is actually helical, with the theta=0 curve wrapping around the magnetic axis as you follow phi around toroidally. Therefore
    # here we convert to an untwisted poloidal angle, such that the theta=0 curve does not wrap around the axis.
    if self.helicity == 0:
        X1s_untwisted = 0
        X1c_untwisted = self.X1c
        Y1s_untwisted = self.Y1s
        Y1c_untwisted = self.Y1c
    else:
        angle = -self.helicity * self.nfp * self.varphi
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        X1s_untwisted = self.X1s *   cosangle  + self.X1c * sinangle
        X1c_untwisted = self.X1s * (-sinangle) + self.X1c * cosangle
        Y1s_untwisted = self.Y1s *   cosangle  + self.Y1c * sinangle
        Y1c_untwisted = self.Y1s * (-sinangle) + self.Y1c * cosangle

    def Frenet_to_cylindrical_residual_func(phi0):
        # Given a point on the axis with toroidal angle phi0, compute phi for the associated point at r>0,
        # and find the difference between this phi and the target value of phi.

        sinphi0 = np.sin(phi0)
        cosphi0 = np.cos(phi0)
        R0_at_phi0   = R0_spline(phi0)
        X_at_phi0    = X_spline(phi0)
        Y_at_phi0    = Y_spline(phi0)
        normal_R     = normal_R_spline(phi0)
        normal_phi   = normal_phi_spline(phi0)
        binormal_R   = binormal_R_spline(phi0)
        binormal_phi = binormal_phi_spline(phi0)

        normal_x   =   normal_R * cosphi0 -   normal_phi * sinphi0
        normal_y   =   normal_R * sinphi0 +   normal_phi * cosphi0
        binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
        binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0

        total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
        total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y

        Frenet_to_cylindrical_residual = np.arctan2(total_y, total_x) - phi_target
        # We expect the residual to be less than pi in absolute value, so if it is not, the reason must be the branch cut:
        if (Frenet_to_cylindrical_residual >  np.pi): Frenet_to_cylindrical_residual = Frenet_to_cylindrical_residual - 2*np.pi
        if (Frenet_to_cylindrical_residual < -np.pi): Frenet_to_cylindrical_residual = Frenet_to_cylindrical_residual + 2*np.pi
        return Frenet_to_cylindrical_residual

    def Frenet_to_cylindrical_1_point(phi0):
        # Given a point on the axis with toroidal angle phi0, compute R and z for the associated point at r>0.
 
        sinphi0 = np.sin(phi0)
        cosphi0 = np.cos(phi0)
        R0_at_phi0   = R0_spline(phi0)
        z0_at_phi0   = z0_spline(phi0)
        X_at_phi0    = X_spline(phi0)
        Y_at_phi0    = Y_spline(phi0)
        normal_R     = normal_R_spline(phi0)
        normal_phi   = normal_phi_spline(phi0)
        normal_z     = normal_z_spline(phi0)
        binormal_R   = binormal_R_spline(phi0)
        binormal_phi = binormal_phi_spline(phi0)
        binormal_z   = binormal_z_spline(phi0)

        normal_x   =   normal_R * cosphi0 -   normal_phi * sinphi0
        normal_y   =   normal_R * sinphi0 +   normal_phi * cosphi0
        binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
        binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0

        total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
        total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y

        total_z = z0_at_phi0           + X_at_phi0 * normal_z + Y_at_phi0 * binormal_z
        total_R = np.sqrt(total_x * total_x + total_y * total_y)
        return total_R, total_z

    ## MAIN SCRIPT
    R_2D = np.zeros((N_theta,N_phi_conversion))
    z_2D = np.zeros((N_theta,N_phi_conversion))
    phi0_2D = np.zeros((N_theta,N_phi_conversion))
    for j_theta in range(N_theta):
       costheta = np.cos(theta[j_theta])
       sintheta = np.sin(theta[j_theta])
       X_at_this_theta = r * (X1c_untwisted * costheta + X1s_untwisted * sintheta)
       Y_at_this_theta = r * (Y1c_untwisted * costheta + Y1s_untwisted * sintheta)
       def X_spline(phi):
          sp=spline(self.phi, X_at_this_theta, k=3, s=0)
          return sp(np.mod(phi,2*np.pi/self.nfp))
       def Y_spline(phi):
          sp=spline(self.phi, Y_at_this_theta, k=3, s=0)
          return sp(np.mod(phi,2*np.pi/self.nfp))
       for j_phi in range(N_phi_conversion):
          # Solve for the phi0 such that r0 + X1 n + Y1 b has the desired phi
          phi_target = phi_conversion[j_phi]
          phi0_rootSolve_min = phi_target - 1.0 / self.nfp
          phi0_rootSolve_max = phi_target + 1.0 / self.nfp
        #   print('-------------')
        #   print(Frenet_to_cylindrical_residual_func(phi0_rootSolve_min))
        #   print(Frenet_to_cylindrical_residual_func(phi0_rootSolve_max))
        #   print(Frenet_to_cylindrical_residual_func(phi_target))
          res = root_scalar(Frenet_to_cylindrical_residual_func, bracket=[phi0_rootSolve_min, phi0_rootSolve_max], x0=phi_target)
        #   print(res)
        #   print('-------------')
          ##### SOLVE FOR THE ZEROS OF Frenet_to_cylindrical_residual !!!!!!!
        #   call quasisymmetry_fzero(Frenet_to_cylindrical_residual, phi0_rootSolve_min, phi0_rootSolve_max, phi_target, &
        #        rootSolve_relerr, rootSolve_abserr, fzeroFlag)
          # Note: fzero returns its answer in phi0_rootSolve_min
          phi0_solution = res.root

          final_R, final_z = Frenet_to_cylindrical_1_point(phi0_solution)
          R_2D[j_theta,j_phi] = final_R
          z_2D[j_theta,j_phi] = final_z
          phi0_2D[j_theta,j_phi] = phi0_solution

    # Fourier transform the result.
    # This is not a rate-limiting step, so for clarity of code, we don't bother with an FFT.
    mpol = int(min(N_theta          / 2, mpol1d))
    ntor = int(min(N_phi_conversion / 2, ntord))
    mpol_nonzero = mpol
    RBC = np.zeros((int(2*ntor+1),int(mpol+1)))
    RBS = np.zeros((int(2*ntor+1),int(mpol+1)))
    ZBC = np.zeros((int(2*ntor+1),int(mpol+1)))
    ZBS = np.zeros((int(2*ntor+1),int(mpol+1)))
    factor = 2 / (N_theta * N_phi_conversion)
    for j_phi in range(N_phi_conversion):
        for j_theta in range(N_theta):
            for m in range(mpol):
                nmin = -ntor
                if m==0: nmin = 1
                for n in range(nmin, ntor):
                    angle = m * theta[j_theta] - n * self.nfp * phi_conversion[j_phi]
                    sinangle = np.sin(angle)
                    cosangle = np.cos(angle)
                    factor2 = factor
                    # The next 2 lines ensure inverse Fourier transform(Fourier transform) = identity
                    if np.mod(N_theta,2) == 0 and m  == (N_theta/2): factor2 = factor2 / 2
                    if np.mod(N_phi_conversion,2) == 0 and abs(n) == (N_phi_conversion/2): factor2 = factor2 / 2
                    RBC[n,m] = RBC[n,m] + R_2D[j_theta, j_phi] * cosangle * factor2
                    RBS[n,m] = RBS[n,m] + R_2D[j_theta, j_phi] * sinangle * factor2
                    ZBC[n,m] = ZBC[n,m] + z_2D[j_theta, j_phi] * cosangle * factor2
                    ZBS[n,m] = ZBS[n,m] + z_2D[j_theta, j_phi] * sinangle * factor2
    RBC[0,0] = np.sum(R_2D) / (N_theta * N_phi_conversion)
    ZBC[0,0] = np.sum(z_2D) / (N_theta * N_phi_conversion)

    if lasym == False:
        RBS = 0
        ZBC = 0

    # Write to VMEC file
    File_object = open(filename,"w+")
    File_object.write("! This &INDATA namelist was generated by pyQSC\n")
    if vmec_template_filename!=None:    File_object.write("! Based on template file ",vmec_template_filename+'\n')
    File_object.write("! r ="+str(r)+'\n')
    File_object.write('!----- Runtime Parameters -----\n')
    File_object.write('&INDATA\n')
    File_object.write('  DELT = '+str(delt)+'\n')
    File_object.write('  NITER = '+str(niter)+'\n')
    File_object.write('  NSTEP = '+str(nstep)+'\n')
    File_object.write('  TCON0 = '+str(tcon0)+'\n')
    File_object.write('  NS_ARRAY = '+str(ns_array)[1:-1]+'\n')
    File_object.write('  FTOL_ARRAY = '+str(ftol_array)[1:-1]+'\n')
    File_object.write('  NITER_ARRAY = '+str(niter_array)[1:-1]+'\n')
    File_object.write('!----- Grid Parameters -----\n')
    File_object.write('  LASYM = '+str(lasym)+'\n')
    File_object.write('  NFP = '+str(self.nfp)+'\n')
    File_object.write('  MPOL = '+str(mpol)+'\n')
    File_object.write('  NTOR = '+str(min(ntor,ntorMax))+'\n')
    File_object.write('  PHIEDGE = '+str(phiedge)+'\n')
    File_object.write('!----- Pressure Parameters -----\n')
    File_object.write('  PRES_SCALE = '+str(pres_scale)+'\n')
    File_object.write("  PMASS_TYPE = '"+pmass_type+"'\n")
    File_object.write('  AM = '+str(am)[1:-1]+'\n')
    File_object.write('!----- Current/Iota Parameters -----\n')
    File_object.write('  CURTOR = '+str(curtor)+'\n')
    File_object.write('  NCURR = '+str(ncurr)+'\n')
    File_object.write("  PCURR_TYPE = '"+pcurr_type+"'\n")
    File_object.write('  AC = '+str(ac)[1:-1]+'\n')
    File_object.write('!----- Axis Parameters -----\n')
    File_object.write('  RAXIS_CC = '+str(raxis_cc)[1:-1]+'\n')
    if lasym==True:
        File_object.write('  RAXIS_CS = '+str(raxis_cs)[1:-1]+'\n')
        File_object.write('  ZAXIS_CC = '+str(zaxis_cc)[1:-1]+'\n')
    File_object.write('  ZAXIS_CS = '+str(zaxis_cs)[1:-1]+'\n')
    File_object.write('!----- Boundary Parameters -----\n')
    for m in range(mpol - 1):
        for n in range(-ntor,ntor):
            if RBC[n,m]!=0 or ZBS[n,m]!=0:
                File_object.write(    '  RBC('+f"{n:03d}"+','+f"{m:03d}"+') = '+f"{RBC[n,m]:+.16e}"+',    ZBS('+f"{n:03d}"+','+f"{m:03d}"+') = '+f"{ZBS[n,m]:+.16e}"+'\n')
                if lasym == True:
                    File_object.write('  RBS('+f"{n:03d}"+','+f"{m:03d}"+') = '+f"{RBS[n,m]:+.16e}"+',    ZBC('+f"{n:03d}"+','+f"{m:03d}"+') = '+f"{ZBC[n,m]:+.16e}"+'\n')
    File_object.write('/\n')
    File_object.close()

    self.RBC = RBC
    self.RBS = RBS
    self.ZBC = ZBC
    self.ZBS = ZBS