"""
This module contains a function to plot a near-axis surface.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline as spline
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

def plot(self,r=0.14,nphi=80,ntheta=50,nsections=4,save=None,azim_default=None,**kwargs):
    """
    Creates 2 matplotlib figures:
        - A plot with several poloidal planes at the specified radius r with the
         corresponding location of the magnetic axis and label using plt.plot
        - A 3D plot with the flux surface and the magnetic field strength
         on the surface using plot_surface
    Args:
      r (float): near-axis radius r where to create the surface
      nphi   (int): Number of grid points in the toroidal angle.
      ntheta (int): Number of grid points in the poloidal angle.
      nsections (int): Number of poloidal planes to show.
      save (str): Filename prefix for the png files to save
      azim_default: Default azimuthal angle for the three subplots
       in the 3D surface plot
      kwargs: Any additional key-value pairs to pass to matplotlib's plot_surface.
    This function can generate figures like this:

    .. image:: 3dplot.png
       :width: 270

    .. image:: poloidalplot.png
       :width: 200
    """

    # Create splines interpolants for the quantities used in the plots
    def Raxisf(phi): return sum([self.rc[i]*np.cos(i*self.nfp*phi) for i in range(len(self.rc))])
    def Zaxisf(phi): return sum([self.zs[i]*np.sin(i*self.nfp*phi) for i in range(len(self.zs))])
    def tangentR(phi):
        sp=spline(self.phi, self.tangent_cylindrical[:,0], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def tangentphi(phi):
        sp=spline(self.phi, self.tangent_cylindrical[:,1], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def tangentZ(phi):
        sp=spline(self.phi, self.tangent_cylindrical[:,2], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def normalR(phi):
        sp=spline(self.phi, self.normal_cylindrical[:,0], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def normalphi(phi):
        sp=spline(self.phi, self.normal_cylindrical[:,1], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def normalZ(phi):
        sp=spline(self.phi, self.normal_cylindrical[:,2], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def binormalR(phi):
        sp=spline(self.phi, self.binormal_cylindrical[:,0], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def binormalphi(phi):
        sp=spline(self.phi, self.binormal_cylindrical[:,1], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def binormalZ(phi):
        sp=spline(self.phi, self.binormal_cylindrical[:,2], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def X1cF(phi):
        sp=spline(self.phi, self.X1c, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def X1sF(phi):
        sp=spline(self.phi, self.X1s, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def Y1cF(phi):
        sp=spline(self.phi, self.Y1c, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def Y1sF(phi):
        sp=spline(self.phi, self.Y1s, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def X20F(phi):
        sp=spline(self.phi, self.X20, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def X2cF(phi):
        sp=spline(self.phi, self.X2c, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def X2sF(phi):
        sp=spline(self.phi, self.X2s, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def Y20F(phi):
        sp=spline(self.phi, self.Y20, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def Y2cF(phi):
        sp=spline(self.phi, self.Y2c, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def Y2sF(phi):
        sp=spline(self.phi, self.Y2s, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def Z20F(phi):
        sp=spline(self.phi, self.Z20, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def Z2cF(phi):
        sp=spline(self.phi, self.Z2c, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def Z2sF(phi):
        sp=spline(self.phi, self.Z2s, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def B20F(phi):
        sp=spline(self.phi, self.B20, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    # Perform the transformation from a near-axis position vector
    # to cylindrical coordinates
    def rSurf(r,phi,theta):
        thetaN = theta-(self.iota-self.iotaN)*phi
        r0 = Raxisf(phi)
        r1 = (X1cF(phi)*np.cos(thetaN)+X1sF(phi)*np.sin(thetaN))*normalR(phi)\
            +(Y1cF(phi)*np.cos(thetaN)+Y1sF(phi)*np.sin(thetaN))*binormalR(phi)
        if self.order!='r1':
            r2 = (X20F(phi)+X2cF(phi)*np.cos(2*thetaN)+X2sF(phi)*np.sin(2*thetaN))*normalR(phi)\
                +(Y20F(phi)+Y2cF(phi)*np.cos(2*thetaN)+Y2sF(phi)*np.sin(2*thetaN))*binormalR(phi)\
                +(Z20F(phi)+Z2cF(phi)*np.cos(2*thetaN)+Z2sF(phi)*np.sin(2*thetaN))*tangentR(phi)
        else:
            r2 = 0
        return r0+r*r1+r**2*r2
    def phiSurf(r,phi,theta):
        thetaN = theta-(self.iota-self.iotaN)*phi
        phi0 = Raxisf(phi)
        phi1 = (X1cF(phi)*np.cos(thetaN)+X1sF(phi)*np.sin(thetaN))*normalphi(phi)\
              +(Y1cF(phi)*np.cos(thetaN)+Y1sF(phi)*np.sin(thetaN))*binormalphi(phi)
        if self.order!='r1':
            phi2 = (X20F(phi)+X2cF(phi)*np.cos(2*thetaN)+X2sF(phi)*np.sin(2*thetaN))*normalphi(phi)\
                  +(Y20F(phi)+Y2cF(phi)*np.cos(2*thetaN)+Y2sF(phi)*np.sin(2*thetaN))*binormalphi(phi)\
                  +(Z20F(phi)+Z2cF(phi)*np.cos(2*thetaN)+Z2sF(phi)*np.sin(2*thetaN))*tangentphi(phi)
        else:
            phi2 = 0
        return phi0+r*phi1+r**2*phi2
    def zSurf(r,phi,theta):
        thetaN = theta-(self.iota-self.iotaN)*phi
        z0 = Zaxisf(phi)
        z1 = (X1cF(phi)*np.cos(thetaN)+X1sF(phi)*np.sin(thetaN))*normalZ(phi)\
            +(Y1cF(phi)*np.cos(thetaN)+Y1sF(phi)*np.sin(thetaN))*binormalZ(phi)
        if self.order!='r1':
            z2 = (X20F(phi)+X2cF(phi)*np.cos(2*thetaN)+X2sF(phi)*np.sin(2*thetaN))*normalZ(phi)\
                +(Y20F(phi)+Y2cF(phi)*np.cos(2*thetaN)+Y2sF(phi)*np.sin(2*thetaN))*binormalZ(phi)\
                +(Z20F(phi)+Z2cF(phi)*np.cos(2*thetaN)+Z2sF(phi)*np.sin(2*thetaN))*tangentZ(phi)
        else:
            z2 = 0
        return z0+r*z1+r**2*z2

    # Create one plot with several cuts at different toroidal planes
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax  = plt.gca()
    theta = np.linspace(0,2*np.pi,ntheta)
    phi1D = np.linspace(0,2*np.pi/self.nfp,nsections,endpoint=False)
    for phi in phi1D:
        rSurfi=rSurf(r,phi,theta)
        zSurfi=zSurf(r,phi,theta)
        if phi*self.nfp/(2*np.pi)==0:
            label = r'$\phi$=0'
        elif phi*self.nfp/(2*np.pi)==0.25:
            label = r'$\phi={\pi}/$'+str(2*self.nfp)
        elif phi*self.nfp/(2*np.pi)==0.5:
            label = r'$\phi=\pi/$'+str(self.nfp)
        elif phi*self.nfp/(2*np.pi)==0.75:
            label = r'$\phi={3\pi}/$'+str(2*self.nfp)
        else:
            label = '_nolegend_'
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(rSurfi,zSurfi,linewidth=2,label=label,color=color)
        plt.plot(Raxisf(phi),Zaxisf(phi),'b+',color=color)
    plt.xlabel('R (meters)')
    plt.ylabel('Z')
    plt.legend()
    plt.tight_layout()
    ax.set_aspect('equal')
    if save!=None:
        fig.savefig(save+'_poloidal.png')

    # Create 3D plot
    def Bf(r,phi,theta):
        thetaN = theta-(self.iota-self.iotaN)*phi
        if self.order=='r1':
            return self.B0*(1+r*self.etabar*np.cos(thetaN))
        else:
            return self.B0*(1+r*self.etabar*np.cos(thetaN))+r*r*(B20F(phi)+self.B2c*np.cos(thetaN)+self.B2s*np.sin(thetaN))
    theta1d = np.linspace(0, 2 * np.pi, ntheta)
    phi1d   = np.linspace(0, 2 * np.pi, nphi)
    phi2D, theta2D = np.meshgrid(phi1d, theta1d)
    rs=rSurf(r,phi2D,theta2D)
    phis=phiSurf(r,phi2D,theta2D)
    Zsurf=zSurf(r,phi2D,theta2D)
    Xsurf=rs*np.cos(phi2D)-phis*np.sin(phi2D)
    Ysurf=rs*np.sin(phi2D)+phis*np.cos(phi2D)
    Bmag=Bf(r,phi2D,theta2D)
    norm = Normalize(vmin=Bmag.min(), vmax=Bmag.max())
    def create_subplot(elev=90,azim=45,dist=7):
        ax.plot_surface(Xsurf, Ysurf, Zsurf, facecolors = cm.plasma(norm(Bmag)), rstride=1, cstride=1, antialiased=False, linewidth=0, alpha=1, **kwargs)
        ax.auto_scale_xyz([Xsurf.min(), Xsurf.max()], [Xsurf.min(), Xsurf.max()], [Xsurf.min(), Xsurf.max()])
        ax.set_axis_off()
        ax.dist = dist
        ax.elev = elev
        ax.azim = azim
    if azim_default == None:
        if self.helicity == 0:
            azim_default = 135
        else:
            azim_default = 0
    fig = plt.figure(constrained_layout=False, figsize=(4.5, 8))
    gs1 = fig.add_gridspec(nrows=3, ncols=1, top=1.02, bottom=-0.3, left=0., right=0.85, hspace=0.0, wspace=0.0)
    ax = fig.add_subplot(gs1[0, 0], projection='3d')
    create_subplot(90,azim_default)
    gs2 = fig.add_gridspec(nrows=3, ncols=1, top=1.09, bottom=-0.3, left=0., right=0.85, hspace=0.0, wspace=0.0)
    ax = fig.add_subplot(gs2[1, 0], projection='3d')
    create_subplot(30,azim_default)
    gs3 = fig.add_gridspec(nrows=3, ncols=1, top=1.12, bottom=-0.15, left=0., right=0.85, hspace=0.0, wspace=0.0)
    ax = fig.add_subplot(gs3[2, 0], projection='3d')
    create_subplot(0,azim_default)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
    m = cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    m.set_array([])
    cbar = plt.colorbar(m, cax=cbar_ax)
    cbar.ax.set_title(r'$|B| [T]$')
    if save!=None:
        fig.savefig(save+'3D.png')

    plt.show()