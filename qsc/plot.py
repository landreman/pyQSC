"""
This module contains a function to plot a near-axis surface.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline as spline
import matplotlib.pyplot as plt
from matplotlib import cm

def plot(self,rPlot=0.1,nphi=60,ntheta=40,nsections=9,saveFile=None):

    # Create splines interpolants for the quantities used in the plots
    def Raxisf(phi): return sum([self.rc[i]*np.cos(i*self.nfp*phi) for i in range(len(self.rc))])
    def Zaxisf(phi): return sum([self.zs[i]*np.sin(i*self.nfp*phi) for i in range(len(self.zs))])
    def tangentR(phi):
        sp=spline(self.phi, self.tangent_cylindrical[:,0], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def tangentZ(phi):
        sp=spline(self.phi, self.tangent_cylindrical[:,2], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def normalR(phi):
        sp=spline(self.phi, self.normal_cylindrical[:,0], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def normalZ(phi):
        sp=spline(self.phi, self.normal_cylindrical[:,2], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/self.nfp))
    def binormalR(phi):
        sp=spline(self.phi, self.binormal_cylindrical[:,0], k=3, s=0)
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
        r1 = (          X1cF(phi)*np.cos(thetaN  )+X1sF(phi)*np.sin(thetaN)  )*normalR(phi)+(          Y1cF(phi)*np.cos(thetaN  )+Y1sF(phi)*np.sin(thetaN)  )*binormalR(phi)
        if self.order!='r1':
            r2 = (X20F(phi)+X2cF(phi)*np.cos(2*thetaN)+X2sF(phi)*np.sin(2*thetaN))*normalR(phi)+(Y20F(phi)+Y2cF(phi)*np.cos(2*thetaN)+Y2sF(phi)*np.sin(2*thetaN))*binormalR(phi)+(Z20F(phi)+Z2cF(phi)*np.cos(2*thetaN)+Z2sF(phi)*np.sin(2*thetaN))*tangentR(phi)
        else:
            r2 = 0
        return r0+r*r1+r**2*r2
    def zSurf(r,phi,theta):
        thetaN = theta-(self.iota-self.iotaN)*phi
        z0 = Zaxisf(phi)
        z1 = (          X1cF(phi)*np.cos(thetaN)  +X1sF(phi)*np.sin(thetaN)  )*normalZ(phi)+(          Y1cF(phi)*np.cos(thetaN)  +Y1sF(phi)*np.sin(thetaN)  )*binormalZ(phi)
        if self.order!='r1':
            z2 = (X20F(phi)+X2cF(phi)*np.cos(2*thetaN)+X2sF(phi)*np.sin(2*thetaN))*normalZ(phi)+(Y20F(phi)+Y2cF(phi)*np.cos(2*thetaN)+Y2sF(phi)*np.sin(2*thetaN))*binormalZ(phi)+(Z20F(phi)+Z2cF(phi)*np.cos(2*thetaN)+Z2sF(phi)*np.sin(2*thetaN))*tangentZ(phi)
        else:
            z2 = 0
        return z0+r*z1+r**2*z2

    # Create one plot with several cuts at different toroidal planes
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax  = plt.gca()
    theta = np.linspace(0,2*np.pi,ntheta)
    phi1D = np.linspace(0,2*np.pi/self.nfp,nsections)
    for phi in phi1D:
        rSurfi=rSurf(rPlot,phi,theta)
        zSurfi=zSurf(rPlot,phi,theta)
        if phi*self.nfp/(2*np.pi)==0:
            label = r'$\phi$=0'
        elif phi*self.nfp/(2*np.pi)==0.25:
            label = r'$\phi={\pi}/$'+str(2*self.nfp)
        elif phi*self.nfp/(2*np.pi)==0.5:
            label = r'$\phi=\pi/$'+str(self.nfp)
        elif phi*self.nfp/(2*np.pi)==0.75:
            label = r'$\phi={3\pi}/$'+str(2*self.nfp)
        elif phi*self.nfp/(2*np.pi)==1.0:
            continue
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
    if saveFile!=None:
        fig.savefig(saveFile+'_poloidal.pdf')

    # Create 3D plot
    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    def Bf(r,phi,theta):
        if self.order=='r1':
            return self.B0*(1+r*self.etabar*np.cos(theta-(self.iota-self.iotaN)*phi))
        else:
            return self.B0*(1+r*self.etabar*np.cos(theta-(self.iota-self.iotaN)*phi))+r*r*(B20F(phi)+self.B2c*np.cos(theta-(self.iota-self.iotaN)*phi)+self.B2s*np.sin(theta-(self.iota-self.iotaN)*phi))
    theta1d = np.linspace(0, 2 * np.pi, ntheta)
    phi1d   = np.linspace(0, 2 * np.pi, nphi)
    phi2D, theta2D = np.meshgrid(phi1d, theta1d)
    rs=rSurf(rPlot,phi2D,theta2D)
    zs=zSurf(rPlot,phi2D,theta2D)
    Xsurf=rs*np.cos(phi2D)
    Ysurf=rs*np.sin(phi2D)
    Zsurf=zs
    Bmag=Bf(rPlot,phi2D,theta2D)
    B_rescaled = (Bmag - Bmag.min()) / (Bmag.max() - Bmag.min())
    ax.plot_surface(Xsurf, Ysurf, Zsurf, facecolors = cm.jet(B_rescaled), rstride=1, cstride=1, antialiased=False, linewidth=0, alpha=0.25)
    ax.auto_scale_xyz([Xsurf.min(), Xsurf.max()], [Xsurf.min(), Xsurf.max()], [Xsurf.min(), Xsurf.max()])   
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    plt.tight_layout()
    if saveFile!=None:
        fig.savefig(saveFile+'3D.pdf')

    plt.show()