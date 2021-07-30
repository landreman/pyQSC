"""
This module contains a function to plot a near-axis surface.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as clr
from scipy.interpolate import interp2d
from matplotlib.colors import LightSource

def set_axes_equal(ax):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def createsubplot(ax, R_2D, Z_2D, nfp, colormap, elev=90, azim=45, dist=7, **kwargs):
    '''
    Construct the surface over a phi=[0,2*pi] domain given
    a surface in cylindrical coordinates R_2D, Z_2D with
    phi=[0,2*pi/Nfp]. A matplotlib figure with elements fig, ax
    must have been previously created and ax is given as input.
    '''
    for i in range(nfp):
        phi = np.linspace(i*2*np.pi/nfp,(i+1)*2*np.pi/nfp,R_2D.shape[1])
        x_2D = R_2D*np.cos(phi)
        y_2D = R_2D*np.sin(phi)
        z_2D = Z_2D
        ax.plot_surface(x_2D, y_2D, z_2D, facecolors = colormap, rstride=1, cstride=1, antialiased=False, linewidth=0, alpha=1, shade=False, **kwargs)
    set_axes_equal(ax)
    ax.set_axis_off()
    ax.dist = dist
    ax.elev = elev
    ax.azim = azim

def plot(self,r=0.1,ntheta_plot=40,nphi_plot=130,ntheta_fourier=16,nsections=8,save=None, colormap=None, azim_default=None,**kwargs):
    """
    Creates 2 matplotlib figures:
        - A plot with several poloidal planes at the specified radius r with the
         corresponding location of the magnetic axis and label using plt.plot
        - A 3D plot with the flux surface and the magnetic field strength
         on the surface using plot_surface
    Args:
      r (float): near-axis radius r where to create the surface
      ntheta_plot (int): Number of grid points to plot in the poloidal angle.
      nphi_plot   (int): Number of grid points to plot in the toroidal angle.
      ntheta_fourier (int): Resolution in the Fourier transform to cylindrical coordinates
      nsections (int): Number of poloidal planes to show.
      save (str): Filename prefix for the png files to save
      colormap (cmap): Custom colormap for the 3D plots
      azim_default: Default azimuthal angle for the three subplots in the 3D surface plot
      kwargs: Any additional key-value pairs to pass to matplotlib's plot_surface.

    This function can generate figures like this:

    .. image:: 3dplot.png
       :width: 270

    .. image:: poloidalplot.png
       :width: 200
    """

    # Obtain the surface shape in cylindrical coordinates
    R_2D, Z_2D, phi0_2D = self.Frenet_to_cylindrical(r, ntheta_fourier)
    # Make it periodic
    R_2D = np.append(R_2D,[R_2D[0,:]],0)
    R_2D = np.append(R_2D,np.array([R_2D[:,0]]).transpose(),1)
    Z_2D = np.append(Z_2D,[Z_2D[0,:]],0)
    Z_2D = np.append(Z_2D,np.array([Z_2D[:,0]]).transpose(),1)
    phi0_2D = np.append(phi0_2D,[phi0_2D[0,:]],0)
    phi0_2D = np.append(phi0_2D,np.array([phi0_2D[:,0]]).transpose(),1)
    # Arrays of original thetas and phis
    theta1d = np.linspace(0, 2 * np.pi, ntheta_fourier+1, endpoint=False)
    phi1d   = np.linspace(0, 2 * np.pi / self.nfp, self.nphi+1, endpoint=False)
    # Arrays of thetas and phis for plots
    theta1dplot  = np.linspace(0, 2 * np.pi, ntheta_plot)
    phi1dplot_RZ = np.linspace(0, 2 * np.pi / self.nfp, nsections, endpoint=False)
    phi1dplot    = np.linspace(0, 2 * np.pi / self.nfp, nphi_plot)
    # Splines interpolants of R_2D and Z_2D
    R_2D_spline = interp2d(phi1d, theta1d, R_2D, kind='cubic')
    Z_2D_spline = interp2d(phi1d, theta1d, Z_2D, kind='cubic')
    phi0_2D_spline = interp2d(phi1d, theta1d, phi0_2D, kind='cubic')
    R_2D_interp = R_2D_spline(phi1dplot,theta1dplot)
    Z_2D_interp = Z_2D_spline(phi1dplot,theta1dplot)

    ## Poloidal plot
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax  = plt.gca()
    for phi in phi1dplot_RZ:
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
        plt.plot(self.R0_func(np.mean(phi0_2D_spline(phi,theta1dplot))),self.Z0_func(np.mean(phi0_2D_spline(phi,theta1dplot))),marker="x",linewidth=2,label=label,color=color)
        plt.plot(R_2D_spline(phi,theta1dplot).flatten(),Z_2D_spline(phi,theta1dplot).flatten(),color=color)
    plt.xlabel('R (meters)')
    plt.ylabel('Z (meters)')
    plt.legend()
    plt.tight_layout()
    ax.set_aspect('equal')
    if save!=None:
        fig.savefig(save+'_poloidal.png')
    # plt.show()
    # exit()

    ## 3D plot
    # Set the default azimuthal angle of view in the 3D plot
    if azim_default == None:
        if self.helicity == 0:
            azim_default = 90
        else:
            azim_default = 45
    # Define the magnetic field modulus and create its theta,phi array
    def Bf(r,theta,phi):
        thetaN = theta-(self.iota-self.iotaN)*phi
        return self.B0*(1+r*self.etabar*np.cos(thetaN))
    phi2D, theta2D = np.meshgrid(phi1dplot,theta1dplot)
    Bmag=Bf(r,theta2D,phi2D)
    norm = clr.Normalize(vmin=Bmag.min(), vmax=Bmag.max())
    # Create a color map similar to the plots in the quasisymmetry
    # papers 2019-2021 if a colormap is not provided
    if colormap==None:
        cmap = clr.LinearSegmentedColormap.from_list('qs_papers',['#4423bb','#4940f4','#2e6dff','#0097f2','#00bacc','#00cb93','#00cb93','#7ccd30','#fbdc00','#f9fc00'], N=256)
        ls = LightSource(azdeg=0, altdeg=10)
        cmap_plot = ls.shade(Bmag, cmap, norm=norm)
        # cmap_plot = cmap(norm(Bmag))
    # Create the 3D figure
    fig = plt.figure(constrained_layout=False, figsize=(4.5, 8))
    gs1 = fig.add_gridspec(nrows=3, ncols=1, top=1.02, bottom=-0.3, left=0., right=0.85, hspace=0.0, wspace=0.0)
    ax = fig.add_subplot(gs1[0, 0], projection='3d')
    createsubplot(ax, R_2D_interp, Z_2D_interp, self.nfp, cmap_plot, elev=90, azim=azim_default, **kwargs)
    # create_subplot(90,azim_default)
    gs2 = fig.add_gridspec(nrows=3, ncols=1, top=1.09, bottom=-0.3, left=0., right=0.85, hspace=0.0, wspace=0.0)
    ax = fig.add_subplot(gs2[1, 0], projection='3d')
    createsubplot(ax, R_2D_interp, Z_2D_interp, self.nfp, cmap_plot, elev=30, azim=azim_default, **kwargs)
    # create_subplot(30,azim_default)
    gs3 = fig.add_gridspec(nrows=3, ncols=1, top=1.12, bottom=-0.15, left=0., right=0.85, hspace=0.0, wspace=0.0)
    ax = fig.add_subplot(gs3[2, 0], projection='3d')
    createsubplot(ax, R_2D_interp, Z_2D_interp, self.nfp, cmap_plot, elev=5,  azim=azim_default, **kwargs)
    # create_subplot(0,azim_default)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])
    cbar = plt.colorbar(m, cax=cbar_ax)
    cbar.ax.set_title(r'$|B| [T]$')
    if save!=None:
        fig.savefig(save+'3D.png')
    plt.show()