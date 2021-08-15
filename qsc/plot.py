"""
This module contains a function to plot a near-axis surface.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as clr
from matplotlib.colors import LightSource
from .to_vmec import to_Fourier
from scipy.interpolate import interp2d
import matplotlib.ticker as tck

def set_axes_equal(ax):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Args:
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

def create_subplot(ax, x_2D_plot, y_2D_plot, z_2D_plot, colormap, elev=90, azim=45, dist=7, **kwargs):
    '''
    Construct the surface given a surface in cartesian coordinates
    x_2D_plot, y_2D_plot, z_2D_plot already with phi=[0,2*pi].
    A matplotlib figure with elements fig, ax
    must have been previously created.

    Args:
        ax: matplotlib figure instance
        x_2d_plot: 2D array for the x coordinates of the surface
        y_2d_plot: 2D array for the x coordinates of the surface
        z_2d_plot: 2D array for the x coordinates of the surface
        elev: elevation angle for the camera view
        azim: azim angle for the camera view
        distance: distance parameter for the camera view
    '''
    ax.plot_surface(x_2D_plot, y_2D_plot, z_2D_plot, facecolors = colormap, rstride=1, cstride=1, antialiased=False, linewidth=0, alpha=1., shade=False, **kwargs)
    set_axes_equal(ax)
    ax.set_axis_off()
    ax.dist = dist
    ax.elev = elev
    ax.azim = azim

def create_field_lines(qsc, alphas, X_2D, Y_2D, Z_2D, phimax=2*np.pi, nphi=500):
    '''
    Function to compute the (X, Y, Z) coordinates of field lines at
    several alphas, where alpha = theta-iota*varphi with (theta,varphi)
    the Boozer toroidal angles. This function relies on a 2D interpolator
    from the scipy library to smooth out the lines

    Args:
      qsc: instance of self
      alphas: array of field line labels alpha
      X_2D: 2D array for the x components of the surface
      Y_2D: 2D array for the y components of the surface
      Z_2D: 2D array for the z components of the surface
      phimax: maximum value for the field line following angle phi
      nphi: grid resolution for the output fieldline
    '''
    phi_array = np.linspace(0,phimax,nphi,endpoint=False)
    fieldline_X = np.zeros((len(alphas),nphi))
    fieldline_Y = np.zeros((len(alphas),nphi))
    fieldline_Z = np.zeros((len(alphas),nphi))
    [ntheta_RZ,nphi_RZ] = X_2D.shape
    phi1D   = np.linspace(0,2*np.pi,nphi_RZ)
    theta1D = np.linspace(0,2*np.pi,ntheta_RZ)
    X_2D_spline = interp2d(phi1D, theta1D, X_2D, kind='cubic')
    Y_2D_spline = interp2d(phi1D, theta1D, Y_2D, kind='cubic')
    Z_2D_spline = interp2d(phi1D, theta1D, Z_2D, kind='cubic')
    for i in range(len(alphas)):
        for j in range(len(phi_array)):
            phi_mod = np.mod(phi_array[j],2*np.pi)
            varphi0=qsc.nu_spline(phi_array[j])+2*phi_array[j]-phi_mod
            theta_fieldline=qsc.iota*varphi0+alphas[i]
            theta_fieldline_mod=np.mod(theta_fieldline,2*np.pi)
            fieldline_X[i,j] = X_2D_spline(phi_mod,theta_fieldline_mod)[0]
            fieldline_Y[i,j] = Y_2D_spline(phi_mod,theta_fieldline_mod)[0]
            fieldline_Z[i,j] = Z_2D_spline(phi_mod,theta_fieldline_mod)[0]
    return fieldline_X, fieldline_Y, fieldline_Z

def create_subplot_mayavi(mlab, R, alphas, x_2D_plot, y_2D_plot, z_2D_plot, fieldline_X, fieldline_Y, fieldline_Z, Bmag, degrees_array_x, degrees_array_z, shift_array):
    '''
    Plotting routine for a mayavi figure instance that plots
    both the surface and the field lines together. The number
    of surfaces to plot is specified by the length of the
    array degrees_array_x

    Args:
      mlab: mayavi package
      R: scipy rotation vector package
      alphas: array of field line labels alpha
      x_2D_plot: 2D array for the x components of the surface
      y_2D_plot: 2D array for the y components of the surface
      z_2D_plot: 2D array for the z components of the surface
      fieldline_X: 2D array for the x components of the field line
      fieldline_Y: 2D array for the x components of the field line
      fieldline_Z: 2D array for the x components of the field line
      Bmag: 2D array for the magnetic field modulus on the (theta,phi) meshgrid
      degrees_array_x: 1D array with the rotation angles in the x direction for each surface
      degrees_array_z: 1D array with the rotation angles in the z direction for each surface
      shift_array: 1D array with a shift in the y direction for each surface
    '''
    assert len(degrees_array_x) == len(degrees_array_z) == len(shift_array)
    for i in range(len(degrees_array_x)):
        # The surfaces and field lines are rotated first in the
        # z direction and then in the x direction
        rx= R.from_euler('x', degrees_array_x[i], degrees=True)
        rz= R.from_euler('z', degrees_array_z[i], degrees=True)
        # Initialize rotated arrays
        x_2D_plot_rotated = np.zeros((x_2D_plot.shape[0],x_2D_plot.shape[1]))
        y_2D_plot_rotated = np.zeros((x_2D_plot.shape[0],x_2D_plot.shape[1]))
        z_2D_plot_rotated = np.zeros((x_2D_plot.shape[0],x_2D_plot.shape[1]))
        fieldline_X_rotated = np.zeros((fieldline_X.shape[0],fieldline_X.shape[1]))
        fieldline_Y_rotated = np.zeros((fieldline_X.shape[0],fieldline_X.shape[1]))
        fieldline_Z_rotated = np.zeros((fieldline_X.shape[0],fieldline_X.shape[1]))
        # Rotate surfaces
        for th in range(x_2D_plot.shape[0]):
            for ph in range(x_2D_plot.shape[1]):
                [x_2D_plot_rotated[th,ph], y_2D_plot_rotated[th,ph], z_2D_plot_rotated[th,ph]] = rx.apply(rz.apply(np.array([x_2D_plot[th,ph], y_2D_plot[th,ph], z_2D_plot[th,ph]])))
        # Rotate field lines
        for th in range(fieldline_X.shape[0]):
            for ph in range(fieldline_X.shape[1]):
                [fieldline_X_rotated[th,ph], fieldline_Y_rotated[th,ph], fieldline_Z_rotated[th,ph]] = rx.apply(rz.apply(np.array([fieldline_X[th,ph], fieldline_Y[th,ph], fieldline_Z[th,ph]])))
        # Plot surfaces
        mlab.mesh(x_2D_plot_rotated, y_2D_plot_rotated-shift_array[i], z_2D_plot_rotated, scalars=Bmag, colormap='viridis')
        # Plot field lines
        for j in range(len(alphas)):
            mlab.plot3d(fieldline_X_rotated[j], fieldline_Y_rotated[j]-shift_array[i], fieldline_Z_rotated[j], color=(0,0,0), line_width=0.001, tube_radius=0.005)

def get_boundary(self, r=0.1, ntheta=40, nphi=130, ntheta_fourier=20, mpol = 13, ntor = 25):
    '''
    Function that, for a given near-axis radial coordinate r, outputs
    the [X,Y,Z,R,Z] components of the boundary. The resolution along the toroidal
    angle phi is equal to the resolution nphi for the axis, while ntheta
    is specified by the used.

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the poloidal angle.
      nphi   (int): Number of grid points to plot in the toroidal angle.
      ntheta_fourier (int): Resolution in the Fourier transform to cylindrical coordinates
      mpol: resolution in poloidal Fourier space
      ntor: resolution in toroidal Fourier space
    '''
    # Get surface shape at fixed off-axis toroidal angle phi
    R_2D, Z_2D, _ = self.Frenet_to_cylindrical(r, ntheta = ntheta_fourier)
    # Get Fourier coefficients in order to plot with arbitrary resolution
    RBC, RBS, ZBC, ZBS = to_Fourier(R_2D, Z_2D, self.nfp, ntheta = ntheta_fourier, mpol = mpol, ntor = ntor, lasym = self.lasym)
    if not self.lasym:
        RBS = np.zeros((int(2*ntor+1),int(mpol+1)))
        ZBC = np.zeros((int(2*ntor+1),int(mpol+1)))

    theta1D = np.linspace(0,2*np.pi,ntheta)
    phi1D = np.linspace(0,2*np.pi,nphi)
    phi2D, theta2D = np.meshgrid(phi1D,theta1D)
    R_2Dnew = np.zeros((ntheta,nphi))
    Z_2Dnew = np.zeros((ntheta,nphi))
    for m in range(mpol+1):
        for n in range(-ntor, ntor+1):
            angle = m * theta2D - n * self.nfp * phi2D
            R_2Dnew += RBC[n+ntor,m] * np.cos(angle) + RBS[n+ntor,m] * np.sin(angle)
            Z_2Dnew += ZBC[n+ntor,m] * np.cos(angle) + ZBS[n+ntor,m] * np.sin(angle)

    # X, Y, Z arrays for the whole surface
    x_2D_plot = R_2Dnew*np.cos(phi1D)
    y_2D_plot = R_2Dnew*np.sin(phi1D)
    z_2D_plot = Z_2Dnew

    return x_2D_plot, y_2D_plot, z_2D_plot, R_2Dnew, Z_2Dnew

def plot(self, r=0.1, ntheta=80, nphi=150, ntheta_fourier=20, nsections=8, fieldlines=False, savefig=None, colormap=None, azim_default=None, **kwargs):
    """
    Plotting routine for the near-axis configurations. There are two main ways of
    running this function:
    If fieldlines=False (default), it creates 2 matplotlib figures:
        - A plot with several poloidal planes at the specified radius r with the
         corresponding location of the magnetic axis and label using plt.plot
        - A 3D plot with the flux surface and the magnetic field strength
         on the surface using plot_surface().
    If fieldlines=True, it creates 1 matplotlib figure:
        - A plot with several poloidal planes at the specified radius r with the
         corresponding location of the magnetic axis and label using plt.plot
        and one mayavi scene
        - A 3D plot with the flux surface the magnetic field strength
         on the surface and several magnetic field lines using mlab.mesh()
        This functionality needs the mayavi package.

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the poloidal angle.
      nphi   (int): Number of grid points to plot in the toroidal angle.
      ntheta_fourier (int): Resolution in the Fourier transform to cylindrical coordinates
      nsections (int): Number of poloidal planes to show.
      fieldlines (bool): Specify if fieldlines are shown. Using mayavi instead of matplotlib due to known bug https://matplotlib.org/2.2.2/mpl_toolkits/mplot3d/faq.html
      savefig (str): Filename prefix for the png files to save
      colormap (cmap): Custom colormap for the 3D plots
      azim_default: Default azimuthal angle for the three subplots in the 3D surface plot
      kwargs: Any additional key-value pairs to pass to matplotlib's plot_surface.

    This function generates similar to the ones below:

    .. image:: 3dplot1.png
       :width: 200

    .. image:: 3dplot2.png
       :width: 200

    .. image:: poloidalplot.png
       :width: 200
    """
    x_2D_plot, y_2D_plot, z_2D_plot, R_2Dnew, Z_2Dnew = self.get_boundary(r=r, ntheta=ntheta, nphi=nphi, ntheta_fourier=ntheta_fourier)

    ## Poloidal plot
    phi1dplot_RZ = np.linspace(0,2*np.pi/self.nfp,nsections,endpoint=False)
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax  = plt.gca()
    for i, phi in enumerate(phi1dplot_RZ):
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
        # Plot location of the axis
        plt.plot(self.R0_func(phi),self.Z0_func(phi),marker="x",linewidth=2,label=label,color=color)
        # Plot location of the poloidal cross-sections
        pos = int(phi/(2*np.pi)*nphi)
        plt.plot(R_2Dnew[:,pos].flatten(),Z_2Dnew[:,pos].flatten(),color=color)
    plt.xlabel('R (meters)')
    plt.ylabel('Z (meters)')
    plt.legend()
    plt.tight_layout()
    ax.set_aspect('equal')
    if savefig!=None:
        fig.savefig(savefig+'_poloidal.png')

    ## 3D plot
    # Set the default azimuthal angle of view in the 3D plot
    # QH stellarators look rotated in the phi direction when
    # azim_default = 0
    if azim_default == None:
        if self.helicity == 0:
            azim_default = 0
        else:
            azim_default = 45
    # Define the magnetic field modulus and create its theta,phi array
    # The norm instance will be used as the colormap for the surface
    theta1D = np.linspace(0,2*np.pi,ntheta)
    phi1D = np.linspace(0,2*np.pi,nphi)
    phi2D, theta2D = np.meshgrid(phi1D,theta1D)
    # Create a color map similar to viridis 
    Bmag=self.B_mag(r,theta2D,phi2D)
    norm = clr.Normalize(vmin=Bmag.min(), vmax=Bmag.max())
    if fieldlines==False:
        if colormap==None:
            # Cmap similar to quasisymmetry papers
            # cmap = clr.LinearSegmentedColormap.from_list('qs_papers',['#4423bb','#4940f4','#2e6dff','#0097f2','#00bacc','#00cb93','#00cb93','#7ccd30','#fbdc00','#f9fc00'], N=256)
            cmap = cm.viridis
            # Add a light source so the surface looks brighter
            ls = LightSource(azdeg=0, altdeg=10)
            cmap_plot = ls.shade(Bmag, cmap, norm=norm)
        # Create the 3D figure and choose the following parameters:
        # gsParams: extension in the top, bottom, left right directions for each subplot
        # elevParams: elevation (distance to the plot) for each subplot
        fig = plt.figure(constrained_layout=False, figsize=(4.5, 8))
        gsParams = [[1.02,-0.3,0.,0.85],[1.09,-0.3,0.,0.85],[1.12,-0.15,0.,0.85]]
        elevParams = [90,30,5]
        for i in range(len(gsParams)):
            gs = fig.add_gridspec(nrows=3, ncols=1, top=gsParams[i][0], bottom=gsParams[i][1], left=gsParams[i][2], right=gsParams[i][3], hspace=0.0, wspace=0.0)
            ax = fig.add_subplot(gs[i, 0], projection='3d')
            create_subplot(ax, x_2D_plot, y_2D_plot, z_2D_plot, cmap_plot, elev=elevParams[i], azim=azim_default, **kwargs)
        # Create color bar with axis placed on the right
        cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
        m = cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array([])
        cbar = plt.colorbar(m, cax=cbar_ax)
        cbar.ax.set_title(r'$|B| [T]$')
        # Save figure
        if savefig!=None:
            fig.savefig(savefig+'3D.png')
        # Show figures
        plt.show()
        # Close figures
        plt.close()
    else:
        ## X, Y, Z arrays for the field lines
        # Plot different field lines corresponding to different alphas
        # where alpha=theta-iota*varphi with (theta,varphi) the Boozer angles
        alphas = [0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4]
        # Create the field line arrays
        fieldline_X, fieldline_Y, fieldline_Z = create_field_lines(self, alphas, x_2D_plot, y_2D_plot, z_2D_plot)
        # Define the rotation arrays for the subplots
        degrees_array_x = [0., -66., 81.] # degrees for rotation in x
        degrees_array_z = [azim_default, azim_default, azim_default] # degrees for rotation in z
        shift_array   = [-1.0, 0.7, 1.8]
        # Import mayavi and rotation packages (takes a few seconds)
        from mayavi import mlab
        from scipy.spatial.transform import Rotation as R
        # Show RZ plot
        plt.show()
        # Close RZ plot
        plt.close()
        # Create 3D figure
        fig = mlab.figure(bgcolor=(1,1,1), size=(430,720))
        # Create subplots
        create_subplot_mayavi(mlab, R, alphas, x_2D_plot, y_2D_plot, z_2D_plot, fieldline_X, fieldline_Y, fieldline_Z, Bmag, degrees_array_x, degrees_array_z, shift_array)
        # Create a good camera angle
        mlab.view(azimuth=0, elevation=0, distance=8.5, focalpoint=(-0.15,0,0), figure=fig)
        # Create the colorbar and change its properties
        cb = mlab.colorbar(orientation='vertical', title='|B| [T]', nb_labels=7)
        cb.scalar_bar.unconstrained_font_size = True
        cb.label_text_property.font_family = 'times'
        cb.label_text_property.bold = 0
        cb.label_text_property.font_size=24
        cb.label_text_property.color=(0,0,0)
        cb.title_text_property.font_family = 'times'
        cb.title_text_property.font_size=34
        cb.title_text_property.color=(0,0,0)
        cb.title_text_property.bold = 1
        # Save figure
        if savefig!=None:
            mlab.savefig(filename=savefig+'3D_fieldlines.png', figure=fig)
        # Show mayavi plot
        mlab.show()
        # Close mayavi plots
        mlab.close(all=True)

def B_fieldline(self, r=0.1, alpha=0, phimax = [], nphi = 400):
    '''
    Plot the modulus of the magnetic field B along a field line with
    the Boozer toroidal angle varphi acting as a field-line following
    coordinate

    Args:
      r (float): near-axis radius r where to create the surface
      alpha (float): Field-line label
      phimax (float): Maximum value of the field-line following parameter varphi
      nphi (int): resolution of the phi grid
    '''
    if phimax == []:
        phimax = 10*np.pi/abs(self.iota)
    varphi_array = np.linspace(0,phimax,nphi)
    _,ax=plt.subplots(1,1,figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.xlabel(r'$\varphi$')
    plt.ylabel(r'$B(\varphi)$')
    plt.title("r = "+str(r)+", alpha = "+str(alpha))
    plt.plot(varphi_array,self.B_mag(r,alpha+self.iota*varphi_array,varphi_array,Boozer_toroidal=True))
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=phimax*abs(self.iota)/np.pi))
    plt.tight_layout()
    plt.show()
    plt.close()

def B_contour(self, r=0.1, ntheta=100, nphi=100, ncontours=10):
    '''
    Plot contours of constant B, with B the modulus of the
    magnetic field, in Boozer coordinates theta and varphi

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the Boozer poloidal angle.
      nphi   (int): Number of grid points to plot in the Boozer toroidal angle.
      ncontours (int): number of contours to show in the plot
    '''
    theta_array=np.linspace(0,2*np.pi,ntheta)
    phi_array=np.linspace(0,2*np.pi,nphi)
    theta_2D, phi_2D = np.meshgrid(theta_array,phi_array)
    magB_2D = self.B_mag(r,phi_2D,theta_2D,Boozer_toroidal=True)
    magB_2D.shape = phi_2D.shape
    fig,ax=plt.subplots(1,1)
    contourplot = ax.contourf(phi_2D, theta_2D, magB_2D, ncontours)
    fig.colorbar(contourplot)
    ax.set_title('r='+str(r))
    ax.set_xlabel(r'$\varphi$')
    ax.set_ylabel(r'$\theta$')
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=1.0))
    plt.tight_layout()
    plt.show()
    plt.close()