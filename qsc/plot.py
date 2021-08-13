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

def create_field_lines(alphas, nfp, iota, R_2D_spline, Z_2D_spline, varphi_spline, phimax=2*np.pi, nphi=200):
    '''
    Function to compute the (X, Y, Z) coordinates of field lines at
    several alphas, where alpha = theta-iota*varphi with (theta,varphi)
    the Boozer toroidal angles.

    Args:
      alphas: array of field line labels alpha
      nfp: number of field periods
      iota: rotational transform
      R_2D_spline: spline interpolant of the radial component of the surface
      Z_2D_spline: spline interpolant of the vertical component of the surface
      varphi_spline: spline interpolant of the Boozer angle varphi
      phimax: maximum value for the field line following angle phi
      nphi: grid resolution for the output fieldline
    '''
    phi_array = np.linspace(0,phimax,nphi)
    fieldline_X = np.zeros((len(alphas),nphi))
    fieldline_Y = np.zeros((len(alphas),nphi))
    fieldline_Z = np.zeros((len(alphas),nphi))
    for i in range(len(alphas)):
        for j in range(len(phi_array)):
            phi_mod = np.mod(phi_array[j],2*np.pi/nfp)
            varphi0=varphi_spline(phi_array[j])+phi_array[j]-phi_mod
            theta_fieldline=iota*varphi0+alphas[i]
            theta_fieldline_mod=np.mod(theta_fieldline,2*np.pi)
            fieldline_R = R_2D_spline(phi_mod,theta_fieldline_mod)[0]
            fieldline_X[i,j] = fieldline_R*np.cos(phi_array[j])
            fieldline_Y[i,j] = fieldline_R*np.sin(phi_array[j])
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

def get_boundary(self, r=0.1, ntheta=40, nphi=130, ntheta_fourier=16, get_splines=False):
    '''
    Function that, for a given near-axis radial coordinate r, outputs
    the [X,Y,Z] components of the boundary and, if specified by the
    user, it also outputs the spline interpolants for the cylindrical
    R and Z coordinates

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the poloidal angle.
      nphi   (int): Number of grid points to plot in the toroidal angle.
      ntheta_fourier (int): Resolution in the Fourier transform to cylindrical coordinates
      get_splines (bool): Specify if spline interpolants of R and Z are outputed (True/False)
    '''
    # Obtain the surface shape in cylindrical coordinates
    R_2D, Z_2D, _ = self.Frenet_to_cylindrical(r, ntheta_fourier)
    # Make it periodic
    R_2D = np.append(R_2D,[R_2D[0,:]],0)
    R_2D = np.append(R_2D,np.array([R_2D[:,0]]).transpose(),1)
    Z_2D = np.append(Z_2D,[Z_2D[0,:]],0)
    Z_2D = np.append(Z_2D,np.array([Z_2D[:,0]]).transpose(),1)
    # Arrays of original thetas and phis
    theta1d = np.linspace(0, 2 * np.pi, ntheta_fourier+1)
    phi1d   = np.linspace(0, 2 * np.pi / self.nfp, self.nphi+1)
    # Arrays of thetas and phis for plots
    theta1dplot  = np.linspace(0, 2 * np.pi, ntheta)
    phi1dplot    = np.linspace(0, 2 * np.pi / self.nfp, nphi)
    # Splines interpolants of R_2D and Z_2D
    # NON-PERIODIC SPLNE INTERPOLANTS -> Might have artifacts at the boundary
    R_2D_spline = interp2d(phi1d, theta1d, R_2D, kind='cubic')
    Z_2D_spline = interp2d(phi1d, theta1d, Z_2D, kind='cubic')
    R_2D_interp = R_2D_spline(phi1dplot,theta1dplot)
    Z_2D_interp = Z_2D_spline(phi1dplot,theta1dplot)

    #### Using RBC and ZBS to reconstruct R_2D and Z_2D
    #### Using ntheta_fourier = ntheta so that no interpolation is needed, only 1D
    #### Make copies of the data at the boundaries to get rid of inconsistencies

    # X, Y, Z arrays for the whole surface
    x_2D_plot = R_2D_interp*np.cos(phi1dplot)
    y_2D_plot = R_2D_interp*np.sin(phi1dplot)
    z_2D_plot = Z_2D_interp
    for i in range(1,self.nfp):
        x_2D_plot = np.concatenate((x_2D_plot,R_2D_interp*np.cos(phi1dplot+i*2*np.pi/self.nfp)), axis=1)
        y_2D_plot = np.concatenate((y_2D_plot,R_2D_interp*np.sin(phi1dplot+i*2*np.pi/self.nfp)), axis=1)
        z_2D_plot = np.concatenate((z_2D_plot,Z_2D_interp), axis=1)
    if get_splines==False:
        return x_2D_plot, y_2D_plot, z_2D_plot
    elif get_splines==True:
        return x_2D_plot, y_2D_plot, z_2D_plot, R_2D_spline, Z_2D_spline

def plot(self, r=0.1, ntheta_plot=40, nphi_plot=130, ntheta_fourier=16, nsections=8, fieldlines=False, savefig=None, colormap=None, azim_default=None, **kwargs):
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
      ntheta_plot (int): Number of grid points to plot in the poloidal angle.
      nphi_plot   (int): Number of grid points to plot in the toroidal angle.
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
    ## Obtain surface shape
    x_2D_plot, y_2D_plot, z_2D_plot, R_2D_spline, Z_2D_spline = self.get_boundary(r=r,ntheta=ntheta_plot,nphi=nphi_plot,ntheta_fourier=ntheta_fourier, get_splines=True)

    ## Poloidal plot
    phi1dplot_RZ = np.linspace(0, 2 * np.pi / self.nfp, nsections, endpoint=False)
    theta1dplot  = np.linspace(0, 2 * np.pi, ntheta_plot)
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
        # Plot location of the axis
        plt.plot(self.R0_func(phi),self.Z0_func(phi),marker="x",linewidth=2,label=label,color=color)
        # Plot location of the poloidal cross-sections
        plt.plot(R_2D_spline(phi,theta1dplot).flatten(),Z_2D_spline(phi,theta1dplot).flatten(),color=color)
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
    def Bf(r,theta,phi):
        thetaN = theta-(self.iota-self.iotaN)*phi
        return self.B0*(1+r*self.etabar*np.cos(thetaN))
    phi1dplot    = np.linspace(0, 2 * np.pi / self.nfp, nphi_plot)
    phi2D, theta2D = np.meshgrid(phi1dplot,theta1dplot)
    Bmag=np.repeat(Bf(r,theta2D,phi2D),self.nfp,axis=1)
    norm = clr.Normalize(vmin=Bmag.min(), vmax=Bmag.max())
    # Create a color map similar to viridis 
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
        varphi_spline = self.convert_to_spline(self.varphi)
        fieldline_X, fieldline_Y, fieldline_Z = create_field_lines(alphas, self.nfp, self.iota, R_2D_spline, Z_2D_spline, varphi_spline)
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