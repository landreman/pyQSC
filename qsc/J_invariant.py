"""
This module contains the routine to calculate the second
adiabatic invariant, usually called J
"""

import numpy as np
from scipy.optimize import brentq
import math
from scipy.integrate import quadrature
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def find_roots(func,xmin,xmax,nx,root_tol):
    """
    The function find_roots finds the roots of a given
    function (func) in the interval (xmin,xmax) by looking
    where func changes zeros in an array with nphi points.

    Args:
        func (function): function to find the zeros
        xmim (float): minimum of the argument of func where to look for zeros
        xmax (float): maximum of the argument of func where to look for zeros
        nx (float): number of points where to look for zeros
        root_tol (float): tolerance for the root-finding algorithm
    """
    x_array = np.linspace(xmin, xmax, nx) # array of points
    x_roots_temp = np.array([]) # temporary array to store all the roots
    func_initial = func(x_array) # values of the function at x_array
    s = np.sign(func_initial) # array of 1 and -1 where the function changes sign
    for i in range(nx-1):
        if s[i] + s[i+1] == 0: # check for oposite signs
            root = brentq(func, x_array[i], x_array[i+1]) # further resolve the root with brentq algorithm
            func_at_root = func(root)
            if np.isnan(func_at_root) or abs(func_at_root) > root_tol: # check that everything is ok
                continue
            # logger.info('Found zero at {}'.format(root))
            x_roots_temp = np.append(x_roots_temp,root)
    return x_roots_temp

def J_invariant(self, r=0.1, alpha=0, Lambda=1.0, plot=False, thetaMin = 0., thetaMax = 12*np.pi, ntheta = 500, root_tol = 1e-6):
    """
    The function J_invariant takes a given radius r, field-line label
    alpha and a pitch-angle Lambda and calculates the corresponding
    second adiabatic invariant J=m*integral(v_parallel,dl) where the
    integral is performed between zeros of v_parallel. It uses
    theta as a field-line following angle. There's the possibility
    of plotting the location along the field line where v_parallel
    is being integrated. The result is normalized by  2 times the
    particle's mass and velocity, J/(2*m*v), where v = sqrt(2*E/m)
    with E the particle's energy.

    Args:
        r (float): near-axis radius 
        alpha (float): field-line label, only relevant at second order
        Lambda (float): normalized pitch-angle mu*B0/E
        plot (bool): True to show a debugging plot
        thetamin (float): minimum of the variable theta where to look for zeros of v_parallel
        thetamax (float): maximum of the variable theta where to look for zeros of v_parallel
        ntheta (float): number of points in the variable theta where to look for zeros of v_parallel
        root_tol (float): tolerance in the root-finding algorithm
        nfp: number of field periods of the surface
        ntheta: poloidal resolution
    """
    # Define v_parallel^2 and J functions analytically
    def vpar2(theta):
        return 1-Lambda*self.B_mag(r,(theta-alpha)/self.iota,theta,Boozer_toroidal=True)
    def J_normalized_integrand(theta):
        return np.sqrt(vpar2(theta))/self.B_mag(r,(theta-alpha)/self.iota,theta,Boozer_toroidal=True)
    # Find zeros of v_parallel
    theta_roots_temp = find_roots(vpar2,thetaMin,thetaMax,ntheta,root_tol)
    # Check if roots are good roots with a two-step approach
    # First see if J is calculated and is not nan
    # Then check if J is nan, which means the interval between
    # roots was too big and we have to find the roots inside that interval
    if len(theta_roots_temp) == 0:
        logger.info('No zeros in the roots above')
        theta_roots = theta_roots_temp
        J_normalized = 0
    else:
        theta_roots = np.array([])
        for i in range(len(theta_roots_temp)):
            if vpar2(1.001*theta_roots_temp[i]) > 0:
                try:
                    theta_roots = np.array([theta_roots_temp[i],theta_roots_temp[i+1]])
                    # Integrate sqrt(v_parallel^2)/B
                    J_normalized = quadrature(J_normalized_integrand,theta_roots[0],theta_roots[1],maxiter=500)[0]
                    if math.isnan(J_normalized):
                        logger.info('J is nan, try again')
                        theta_roots2 = find_roots(vpar2,theta_roots[0],theta_roots[1],ntheta,root_tol)
                        logger.info('Old roots: {}'.format(theta_roots))
                        logger.info('New roots: {}'.format(theta_roots2))
                        for i in range(len(theta_roots_temp)):
                            if vpar2(1.001*theta_roots2[i]) > 0:
                                try:
                                    theta_roots = np.array([theta_roots2[i],theta_roots2[i+1]])
                                    logger.info('Getting new theta_roots: {}'.format(theta_roots))
                                    # Integrate sqrt(v_parallel^2)/B
                                    J_normalized = quadrature(J_normalized_integrand,theta_roots[0],theta_roots[1],maxiter=500)[0]
                                    break
                                except:
                                    logger.info("Vpar2 not positive between roots")
                                    break
                    else:
                        logger.info("J_invariant successfully calculated, J_normalized="+str(J_normalized))
                        break
                except:
                    logger.info("Vpar2 not positive between roots")
                    break
    if len(theta_roots) == 0:
        J_normalized = 0
    if self.order=='r1':
        J = J_normalized*(self.G0+self.I2*r**2)/abs(self.iotaN)
    else:
        J = J_normalized*(self.G0+self.I2*r**2+self.G2*r**2)/abs(self.iotaN)
    # Plot v_parallel^2 to check that we've found the proper zeros
    if plot==True:
        print("Roots of v_parallel =",theta_roots)
        thetaArray = np.linspace(thetaMin,thetaMax,ntheta)
        fig_plot = plt.figure()
        plt.plot(thetaArray,vpar2(thetaArray))
        plt.plot(theta_roots,vpar2(theta_roots), linestyle="None", marker='o')
        plt.title('r='+str(r)+', lambda='+str(Lambda)+', J='+str(J_normalized))
        plt.draw()
        # plt.close(fig_plot)
    return J

def npmap2d(fun, xs, ys, **kwargs):
    """
    The function npmap2d cicles the funcion "fun" through the
    values of the arrays xs and ys and returns meshgrid representations
    of the function values and both input arrays ready to be plotted
    in a contour plot

    Args:
        fun (function): function to be cicled over
        xs (array): first array of floats to cycle through
        ys (array): second array of floats to cycle through
        nalpha (float): number of grid points in the field-line alpha array
        Lambda (float): normalized pitch-angle mu*B0/E
        plot (bool): true to show a debugging plot
        ncontours (int): number of contours to show in the contour plot
    """
    Z = np.empty(len(xs) * len(ys))
    i = 0
    for y in ys:
        for x in xs:
            Z[i] = fun(r=x, alpha=y, **kwargs)
            i += 1
    Z.shape = (len(ys),len(xs))
    return Z

def J_contour(self, rmin=0.001, rmax=0.1, nr=6, nalpha=20, lambdas=[0.9,1.0,1.1], numCols=3, plot_debug=False, ncontours=None):
    """
    The function J_contour plots the second adiabatic invariant on a contour plot
    with the radial variable an array in an evenly spaced sequence in the interval
    (rmin,rmax) with nr elements, and similarly for alpha with alpha in (0,2*pi).
    The range for the pitch-angles (lambdas) is given as input

    Args:
        rmin (float): minimum near-axis radius 
        rmax (float): maximum near-axis radius 
        nr (float): number of grid points in the near-axis radius array
        nalpha (float): number of grid points in the field-line alpha array
        lambdas (array): values of the normalized pitch-angle mu*B0/E to show in the plot
        numCols (int): number of columns to show in the plot
        plot_debug (bool): true to show a debugging plot
        ncontours (int): number of contours to show in the contour plot
    """
    # If the user does not specify the number of contours, just use the number of radial points
    if ncontours==None:
        ncontours=nr
    # Create arrays to loop through
    r_array = np.linspace(rmin,rmax,nr)
    alpha_array = np.linspace(0,2*np.pi,nalpha)
    # Calculate the number of rows and columns to show in the figure
    if len(lambdas)<numCols:
        numCols = len(lambdas)
    nrows=int(np.ceil(len(lambdas)/numCols))
    # Initialize figure
    fig, ax = plt.subplots(nrows, numCols, figsize=(0.5+3*numCols,0.5+3*nrows), subplot_kw=dict(projection='polar'))
    # Calculate J for the given values of r, alpha and lambda
    r_2D, theta_2D = np.meshgrid(r_array,alpha_array)
    J_2D = np.zeros((len(lambdas),nalpha,nr))
    for i in range(len(lambdas)):
        J_2D[i] = npmap2d(self.J_invariant, r_array, alpha_array, Lambda=lambdas[i], plot=plot_debug)
        if nrows==1:
            if numCols==1:
                axs = ax
            else:
                axs = ax[np.mod(i,numCols)]
        else:
            axs = ax[int(i/numCols), np.mod(i,numCols)]
        contourplot = axs.contourf(theta_2D, r_2D, J_2D[i], ncontours)
        # Add axis and title
        axs.yaxis.grid(False)
        # axs.yaxis.set_ticklabels([])
        axs.set_xticklabels([])
        axs.spines["polar"].set_visible(False)
        axs.set_rgrids(np.linspace(min(r_array),max(r_array),3), angle=0.)
        axs.set_rticks([0,r_array[-1]])
        axs.xaxis.grid(True,color='black',linestyle='-')
        # axs.set_rmin(0) # needs to be after ax.fill. No idea why.
        axs.set_xticks(np.linspace(0, 2*np.pi, 4, endpoint=False))
        axs.set_xticklabels(["0",r"$\pi$/2",r"$\pi$",r"3$\pi$/2"])
        axs.title.set_text(r'$\lambda$='+str(lambdas[i]))
        # Add colorbar
        fig.colorbar(contourplot, ax=axs, fraction=0.046, pad=0.13)
    fig.tight_layout()
    plt.show()