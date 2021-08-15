"""
This module contains a function for Newton's method.
"""

import numpy as np
import logging

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def newton(f, x0, jac, niter=20, tol=1e-13, nlinesearch=5):
    """
    Solve a system of nonlinear equations using Newton's method with a
    line search.

    f = function providing the residual vector.
    x0 = initial guess
    jac = function providing the Jacobian.
    niter = max number of Newton iterations.
    tol = stop when the residual norm is less than this.
    """

    x = np.copy(x0)
    x_best = np.copy(x0)
    residual = f(x0)
    initial_residual_norm = np.sqrt(np.sum(residual * residual))
    residual_norm = initial_residual_norm
    logger.info('Beginning Newton method. residual {}'.format(residual_norm))

    newton_tolerance_achieved = False
    for jnewton in range(niter):
        last_residual_norm = residual_norm
        if residual_norm < tol:
            newton_tolerance_achieved = True
            break

        j = jac(x0)
        x0 = np.copy(x)
        logger.info('Newton iteration {}'.format(jnewton))
        step_direction = -np.linalg.solve(j, residual)

        step_scale = 1.0
        for jlinesearch in range(nlinesearch):
            x = x0 + step_scale * step_direction
            residual = f(x)
            residual_norm = np.sqrt(np.sum(residual * residual))
            logger.info('  Line search step {} residual {}'.format(jlinesearch, residual_norm))
            if residual_norm < last_residual_norm:
                x_best = np.copy(x)
                break

            step_scale /= 2
            
        if residual_norm >= last_residual_norm:
            logger.info('Line search failed to reduce residual')
            break
    return x_best
