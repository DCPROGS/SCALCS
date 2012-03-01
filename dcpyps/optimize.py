"""A collection of some HJCFit related functions."""

import random
import sys
import numpy as np
from math import*

def simplex_make(theta, stpfac, func, args):

    k = np.size(theta)
    simp = np.zeros((k+1,k), dtype=theta.dtype)
    fval = np.zeros((k+1,))
    fval[0], theta = func(theta, args)
    simp[0] = theta

    stpfac = log(stpfac)
    step = np.ones((k)) * stpfac
    fac = (sqrt(k+1) - 1.) / (k * sqrt(2.))

    for i in range(k):
        #ar = np.copy(simp[0])
        ar = theta + step * fac
        ar[i] = theta[i] + step[i] * (fac + 1. / sqrt(2))
        fval[i+1], simp[i+1] = func(ar, args)
#        simp[i+1] = th
#        fval[i+1] = f

    return fval, simp

def simplex_sort(fval, simp):
    ind = np.argsort(fval)
    fval = np.take(fval,ind,0)
    # sort so simp[0,:] has the lowest function value
    simp = np.take(simp,ind,0)
    return fval, simp

def simplex_shrink(fval, simp, shrfac, func, args):
    n = np.size(fval)
    for j in range(1, n):
        ar = simp[0] + shrfac * (simp[j] - simp[0])
        fval[j], th = func(ar, args)
        simp[j] = th
    return fval, simp

def simplex_random(theta, perfac, func, args):
    rand = random.random()
    perfac = 0.3
    xr = theta * (rand * 2 * perfac + (1 - perfac))
    fr, xr = func(xr, args)
    return fr, xr

def simplex_local(theta, errpar, func, args):
    """
    Do local search with each param +/- crtstp at current best vertex.
    """

    xl1 = theta + errpar
    fxl1, xl1 = func(xl1, args)
    xl2 = theta - errpar
    fxl2, xl2 = func(xl2, args)
    if fxl1 < fxl2:
        xout = xl1
        fout = fxl1
    else:
        xout = xl2
        fout = fxl2
    return fout, xout

def simplex(func, theta, args=None,
    stpfac=10, reffac=1.0, extfac=2.0, confac=0.5,
    shrfac=0.5, resfac=10.0, perfac=0.1,
    errpar=1e-3, errfunc=1e-3,
    maxiter=10000, maxeval=100000,
    display=False):
    """
    Minimize a function using the Nelder-Mead simplex algorithm to find the
    minimum of function.

    Parameters
    ----------
    func : callable func(x, args)
        The objective function to be minimized.
    theta : ndarray
        Initial guess.
    args : dictionary
        Extra arguments passed to func.

    errpar : float
        Relative error in xopt acceptable for convergence.
    errfunc : number
        Relative error in func(xopt) acceptable for convergence.
    maxiter : int
        Maximum number of iterations to perform.
    maxeval : number
        Maximum number of function evaluations to make.

    Returns
    -------
    xout : ndarray
        Parameters that minimize function.
    fout : float
        Value of function at minimum.
    iterations : int
        Number of iterations performed.
    funcalls : int
        Number of function calls made.
    """

    k = np.size(theta)
    restart = True
    iterations = 1
    fcalls = 0

    while restart:

        fval, simp = simplex_make(theta, stpfac, func, args)
        fcalls += (k + 1)
        fval, simp = simplex_sort(fval, simp)
        
        while (fcalls < maxeval and iterations < maxiter):

            if display and (iterations % 10) == 0:
                print ('iter# {0:d}\tlik= {1:f}'.format(iterations, -fval[0]))
                print 'theta=', np.exp(simp[0])

            # Check for convergence.
            if (max(np.ravel(abs(simp[1:] - simp[0]))) <= errpar \
                and max(abs(fval[0] - fval[1:])) <= errfunc):
                #do local search
                floc, xloc = simplex_local(theta, errpar, func, args)
                if floc < fval[0]:
                    theta = xloc
                    stpfac = resfac * errpar
                    restart = True
                else:
                    restart = False
                break

            centre = np.sum(simp[:-1,:], axis=0) / float(k)
            # Reflect
            xr = centre + reffac * (centre - simp[-1])
            fxr, xr = func(xr, args)
            fcalls += 1

            if fxr < fval[0]:
                # Extend
                xe = centre + extfac * (xr - centre)
                fxe, xe = func(xe, args)
                fcalls += 1
                if fxe < fxr:
                    simp[-1] = xe
                    fval[-1] = fxe
                else:
                    simp[-1] = xr
                    fval[-1] = fxr

            else: # fval[0] <= fxr reflected vertex is not better than the best


                if fxr < fval[-2]:
                    simp[-1] = xr
                    fval[-1] = fxr
                else: # fxr >= fval[-2]
                    # Perform contraction
                    if fxr < fval[-1]:
                        simp[-1] = xr
                        fval[-1] = fxr
                    xc = centre + confac * (simp[-1] - centre)
                    fxc, xc = func(xc, args)
                    fcalls += 1

                    if fxc <= fval[-1]:
                        simp[-1] = xc
                        fval[-1] = fxc
                    else:
                        fval, simp = simplex_shrink(fval, simp, shrfac, func, args)
                        fcalls += k

            fval, simp = simplex_sort(fval, simp)
            iterations += 1

    xout = simp[0]
    fout = fval[0]

    if fcalls >= maxeval:
        sys.stderr.write(
            "simplex: Warning: Maximum number of function evaluations has "\
            "been exceeded.")
    if iterations >= maxiter:
        sys.stderr.write(
            "simplex:Warning: Maximum number of iterations has been exceeded")

    return xout, fout, iterations, fcalls
