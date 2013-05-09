"""A collection of some HJCFit related functions."""

import sys
import random
from math import*
import numpy as np

import qmatlib as qml

def simplex(func, theta, args=None,
    stpfac=10, reffac=1.0, extfac=2.0, confac=0.5,
    shrfac=0.5, resfac=10.0, perfac=0.1,
    errpar=1e-3, errfunc=1e-3,
    maxiter=10000, maxeval=100000,
    display=False, outdev=sys.stdout):
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

            if display: # and (iterations % 10) == 0:
                outdev.write('\niter# {0:d}\tlik= {1:f}\ttheta=\n'.format(iterations, -fval[0]))
                for th in np.exp(simp[0]):
                    outdev.write('{0:6e}\t'.format(th))

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

def bisect_gFB(s, tres, Q11, Q22, Q12, Q21, k1, k2):
    """
    Find number of eigenvalues of H(s) that are equal to or less than s.

    Parameters
    ----------
    s : float
        Laplace transform argument.
    tres : float
        Time resolution (dead time).
    Q11 : array_like, shape (k1, k1)
    Q22 : array_like, shape (k2, k2)
    Q21 : array_like, shape (k2, k1)
    Q12 : array_like, shape (k1, k2)
        Q11, Q12, Q22, Q21 - submatrices of Q.
    k1 : int
        A number of open/shut states in kinetic scheme.
    k2 : int
        A number of shut/open states in kinetic scheme.

    Returns
    -------
    ng : int
    """

    h = qml.H(s, tres, Q11, Q22, Q12, Q21, k2)
    eigval, A = qml.eigs(h)
    ng = 0
    for i in range(k1):
        if eigval[i] <= s: ng += 1
#    if dcpypsrc.debug:
#        print ('number of eigenvalues that are <= s (=', s, ') =', ng)
    return ng

def bisect_intervals(sa, sb, tres, Q11, Q22, Q12, Q21, k1, k2):
    """
    Find, according to Frank Ball's method, suitable starting guesses for
    each HJC root- the upper and lower limits for bisection. Exactly one root
    should be between those limits.

    Parameters
    ----------
    sa, sb : float
        Laplace transform arguments.
    tres : float
        Time resolution (dead time).
    Q11 : array_like, shape (k1, k1)
    Q22 : array_like, shape (k2, k2)
    Q21 : array_like, shape (k2, k1)
    Q12 : array_like, shape (k1, k2)
        Q11, Q12, Q22, Q21 - submatrices of Q.
    k1, k2 : int
        Numbers of open/shut states in kinetic scheme.

    Returns
    -------
    sr : array_like, shape (k2, 2)
        Limits of s value intervals containing exactly one root.
    """

    nga = bisect_gFB(sa, tres, Q11, Q22, Q12, Q21, k1, k2)
    if nga > 0:
        sa = sa * 4
    ngb = bisect_gFB(sb, tres, Q11, Q22, Q12, Q21, k1, k2)
    if ngb < k2:
        sb = sb / 4

    sr = np.zeros((k1, 2))
    sv = np.empty((101, 4))
    sv[0,0] = sa
    sv[0,1] = sb
    sv[0,2] = nga
    sv[0,3] = ngb
    ntodo = 0
    ndone = 0
    nsplit = 0

    while (ndone < k1) and (nsplit < 1000):
        sa = sv[ntodo, 0]
        sb = sv[ntodo, 1]
        nga = sv[ntodo, 2]
        ngb = sv[ntodo, 3]
        sa1, sc, sb2, nga1, ngc, ngb2 = bisect_split(sa, sb,
            nga, ngb, tres, Q11, Q22, Q12, Q21, k1, k2)
        nsplit = nsplit + 1
        ntodo = ntodo - 1

        # Check if either or both of the two subintervals output from
        # SPLIT contain only one root?
        if (ngc - nga1) == 1:
            sr[ndone, 0] = sa1
            sr[ndone, 1] = sc
            ndone = ndone + 1
            if ndone == k1:
                break
        else:
            ntodo = ntodo + 1
            sv[ntodo, 0] = sa1
            sv[ntodo, 1] = sc
            sv[ntodo, 2] = nga1
            sv[ntodo, 3] = ngc
        if (ngb2 - ngc) == 1:
            sr[ndone, 0] = sc
            sr[ndone, 1] = sb2
            ndone = ndone + 1
        else:
            ntodo = ntodo + 1
            sv[ntodo, 0] = sc
            sv[ntodo, 1] = sb2
            sv[ntodo, 2] = ngc
            sv[ntodo, 3] = ngb2

    if ndone < k1:
        sys.stderr.write(
            "bisectHJC: Warning: Only {0:d} roots out of {1:d} were located.".
            format(ndone, k1))

    return sr

def bisect_split(sa, sb, nga, ngb, tres, Q11, Q22, Q12, Q21, k1, k2):
    """
    Split interval [sa, sb] into two subintervals, each of which contains
    at least one root.

    Parameters
    ----------
    sa, sb : float
        Limits of Laplace transform argument interval.
    nga, ngb : int
        Number of eigenvalues (roots) below sa or sb, respectively.
    tres : float
        Time resolution (dead time).
    Q11 : array_like, shape (k1, k1)
    Q22 : array_like, shape (k2, k2)
    Q21 : array_like, shape (k2, k1)
    Q12 : array_like, shape (k1, k2)
        Q11, Q12, Q22, Q21 - submatrices of Q.
    k1, k2 : int
        Numbers of open/shut states in kinetic scheme.

    Returns
    -------
    sa, sc, sb : floats
        Limits of s value intervals.
    nga, ngc, ngb : ints
        Number of eigenvalues below corresponding s values.
    """

    ntrymax = 1000
    ntry = 0
    #nerrs = False
    end = False

    while (not end) and (ntry < ntrymax):
        sc = (sa + sb) / 2.0
        ngc = bisect_gFB(sc, tres, Q11, Q22, Q12, Q21, k1, k2)
        if ngc == nga: sa = sc
        elif ngc == ngb: sb = sc
        else:
            end = True
        ntry += 1
    if not end:
        sys.stderr.write(
        "bisectHJC: Warning: unable to split intervals for bisection.")

    return sa, sc, sb, nga, ngc, ngb
