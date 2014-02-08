"""A collection of some HJCFit related functions."""

import sys
import time
import numpy as np
from scipy.optimize import minimize
import random
from math import*
import numpy as np
from numpy import linalg as nplin

import qmatlib as qml
from dcpyps.reports import FitReportHTML
from dcprogs.likelihood import Log10Likelihood

class FittingSession():
    """
    """
    def __init__(self, mec, recs):
        self.mec = mec
        self.recs = recs
        self.likelihood()
        self.report = FitReportHTML()
        self.report.dataset(self.recs)
        start_lik = self.dcprogslik(np.log(self.mec.theta()))
        print ("Starting likelihood = {0:.6f}".format(-start_lik))
        self.report.rates(self.mec, start_lik)
        self.iternum = 0
        
    def likelihood(self):
        self.likelihood = []
        kwargs = {'nmax': 2, 'xtol': 1e-12, 'rtol': 1e-12, 'itermax': 100,
            'lower_bound': -1e6, 'upper_bound': 0}
        for i in range(len(self.recs)):
            self.likelihood.append(Log10Likelihood(self.recs[i].bursts, self.mec.kA,
                self.recs[i].tres, self.recs[i].tcrit, **kwargs))
    
    def run(self):
        # FITTING BIT
        start = time.clock()
        result = minimize(self.dcprogslik, np.log(self.mec.theta()),
            method='Nelder-Mead', callback=self.printiter,
            options={'xtol':1e-4, 'ftol':1e-4, 'maxiter': 5000, 'maxfev': 10000,
            'disp': True})
        end = time.clock() - start
        # FITTING BIT
        print ("\n\nFitting with DCPROGS likelihood finished: %4d/%02d/%02d %02d:%02d:%02d"
            %time.localtime()[0:6])
        print '\n Time in simplex=', end
        print ('\n Final log-likelihood = {0:.6f}'.format(-result.fun))
        print ('\n Number of iterations = {0:d}'.format(result.nit))
        self.mec.theta_unsqueeze(np.exp(result.x))
        print "\n Final rate constants:"
        self.mec.printout(sys.stdout)

        self.report.fit_result(self.mec, start, end, result.fun, result.nit)
        self.report.rates(self.mec, result.fun, True)
        self.report.distributions(self.mec, self.recs)
        self.report.finalise()

    def printiter(self, theta):
        self.iternum += 1
        lik = self.dcprogslik(theta)
        print("iteration # {0:d}; log-lik = {1:.6f}".format(self.iternum, -lik))
        print(np.exp(theta))
    
    def dcprogslik(self, x, args=None):
        self.mec.theta_unsqueeze(np.exp(x))
        lik = 0
        for i in range(len(self.recs)):
            self.mec.set_eff('c', self.recs[i].conc)
            lik += -self.likelihood[i](self.mec.Q) * log(10)
        return lik
# end of FittingSession #

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

            if display and (iterations % 10) == 0:
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
    eigval = nplin.eigvals(h)
    ng = (eigval <= s).sum()
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
    if nga > 0: sa = sa * 4
    ngb = bisect_gFB(sb, tres, Q11, Q22, Q12, Q21, k1, k2)
    if ngb < k2: sb = sb / 4

    done = []
    todo = [[sa, sb, nga, ngb]]
#    nsplit = 0

#    while (len(done) < k1) and (nsplit < 1000):
    while todo:
        svv = todo.pop()
        sa1, sc, sb2, nga1, ngc, ngb2 = bisect_split(svv[0], svv[1], svv[2], svv[3],
            tres, Q11, Q22, Q12, Q21, k1, k2)
#        nsplit += 1

        # Check if either or both of the two subintervals output from
        # SPLIT contain only one root?
        if (ngc - nga1) == 1:
            done.append([sa1, sc])
#            if len(done) == k1:
#                break
        else:
            todo.append([sa1, sc, nga1, ngc])
        if (ngb2 - ngc) == 1:
            done.append([sc, sb2])
        else:
            todo.append([sc, sb2, ngc, ngb2])

    if len(done) < k1:
        sys.stderr.write(
            "bisectHJC: Warning: Only {0:d} roots out of {1:d} were located.".
            format(len(done), k1))
    return np.array(done)

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
