"""A collection of some HJCFit related functions."""

import random
import numpy as np
from math import*

def printit(xk):
    print np.exp(xk)

def find_min(fval, theta, absmin, thmin):
    """
    """
    if fval < absmin:
        absmin = fval
        thmin = theta
    return absmin, thmin

def sortShell(vals, simp):
    """
    Shell sort using Shell's (original) gap sequence: n/2, n/4, ..., 1.
    """
    n = np.size(vals)
    gap = n // 2
    while gap > 0:
         # do the insertion sort
         for i in range(gap, n):
             val = vals[i]
             tsimp = np.zeros((n-1))
             for l in range(0, n-1):
                 tsimp[l] = simp[i,l]
             j = i
             while j >= gap and vals[j - gap] > val:
                 vals[j] = vals[j - gap]
                 simp[j] = simp[j - gap]
                 j -= gap
             vals[j] = val
             simp[j] = tsimp
         gap //= 2
    return vals, simp

def simplexHJC(func, theta, data, opts, verbose=0):
    """
    Python implementation of DC's SIMPHJC.FOR subroutine used in HJCFIT.
    Search for a function minimum using modified Nelder-Mead method.
    Theta contains only free parameters which should be natural logaritms
    of initial guesses.

    Parameters
    ----------
    theta :
    data :
    func :
    opts :

    Returns
    -------
    loglik :
    newtheta :
    """

    #TODO: these might come as parameters
    errfac = 1.e-3
    stpfac= 5   # 5 for logfit; initial step size factor; 1.01 < stpfac < 20
    stpfac = log(stpfac)
    reffac = 1.0    # reflection coeff => 1
    confac = 0.5     # contraction coeff 0 < beta < 1
    extfac = 2.0     # extension factor > 1
    resfac = 10.0

    k = np.size(theta)
    n = k + 1    # # of vertices in simplex
    simp = np.zeros((n, k))
    fval = np.zeros((n))
    pnew = np.zeros((k))
    pnew1 = np.zeros((k))
    thmin = np.zeros((k))

    step = np.ones((k)) * stpfac
    crtstp = np.ones((k)) * errfac
    # Factor to offset the vertices of the starting simplex
    fac = (sqrt(n) - 1.0) / (k * sqrt(2.0))

    neval = 0    # counts function evaluations
    nevalmax = 100000
    niter = 0
    nitermax = 10000
    nrestart = 0    # counts restarts
    nrestartmax = 3    # max number of restarts

    L = 0

    while nrestart < nrestartmax and L < 1:

        restart = False
        fval[0], theta = func(theta, data, opts)
        neval += 1
        print ("Starting likelihood = {0:.6f}".format(-fval[0]))
        simp[0] = theta
        absmin = fval[0]
        thmin = theta
     
        for i in range(1, n):
            # specify all other vertices of the starting simplex.
            simp[i] = simp[0] + step * fac
            simp[i, i-1] = simp[0, i-1] + step[i-1] * (fac + 1. / sqrt(2))
            #  and calculate their residuals.
            fval[i], simp[i] = func(simp[i], data, opts)
        neval += k
        # Sort simplex according residuals.
        fval, simp = sortShell(fval, simp)
        absmin, thmin = find_min(fval[0], simp[0], absmin, thmin)

        while L == 0 and niter < nitermax and restart == False:
            niter += 1
            if neval > nevalmax:
                print '\n No convergence after', neval, 'evaluations.'
                return simp[0], fval[0]

            # ----- compute centroid of all vertices except the worst
#            centre = np.zeros((k))
#            for i in range(k):
#                for j in range(n-1):
#                    centre[i] = centre[i] + simp[j,i]
#            centre = centre / float(k)
            centre = np.sum(simp[:-1,:], axis=0) / float(k)

            # ----- reflect, with next vertex taken as reflection of worst
            pnew = centre - reffac * (simp[-1] - centre)
            fnew, pnew = func(pnew, data, opts)
            absmin, thmin = find_min(fnew, pnew, absmin, thmin)
            neval += 1
#            print 'reflection'

            if fnew < fval[0]:
                # ----- new vertex is better than previous best so extend it
                pnew1 = centre + extfac * (pnew - centre)
                fnew1, pnew1 = func(pnew1, data, opts)
                absmin, thmin = find_min(fnew1, pnew1, absmin, thmin)
                neval += 1
#                print 'extention'

                if fnew1 < fnew:     # ----- still better
                    simp[-1] = pnew1
                    fval[-1] = fnew1
                else:
                    simp[-1] = pnew
                    fval[-1] = fnew
                # go for convergence check

            else:     # come here if reflected vertex not
                      # better than best vertex, so no extension wanted
                if fnew < fval[-2]:
                    simp[-1] = pnew
                    fval[-1] = fnew
                else:
                    if fnew < fval[-1]:
                        simp[-1] = pnew
                        fval[-1] = fnew
                    # Contract on the worst side of the centroid
                    pnew1 = centre + confac * (simp[-1] - centre)

                    fnew1, pnew1 = func(pnew1, data, opts)
                    absmin, thmin = find_min(fnew1, pnew1, absmin, thmin)
                    neval += 1
#                    print 'contraction'

                    # ----- is contracted vertex better than the worst vertex
                    if fnew1 <= fval[-1]:
                        simp[-1] = pnew1
                        fval[-1] = fnew1
                    else:
                        #  ----- no, it is still bad, shrink whole simplex towards best vertex
                        for i in range(n):
                            for j in range(k):
                                if j != i:
                                    simp[i,j] = simp[0,j] + confac * (simp[i,j] - simp[0,j])
                            fval[i], simp[i] = func(simp[i], data, opts)
                            neval += 1
#                            print 'reduction'

            fval, simp = sortShell(fval, simp)
            absmin, thmin = find_min(fval[0], simp[0], absmin, thmin)

            L, theta, val, step, restart = simplexHJC_converge(simp, fval, thmin,
                absmin, k, data, func, opts, crtstp, step, resfac,
                nrestart, nrestartmax)

            if (niter % 10) == 0:
                print ('iter# {0:d}\tlik= {1:f}'.format(niter, -fval[0]))
                print 'theta=', np.exp(simp[0])
        # end of iteration (while L == 0:)

        nrestart += 1

    return simp[0], fval[0], neval

def simplexHJC_converge(simp, fval, thmin, absmin, k,
    data, func, opts, crtstp, step, resfac, nrestart, nrestartmax):
    """
    Check simplexHJC convergence. This version uses difference between
    highest and lowest value of parameter of the n values that define a vertex.
    L=0 not converged - do next iteration
    L=1 converged via crtstp, but yet better vertex found
    L=2 converged via crtstp; end of fit
    L=3 for abort (no restarts)- not implemented yet.

    If convergence attained, options for ending in this version are:
    (1) look at current best vertex
    (2) look at param values averaged over vertices
    (3) look at absmin,thmin. If better, restart at absmin, as below.
    (4) do local search with each param +/- crtstp, as in O'Neill
        version, starting at current best vertex. If none are better
        input current best vertex. If some better restart at better
        value with crtstp taken as approptiately small initial step.
    """

    L = 1    #  conv via crtstp
    restart = False
    for i in range(k):
        if (simp[:,i].max() - simp[:,i].min() > fabs(crtstp[i])) : L = 0

    theta = simp[0]
    val = fval[0]

    if L == 1:
        
        exvals = np.empty((5))
        exvals[0] = fval[0]

        # next average over vertices-put values in pnew()
        pnew = np.sum(simp, axis=0) / float(k)
#            for j in range(k):
#                pnew[j] = 0.0
#                for i in range(n):
#                    pnew[j] = pnew[j] + simp[i,j]
#                pnew[j] = pnew[j] / float(n)
        fvalav, pnew = func(pnew, data, opts)
        exvals[1] = fvalav

        exvals[2] = absmin

        # do local search. Put altered values in pnew1
        pnew1 = simp[0] + crtstp
        fval1, pnew1 = func(pnew1, data, opts)
        if fval1 < fval[0]:
            exvals[3] = fval1
        else:
            # step in other direction
            pnew1 = simp[0] - crtstp
            fval1, pnew1 = func(pnew1, data, opts)
            exvals[3] = fval1

        rand = random.random()
        perfac = 0.3
	ranth = simp[0] * (rand * 2 * perfac + (1 - perfac))
        frand, ranth = func(ranth, data, opts)
        exvals[4] = frand

        # Test which is best.
        case = exvals.argmin()
        if case == 0:
            print '\n Returned with best vertex'

        elif case == 1:
            print '\n Returned with averaged vertices'
            theta = pnew
            val = fvalav

        elif case == 2:
            if nrestart >= nrestartmax:
                print '\n Returned with absolut minimum'
                theta = thmin
                val = absmin
            else:
                L = 0
                theta = thmin
                restart = True

        elif case == 3:
            if nrestart >= nrestartmax:
                print '\n Returned with result of local search minimum'
                theta = pnew1
                val = fval1
            else:
                L = 0
                theta = pnew1
                step = resfac * crtstp
                restart = True

        else:
            if nrestart >= nrestartmax:
                print '\n Returned with result of random perturbation.'
                theta = ranth
                val = frand
            else:
                L = 0
                theta = ranth
                restart = True

    return L, theta, val, step, restart
