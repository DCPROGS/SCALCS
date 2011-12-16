"""A collection of some HJCFit related functions."""

import numpy as np
from math import*

def printit(xk):
    print np.exp(xk)

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
    reffac = 1.0    # reflection coeff => 1
    confac = 0.5     # contraction coeff 0 < beta < 1
    extfac = 2.0     # extension factor > 1
    resfac = 10.0

    stpfac = log(stpfac)

    k = np.size(theta)
    n = k + 1    # # of vertices in simplex
    simp = np.zeros((n, k))
    fval = np.zeros((n))
    pnew = np.zeros((k))
    pnew1 = np.zeros((k))
    thmin = np.zeros((k))

    step = np.ones((k)) * stpfac
    crtstp = np.ones((k)) * errfac

    # Compute offset of the vertices of the starting simplex
    fac = (sqrt(n) - 1.0) / (k * sqrt(2.0))

    neval = 0    # counts function evaluations
    nevalmax = 100000
    niter = 0
    nrestartmax = 3    # max number of restarts
    nrestart = 0    # counts restarts
    L = 0
    niter = 0
    nitermax = 10000

    while nrestart < nrestartmax and L <= 1:

        fval[0], theta = func(theta, data, opts)
        print ("Starting likelihood = {0:.6f}".format(-fval[0]))
        neval += 1
        simp[0] = theta
        
        fsav = fval[0]

        absmin = fval[0]
        thmin = theta
     

        # specify all other vertices of the starting simplex.
        for i in range(1, n):
            simp[i] = simp[0] + step * fac
            simp[i, i-1] = simp[0, i-1] + step[i-1] * (fac + 1. / sqrt(2))
            #  and calculate their residuals.
            fval[i], simp[i] = func(simp[i], data, opts)
        neval += k
        # Sort simplex according residuals.
        fval, simp = sortShell(fval, simp)
        if fval[0] < absmin:
            absmin = fval[0]
            thmin = simp[0]

        while L == 0 and niter < nitermax:
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
            if fnew < absmin:
                absmin = fnew
                thmin = pnew
            neval += 1
#            print 'reflection'

            if fnew < fval[0]:
                # ----- new vertex is better than previous best so extend it
                pnew1 = centre + extfac * (pnew - centre)
                fnew1, pnew1 = func(pnew1, data, opts)
                if fnew1 < absmin:
                    absmin = fnew1
                    thmin = pnew1
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
                    if fnew1 < absmin:
                        absmin = fnew1
                        thmin = pnew1
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
            if fval[0] < absmin:
                absmin = fval[0]
                thmin = simp[0]


            # CHECK CONVERGENCE.
            # This version uses diff between highest and lowest value of
            # parameter of the n values that define a vertex.
            # L=0 for not converged - do next iteration
            # L=1 for converged via crtstp
            # L=3 for abort (no restarts)

            L = 1    #  conv via crtstp
            for j in range(k):     # test each parameter
                if(simp[-1,j] - simp[0,j]) > fabs(crtstp[j]): L = 0 # not conv
#            diff = simp[-1] - simp[0]
#            if np.any(np.less_equal(diff, np.fabs(crtstp))): L = 0

            if (niter % 10) == 0:
                print 'iter#', niter, 'f=', -fval[0], 'theta', np.exp(simp[0])
        # end of iteration (while L == 0:)

        # ----- convergence attained. Options for ending in this version are:
        #       (1)look at current best vertex
        #       (2)look at param values averaged over vertices
        #       (3)look at absmin,thmin. If better, restart at absmin, as below.
        #       (4)do local search with each param +/- crtstp, as in O'Neill
        #        version, starting at current best vertex. If none are better
        #        input current best vertex. If some better restart at better
        #        value with crtstp taken as approptiately small initial step.


        if L == 1:
            exvals = []
            exvals.append(fval[0])

            # next average over vertices-put values in pnew()
            pnew = np.sum(simp, axis=0) / float(k)
#            for j in range(k):
#                pnew[j] = 0.0
#                for i in range(n):
#                    pnew[j] = pnew[j] + simp[i,j]
#                pnew[j] = pnew[j] / float(n)
            fvalav, pnew = func(pnew, data, opts)
            exvals.append(fvalav)

            exvals.append(absmin)

            # do local search. Put altered values in pnew1
            pnew1 = simp[0] + crtstp
            fval1, pnew1 = func(pnew1, data, opts)
            if fval1 < fval[0]:
                exvals.append(fval1)
            else:
                # step in other direction
                pnew1 = simp[0] - crtstp 
                fval1, pnew1 = func(pnew1, data, opts)
                exvals.append(fval1)

            # Test which is best.
            il = 0
            for i in range(1, 4):
                if exvals[i] < exvals[il]: il = i
            if il == 0:
                if nrestart == nrestartmax or fsav == fval[0]:
                    print '\n Returned with best vertex'
                    return simp[0], fval[0]
                else:
                    L = 0
                    theta = simp[0]
                    print '\n Restarted at best vertex'
            elif il == 1:
                if nrestart == nrestartmax or fsav == fvalav:
                    print '\n Returned with averaged vertices'
                    return pnew, fvalav
                else:
                    L = 0
                    theta = pnew
                    print '\n Restarted at averaged vertices'
            elif il == 2:
                if nrestart == nrestartmax or fsav == absmin:
                    print '\n Returned with absolut minimum'
                    return thmin, absmin
                else:
                    L = 0
                    theta = thmin
                    print '\n Restarted at absolut minimum'
            else:
                if nrestart == nrestartmax or fsav == fval1:
                    print '\n Returned with result of local search minimum'
                    return pnew1, fval1
                else:
                    L = 0
                    theta = pnew1
                    print '\n Restarted at result of local search minimum'
                    step = resfac * crtstp

        nrestart += 1

    return simp[0], fval[0]


