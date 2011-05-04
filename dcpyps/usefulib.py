"""A collection of some useful functions used in DC_PyPs project."""

import numpy as np
from math import*

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

def simplex(theta, data, func, opts, verbose=0):
    """
    Python implementation of DC's SIMPLEXV.FOR subroutine.
    """

    print '\n USING FAST VERSION OF SIMPLEX'

    #TODO: these might come as parameters
    errfac = 1.e-3   #1.e-4
    stpfac= 0.1   #0.1
    reffac = 2    # reflection coeff => 1
    confac = 0.5     # contraction coeff 0 < beta < 1
    extfac = 2     # extension factor > 1
    locfac = 5    # 1

    k = np.size(theta)
    n = k + 1    # # of vertices in simplex
    simp = np.zeros((n, k))
    fval = np.zeros((n))
    step = np.zeros((k))
    crtstp = np.zeros((k))
    pnew = np.zeros((k))
    pnew1 = np.zeros((k))
    thmin = np.zeros((k))

#    for j in range(k):
#        step[j] = stpfac * theta[j]
#	crtstp[j] = errfac * theta[j]
    step = stpfac * theta
    crtstp = errfac * theta

    neval = 0	 # counts function evaluations
    nrestart = 100    # max number of restarts
    irestart = 0    # counts restarts

    while irestart < nrestart: 

        irestart += 1
        print 'RESTART#', irestart

        simp[0] = theta
        fval[0] = func(theta, data, opts)
        fsav = fval[0]
        absmin = fval[0]
        thmin = theta
        neval += 1

        # compute offset of the vertices of the starting simplex
	fac = (sqrt(n) - 1) / (k * sqrt(2))

        #specify all other vertices of the starting simplex
        for i in range(1, n):
#            for j in range(k):
#                simp[i, j] = simp[0, j] + step[j] * fac
#            simp[i, i-1] = simp[0, i-1] + step[i-1] * (fac + 1. / sqrt(2))
            simp[i] = simp[0] + step * fac
            simp[i, i-1] = simp[0, i-1] + step[i-1] * (fac + 1. / sqrt(2))

            #  and calculate their residuals
            fval[i] = func(simp[i], data, opts)
        neval += k

        if verbose: print '\n simplex at the beginning of restart='
        if verbose: print simp
        if verbose: print ' fval at the begining', fval

        fval, simp = sortShell(fval, simp)
        if fval[0] < absmin:
            absmin = fval[0]
            thmin = simp[0]

        L = 0
        niter	= 0
        while L == 0:
            niter += 1
            print 'iter#', niter, 'f=', fval[0], 'theta', simp[0]

            # ----- compute centroid of all vertices except the worst
            centre = np.zeros((k))
            for i in range(k):
                for j in range(n-1):
                    centre[i] = centre[i] + simp[j,i]

            # ----- reflect, with next vertex taken as reflection of worst
#            for j in range(k):
#                centre[j] = centre[j] / float(k)
#                pnew[j] = centre[j] - reffac * (simp[-1, j] - centre[j])
            centre = centre / float(k)
            pnew = centre - reffac * (simp[-1] - centre)

            fnew = func(pnew, data, opts)
            if fnew < absmin:
                absmin = fnew
                thmin = pnew
            neval += 1
            if verbose:
                print 'reflection: e#', neval, 'f=',fnew, 'pnew=', pnew

            if fnew < fval[0]:
                # ----- new vertex is better than previous best so extend it
#                for j in range(k):
#                    pnew1[j] = centre[j] + extfac * (pnew[j] - centre[j])
                pnew1 = centre + extfac * (pnew - centre)

                fnew1 = func(pnew1, data, opts)
                if fnew1 < absmin:
                    absmin = fnew1
                    thmin = pnew1
                neval += 1
                if verbose:
                    print 'extention: e#', neval, 'f1=',fnew1, 'pnew1=', pnew1

                if fnew1 < fnew:     # ----- still better
                    simp[-1] = pnew1
                    fval[-1] = fnew1
                else:
                    simp[-1] = pnew
                    fval[-1] = fnew
                fval, simp = sortShell(fval, simp)
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
                    # Contract on the original fval(IHI) side of the centroid
#                    for j in range(k):
#                        pnew1[j] = centre[j] + confac * (simp[-1, j] - centre[j])
                    pnew1 = centre + confac * (simp[-1] - centre)

                    fnew1 = func(pnew1, data, opts)
                    if fnew1 < absmin:
                        absmin = fnew1
                        thmin = pnew1
                    neval += 1
                    if verbose:
                        print 'contract: e#', neval, 'f=',fnew1, 'pnew1=', pnew1

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
                            fval[i] = func(simp[i], data, opts)
                            neval += 1
                            if verbose:
                                print 'reduction: e#', neval, 'f=',fval[i], 'theta=', theta

            fval, simp = sortShell(fval, simp)
            if fval[0] < absmin:
                absmin = fval[0]
                thmin = simp[0]


            # CHECK CONVERGENCE. IF NOT CONVERGED GOTO 2000.
            # This version uses diff between highest and lowest value
            # of parameter of the n values that define a vertex
            # (as in O'Neill version)

            #  ----- order the vertices for all vertices
            # Define L=0 for not converged- do next iteration
            # L=1 for converged via crtstp
            # L=2 for converged via delmin (no restarts)
            # L=3 for abort (no restarts)

            L = 1    #  conv via crtstp
            for j in range(k):     # test each parameter
                if(simp[-1,j] - simp[0,j]) > fabs(crtstp[j]): L = 0 # not conv
        # end of iteration (while L == 0:)

        # ----- convergence attained. Options for ending in this version are:
        # 	(1)look at current best vertex
        # 	(2)look at param values averaged over vertices
        # 	(3)look at absmin,thmin. If better, restart at absmin, as below.
        # 	(4)do local search with each param +/- crtstp, as in O'Neill
        # 	 version, starting at current best vertex. If none are better
        # 	 input current best vertex. If some better restart at better
        # 	 value with crtstp taken as approptiately small initial step.

        if L == 1:
            exvals = []
            exvals.append(fval[0])

            # next average over vertices-put values in pnew()
            for j in range(k):
                pnew[j] = 0.0
                for i in range(n):
                    pnew[j] = pnew[j] + simp[i,j]
                pnew[j] = pnew[j] / float(n)
            fvalav = func(pnew, data, opts)
            exvals.append(fvalav)

            exvals.append(absmin)

            # do local search. Put altered values in pnew1
            for j in range(k):
                pnew1[j] = 0.0
                pnew1[j] = simp[0,j] + locfac * crtstp[j]
            fval1 = func(pnew1, data, opts)
            if fval1 < fval[0]:
                exvals.append(fval1)
            else:
                for j in range(k):
                    # step in other direction
                    pnew1[j] = simp[0,j] - locfac * crtstp[j] 
                fval1 = func(pnew1, data, opts)
                exvals.append(fval1)

            # Test which is best.
            il = 0
            for i in range(1, 4):
                if exvals[i] < exvals[il]: il = i
            if il == 0:
                if irestart == nrestart or fsav == fval[0]:
                    print '\n Returned with best vertex'
                    return simp[0], fval[0]
                else:
                    L = 0
                    theta = simp[0]
                    print '\n Restarted at best vertex'
            elif il == 1:
                if irestart == nrestart or fsav == fvalav:
                    print '\n Returned with averaged vertices'
                    return pnew, fvalav
                else:
                    L = 0
                    theta = pnew
                    print '\n Restarted at averaged vertices'
            elif il == 2:
                if irestart == nrestart or fsav == absmin:
                    print '\n Returned with absolut minimum'
                    return thmin, absmin
                else:
                    L = 0
                    theta = thmin
                    print '\n Restarted at absolut minimum'
            else:
                if irestart == nrestart or fsav == fval1:
                    print '\n Returned with result of local search minimum'
                    return pnew1, fval1
                else:
                    L = 0
                    theta = pnew1
                    print '\n Restarted at result of local search minimum'

