"""A collection of some HJCFit related functions."""

import numpy as np
from math import*

import qmatlib  as qml
import scalcslib as scl

def test_CHS(theta, opts):

    mec = opts['mec']
    conc = opts['conc']
    tres = opts['tres']
    tcrit = opts['tcrit']
    is_chsvec = opts['isCHS']

    mec.set_rateconstants(np.exp(theta))
    mec.set_eff('c', conc)

    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    expQFF = qml.expQt(mec.QFF, tres)
    expQAA = qml.expQt(mec.QAA, tres)
    print 'expQAA=', expQAA
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)
    phiF = qml.phiHJC(eGFA, eGAF, mec.kF)
    startB = qml.phiHJC(eGAF, eGFA, mec.kA)
    endB = np.ones((mec.kF, 1))

    Aeigvals, AZ00, AZ10, AZ11 = qml.Zxx(mec.Q, mec.kA, mec.QFF,
        mec.QAF, mec.QFA, expQFF, True)
    Aroots = scl.asymptotic_roots(tres,
        mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)
    Feigvals, FZ00, FZ10, FZ11 = qml.Zxx(mec.Q, mec.kA, mec.QAA,
        mec.QFA, mec.QAF, expQAA, False)
    Froots = scl.asymptotic_roots(tres,
        mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA)

    startB, endB = qml.CHSvec(Froots, tres, tcrit,
        mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF, phiF)
    print 'startB=', startB
    print 'endB=', endB

    t = 0.0010134001973
    print '\n\n opening t = ', t
    eGAFt = qml.eGAF(t, tres, Aeigvals, AZ00, AZ10, AZ11, Aroots,
        mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF, expQFF)
    print 'eGAFt=', eGAFt

    t = 0.0001
    print '\n\n shutting t = ', t
    eGAFt = qml.eGAF(t, tres, Feigvals, FZ00, FZ10, FZ11, Froots,
        mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA, expQAA)
    print 'eGAFt=', eGAFt


def HJClik(theta, bursts, opts):
    #HJClik(bursts, mec, tres, tcrit, is_chsvec=False):

    """
    Calculate likelihood for a series of open and shut times using HJC missed
    events probability density functions (first two dead time intervals- exact
    solution, then- asymptotic).

    Lik = phi * eGAF(t1) * eGFA(t2) * eGAF(t3) * ... * eGAF(tn) * uF
    where t1, t3,..., tn are open times; t2, t4,..., t(n-1) are shut times.

    Gaps > tcrit are treated as unusable (e.g. contain double or bad bit of
    record, or desens gaps that are not in the model, or gaps so long that
    next opening may not be from the same channel). However this calculation
    DOES assume that all the shut times predicted by the model are present
    within each group. The series of multiplied likelihoods is terminated at
    the end of the opening before an unusable gap. A new series is then
    started, using appropriate initial vector to give Lik(2), ... At end
    these are multiplied to give final likelihood.

    Parameters
    ----------
    bursts : dictionary
        A dictionary containing lists of open and shut intervals.
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).
    is_chsvec : bool
        True if CHS vectors should be used (Eq. 5.7, CHS96).

    Returns
    -------
    loglik : float
        Log-likelihood.
    """

    mec = opts['mec']
    conc = opts['conc']
    tres = opts['tres']
    tcrit = opts['tcrit']
    is_chsvec = opts['isCHS']

    mec.set_rateconstants(np.exp(theta))
    mec.set_eff('c', conc)

    # TODO: Here reset rates which reached limit or are negative.
    # TODO: Make new Q from theta.
    # TODO: Errors.

    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    expQFF = qml.expQt(mec.QFF, tres)
    expQAA = qml.expQt(mec.QAA, tres)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)
    phiF = qml.phiHJC(eGFA, eGAF, mec.kF)
    startB = qml.phiHJC(eGAF, eGFA, mec.kA)
    endB = np.ones((mec.kF, 1))

    Aeigvals, AZ00, AZ10, AZ11 = qml.Zxx(mec.Q, mec.kA, mec.QFF,
        mec.QAF, mec.QFA, expQFF, True)
    Aroots = scl.asymptotic_roots(tres,
        mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)
    Feigvals, FZ00, FZ10, FZ11 = qml.Zxx(mec.Q, mec.kA, mec.QAA,
        mec.QFA, mec.QAF, expQAA, False)
    Froots = scl.asymptotic_roots(tres,
        mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA)

    if is_chsvec:
        startB, endB = qml.CHSvec(Froots, tres, tcrit,
            mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF, phiF)

    loglik = 0
    for ind in bursts:
        burst = bursts[ind]
        grouplik = startB
        for i in range(len(burst)):
            t = burst[i] * 0.001
            if i % 2 == 0: # open time
                #eGAFt = np.zeros(Axaf[0].shape)
                eGAFt = qml.eGAF(t, tres, Aeigvals, AZ00, AZ10, AZ11, Aroots,
                    mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF, expQFF)
                #eGAFt = qml.eGAF(t, tres, Aroots, Axaf, Aeigvals, AZ00, AZ10, AZ11)
            else: # shut
                #eGAFt = np.zeros(Fxaf[0].shape)
                eGAFt = qml.eGAF(t, tres, Feigvals, FZ00, FZ10, FZ11, Froots,
                    mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA, expQAA)
                #eGAFt = qml.eGAF(t, tres, Froots, Fxaf, Feigvals, FZ00, FZ10, FZ11)
            grouplik = np.dot(grouplik, eGAFt)
            if grouplik.max() > 1e50:
                grouplik = grouplik * 1e-100
                print 'grouplik was scaled down'
        grouplik = np.dot(grouplik, endB)
        loglik += log(grouplik[0])
    return -loglik, np.log(mec.unit_rates())

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

def sortShell2(vals, simp):
    """
    Shell sort using Shell's (original) gap sequence: n/2, n/4, ..., 1.
    """
    n = np.size(vals)
    gap = n // 2
    while gap > 0:
         # do the insertion sort
         for i in range(gap, n):
             val = vals[i]
             tsimp = simp[i]
             j = i
             while j >= gap and vals[j - gap] > val:
                 vals[j] = vals[j - gap]
                 simp[j] = simp[j - gap]
                 j -= gap
             vals[j] = val
             simp[j] = tsimp
         gap //= 2
    return vals, simp

def simplexHJC(theta, data, func, opts, verbose=0):
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
    nevalmax = 1000
    niter = 0
    nrestartmax = 3    # max number of restarts
    nrestart = 0    # counts restarts
    L = 0
    niter = 0
    nitermax = 1000

    while nrestart < nrestartmax and L <= 1:

        fval[0], theta = func(theta, data, opts)
        print "Starting likelihood =", -fval[0]
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
            print 'reflection'

            if fnew < fval[0]:
                # ----- new vertex is better than previous best so extend it
                pnew1 = centre + extfac * (pnew - centre)
                fnew1, pnew1 = func(pnew1, data, opts)
                if fnew1 < absmin:
                    absmin = fnew1
                    thmin = pnew1
                neval += 1
                print 'extention'

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
                    print 'contraction'

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
                            print 'reduction'

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

    return np.exp(simp[0]), fval[0]


