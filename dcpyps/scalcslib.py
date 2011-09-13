"""A collection of functions for dwell time ideal, asymptotic and exact
probabulity density function calculations.

Notes
-----
DC_PyPs project are pure Python implementations of Q-Matrix formalisms
for ion channel research. To learn more about kinetic analysis of ion
channels see the references below.

References
----------
CH82: Colquhoun D, Hawkes AG (1982)
On the stochastic properties of bursts of single ion channel openings
and of clusters of bursts. Phil Trans R Soc Lond B 300, 1-59.

HJC92: Hawkes AG, Jalali A, Colquhoun D (1992)
Asymptotic distributions of apparent open times and shut times in a
single channel record allowing for the omission of brief events.
Phil Trans R Soc Lond B 337, 383-404.

CH95a: Colquhoun D, Hawkes AG (1995a)
The principles of the stochastic interpretation of ion channel
mechanisms. In: Single-channel recording. 2nd ed. (Eds: Sakmann B,
Neher E) Plenum Press, New York, pp. 397-482.

CH95b: Colquhoun D, Hawkes AG (1995b)
A Q-Matrix Cookbook. In: Single-channel recording. 2nd ed. (Eds:
Sakmann B, Neher E) Plenum Press, New York, pp. 589-633.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 20:29:14$"

import sys
import math

import numpy as np
from numpy import linalg as nplin

import qmatlib as qml
import bisectHJC
import pdfs

def ideal_dwell_time_pdf(t, QAA, phiA):
    """
    Probability density function of the open time.
    f(t) = phiOp * exp(-QAA * t) * (-QAA) * uA
    For shut time pdf A by F in function call.

    Parameters
    ----------
    t : float
        Time (sec).
    QAA : array_like, shape (kA, kA)
        Submatrix of Q.
    phiA : array_like, shape (1, kA)
        Initial vector for openings

    Returns
    -------
    f : float
    """

    kA = QAA.shape[0]
    uA = np.ones((kA, 1))
    expQAA = qml.expQt(QAA, t)
    f = np.dot(np.dot(np.dot(phiA, expQAA), -QAA), uA)
    return f

def ideal_dwell_time_pdf_components(QAA, phiA):
    """
    Calculate time constants and areas for an ideal (no missed events)
    exponential open time probability density function.
    For shut time pdf A by F in function call.

    Parameters
    ----------
    t : float
        Time (sec).
    QAA : array_like, shape (kA, kA)
        Submatrix of Q.
    phiA : array_like, shape (1, kA)
        Initial vector for openings

    Returns
    -------
    taus : ndarray, shape(k, 1)
        Time constants.
    areas : ndarray, shape(k, 1)
        Component relative areas.
    """

    kA = QAA.shape[0]
    areas = np.zeros(kA)
    eigs, A = qml.eigs(-QAA)
    uA = np.ones((kA, 1))
    #TODO: remove 'for'
    for i in range(kA):
        areas[i] = (np.dot(np.dot(np.dot(phiA, A[i]),
            (-QAA)), uA) / eigs[i])

    taus = 1 / eigs
    return taus, areas

def ideal_subset_time_pdf(Q, k1, k2, t):
    """
    """
    
    u = np.ones((k2 - k1 + 1, 1))
    phi, QSub = qml.phiSub(Q, k1, k2)
    expQSub = qml.expQt(QSub, t)
    f = np.dot(np.dot(np.dot(phi, expQSub), -QSub), u)
    return f

def ideal_subset_mean_life_time(Q, state1, state2):
    """
    Calculate mean life time in a specified subset. Add all rates out of subset
    to get total rate out. Skip rates within subset.

    Parameters
    ----------
    mec : instance of type Mechanism
    state1,state2 : int
        State numbers (counting origin 1)

    Returns
    -------
    mean : float
        Mean life time.
    """

    k = Q.shape[0]
    p = qml.pinf(Q)
    # Total occupancy for subset.
    pstot = np.sum(p[state1-1 : state2])

    # Total rate out
    s = 0.0
    for i in range(state1-1, state2):
        for j in range(k):
            if (j < state1-1) or (j > state2 - 1):
                s += Q[i, j] * p[i] / pstot

    mean = 1 / s
    return mean

def ideal_mean_latency_given_start_state(mec, state):
    """
    Calculate mean latency to next opening (shutting), given starting in
    specified shut (open) state.

    mean latency given starting state = pF(0) * inv(-QFF) * uF

    F- all shut states (change to A for mean latency to next shutting
    calculation), p(0) = [0 0 0 ..1.. 0] - a row vector with 1 for state in
    question and 0 for all other states.

    Parameters
    ----------
    mec : instance of type Mechanism
    state : int
        State number (counting origin 1)

    Returns
    -------
    mean : float
        Mean latency.
    """

    if state <= mec.kA:
        # for calculating mean latency to next shutting
        p = np.zeros((mec.kA))
        p[state-1] = 1
        uF = np.ones((mec.kA, 1))
        invQFF = nplin.inv(-mec.QAA)
    else:
        # for calculating mean latency to next opening
        p = np.zeros((mec.kF))
        p[state-mec.kA-1] = 1
        uF = np.ones((mec.kF, 1))
        invQFF = nplin.inv(-mec.QFF)

    mean = np.dot(np.dot(p, invQFF), uF)[0]

    return mean

def asymptotic_roots(mec, tres, open):
    """
    Find roots for the asymptotic probability density function (Eqs. 52-58,
    HJC92).

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    open : bool
        True if searching roots for open time pdf; False if searching roots
        for shut time distribution.

    Returns
    -------
    roots : array_like, shape (k,)
    """

    sao = -100000
    sbo = -10
    sas = -100000
    sbs = -0.0001

    if open:
        sro = bisectHJC.bisection_intervals(sao, sbo, tres,
            mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)
        roots = np.zeros(mec.kA)
        for i in range(mec.kA):
            roots[i] = bisectHJC.bisect(sro[i,0], sro[i,1], tres,
                mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)
    else:
        sro = bisectHJC.bisection_intervals(sas, sbs, tres,
            mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA)
        roots = np.zeros(mec.kF)
        for i in range(mec.kF):
            roots[i] = bisectHJC.bisect(sro[i,0], sro[i,1], tres,
                mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA)

    return roots

def asymptotic_areas(mec, tres, roots, open):
    """
    Find the areas of the asymptotic pdf (Eq. 58, HJC92).

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    roots : array_like, shape (k,)
        Roots of the asymptotic pdf.
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    open : bool
        True if searching roots for open time pdf; False if searching roots
        for shut time distribution.

    Returns
    -------
    areas : array_like, shape (k,)
    """

    if open:
        k1 = mec.kA
        k2 = mec.kF
        Q22 = mec.QFF
        Q11 = mec.QAA
        Q12 = mec.QAF
        Q21 = mec.QFA
        G12, G21 = qml.iGs(mec.Q, mec.kA, mec.kF)
    else:
        k1 = mec.kF
        k2 = mec.kA
        Q11 = mec.QFF
        Q22 = mec.QAA
        Q21 = mec.QAF
        Q12 = mec.QFA
        G21, G12 = qml.iGs(mec.Q, mec.kA, mec.kF)

    expQ22 = qml.expQt(Q22, tres)
    expQ11 = qml.expQt(Q11, tres)
    eG12 = qml.eGs(G12, G21, k1, k2, expQ22)
    eG21 = qml.eGs(G21, G12, k2, k1, expQ11)
    phi1 = qml.phiHJC(eG12, eG21, k1)

    areas = np.zeros(k1)
    rowA = np.zeros((k1,k1))
    colA = np.zeros((k1,k1))
    for i in range(k1):
        WA = qml.W(roots[i], tres,
            Q11, Q22, Q12, Q21, k1, k2)
            #Q22, Q11, Q21, Q12, k2, k1)
        rowA[i] = qml.pinf(WA)
        AW = np.transpose(WA)
        colA[i] = qml.pinf(AW)

    for i in range(k1):
        u2 = np.ones((k2,1))
        nom = np.dot(np.dot(np.dot(np.dot(np.dot(phi1, colA[i]), rowA[i]),
            Q12), expQ22), u2)
        W1A = qml.dW(roots[i], tres, Q12, Q22, Q21, k1, k2)
        denom = -roots[i] * np.dot(np.dot(rowA[i], W1A), colA[i])
        areas[i] = nom / denom

    return areas

def hjc_mean_time(mec, tres, open):
    """
    Calculate exact mean open or shut time from HJC probability density
    function.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    tres : float
        Time resolution (dead time).
    open : bool
        True to calculate mean open time, False to calculate mean shut time.

    Returns
    -------
    mean : float
        Apparent mean open/shut time.
    """

    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    expQFF = qml.expQt(mec.QFF, tres)
    expQAA = qml.expQt(mec.QAA, tres)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)

    if open:
        phiA = qml.phiHJC(eGAF, eGFA, mec.kA)
        QexpQF = np.dot(mec.QAF, expQFF)
        DARS = qml.dARSdS(tres, mec.QAA, mec.QFF,
            GAF, GFA, expQFF, mec.kA, mec.kF)
        uF = np.ones((mec.kF, 1))
        # meanOpenTime = tres + phiA * DARS * QexpQF * uF
        mean = tres + np.dot(phiA, np.dot(np.dot(DARS, QexpQF), uF))
    else:
        phiF = qml.phiHJC(eGFA, eGAF, mec.kF)
        QexpQA = np.dot(mec.QFA, expQAA)
        DFRS = qml.dARSdS(tres, mec.QFF, mec.QAA,
            GFA, GAF, expQAA, mec.kF, mec.kA)
        uA = np.ones((mec.kA, 1))
        # meanShutTime = tres + phiF * DFRS * QexpQA * uA
        mean = tres + np.dot(phiF, np.dot(np.dot(DFRS, QexpQA), uA))

    return mean[0]


def pdf_exact(t, tres, roots, areas, eigvals, gamma00, gamma10, gamma11):
    r"""
    Calculate exponential probabolity density function with exact solution for 
    missed events correction (Eq. 21, HJC92).

    .. math::
       :nowrap:

       \begin{align*}
       f(t) =
       \begin{cases} 
       f_0(t)                          & \text{for}\; 0 \leq t \leq t_\text{res} \\
       f_0(t) - f_1(t - t_\text{res})  & \text{for}\; t_\text{res} \leq t \leq 2 t_\text{res}
       \end{cases}
       \end{align*}

    Parameters
    ----------
    t : float
        Time.
    tres : float
        Time resolution (dead time).
    roots : array_like, shape (k,)
    areas : array_like, shape (k,)
    eigvals : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    gama00, gama10, gama11 : lists of floats
        Constants for the exact open/shut time pdf.

    Returns
    -------
    f : float
    """

    if t < tres:
        f = 0
    elif ((tres < t) and (t < (2 * tres))):
        f = qml.f0((t - tres), eigvals, gamma00)
    elif ((tres * 2) < t) and (t < (3 * tres)):
        f = (qml.f0((t - tres), eigvals, gamma00) -
            qml.f1((t - 2 * tres), eigvals, gamma10, gamma11))
    else:
        f = pdfs.expPDF(t - tres, -1 / roots, areas)
    return f

def GAMAxx(mec, tres, open):
    """
    Calculate gama constants for the exact open/shut time pdf (Eq. 3.22, HJC90).

    Parameters
    ----------
    tres : float
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    open : bool
        True for open time pdf and False for shut time pdf.

    Returns
    -------
    eigen : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    gama00, gama10, gama11 : lists of floats
        Constants for the exact open/shut time pdf.
    """

    k = mec.Q.shape[0]
    expQFF = qml.expQt(mec.QFF, tres)
    expQAA = qml.expQt(mec.QAA, tres)
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)

    gama00 = []
    gama10 = []
    gama11 = []

    if open:
        phiA = qml.phiHJC(eGAF, eGFA, mec.kA)
        eigen, Z00, Z10, Z11 = qml.Zxx(mec.Q, mec.kA,
            mec.QFF, mec.QAF, mec.QFA, expQFF)
        uF = np.ones((mec.kF,1))
        for i in range(k):
            gama00.append(np.dot(np.dot(phiA, Z00[i]), uF)[0])
            gama10.append(np.dot(np.dot(phiA, Z10[i]), uF)[0])
            gama11.append(np.dot(np.dot(phiA, Z11[i]), uF)[0])

#        gama00 = np.array([np.dot(np.dot(phiA, Z), uF)[0] for Z in Z00])
#        gama10 = np.array([np.dot(np.dot(phiA, Z), uF)[0] for Z in Z10])
#        gama11 = np.array([np.dot(np.dot(phiA, Z), uF)[0] for Z in Z11])

    else:
        phiF = qml.phiHJC(eGFA, eGAF, mec.kF)
        eigen, Z00, Z10, Z11 = qml.Zxx(mec.Q, mec.kA,
            mec.QAA, mec.QFA, mec.QAF, expQAA)
        uA = np.ones((mec.kA, 1))
        for i in range(k):
            gama00.append(np.dot(np.dot(phiF, Z00[i]), uA)[0])
            gama10.append(np.dot(np.dot(phiF, Z10[i]), uA)[0])
            gama11.append(np.dot(np.dot(phiF, Z11[i]), uA)[0])
#        gama00 = np.array([np.dot(np.dot(phiF, Z), uA)[0] for Z in Z00])
#        gama10 = np.array([np.dot(np.dot(phiF, Z), uA)[0] for Z in Z10])
#        gama11 = np.array([np.dot(np.dot(phiF, Z), uA)[0] for Z in Z11])

    return eigen, np.array(gama00), np.array(gama10), np.array(gama11)

def ini_vectors(mec, eGFA, eGAF, expQFF, XFA,
        roots, tres, tcrit, is_chsvec=False):
    """
    Get initial and final vectors, startB and endB, for HJC likelihood
    calculation (Eqs. 5.5 or 5.7, CHS96).

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).
    tcrit : float
        Critical gap length (critical shut time).
    is_chsvec : bool
        True if CHS vectors should be used (Eq. 5.7, CHS96).

    Returns
    -------
    startB : ndarray, shape (1, kA)
        Initial vector for openings or initial CHS vector (Eq. 5.11, CHS96).
    endB : ndarray, shape (kF, 1)
        Column of 1's or final CHS vector (Eq. 5.8, CHS96).
    """

#    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
#    expQFF = qml.expQt(mec.QFF, tres)
#    expQAA = qml.expQt(mec.QAA, tres)
#    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
#    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)
    uA = np.ones((mec.kA, 1))

    if is_chsvec:
#        roots = asymptotic_roots(mec, tres, False)
        HFA = np.zeros((mec.kF, mec.kA))
#        XFA = qml.XAF(tres, roots, mec.QFF, mec.QAA, mec.QFA,
#            mec.QAF, expQFF)
        for i in range(mec.kF):
            coeff = -math.exp(roots[i] * (tcrit - tres)) / roots[i]
            HFA += coeff * XFA[i]
        phiF = qml.phiHJC(eGFA, eGAF, mec.kF)

        startB = np.dot(phiF, HFA) / np.dot(np.dot(phiF, HFA), uA)
        endB = np.dot(HFA, uA)
    else:
        startB = qml.phiHJC(eGAF, eGFA, mec.kA)
        endB = np.ones((mec.kF, 1))

    return startB, endB

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


    Aeigvals, AZ00, AZ10, AZ11 = qml.Zxx(mec.Q, mec.kA, mec.QFF,
        mec.QAF, mec.QFA, expQFF)
    Aroots = asymptotic_roots(mec, tres, True)
    Axaf = qml.XAF(tres, Aroots, mec.QAA, mec.QFF, mec.QAF, mec.QFA, expQFF)
    Feigvals, FZ00, FZ10, FZ11 = qml.Zxx(mec.Q, mec.kA, mec.QAA,
        mec.QFA, mec.QAF, expQAA)
    Froots = asymptotic_roots(mec, tres, False)
    Fxaf = qml.XAF(tres, Froots, mec.QFF, mec.QAA, mec.QFA, mec.QAF, expQAA)
    startB, endB = ini_vectors(mec, eGFA, eGAF, expQAA, Fxaf, Froots,
        tres, tcrit, is_chsvec)
#    print 'startB=', startB
#    print 'endB=', endB

    loglik = 0
    for ind in bursts:
        burst = bursts[ind]
        grouplik = startB
        for i in range(len(burst)):
            t = burst[i] * 0.001
            if i % 2 == 0: # open time
                #eGAFt = np.zeros(Axaf[0].shape)
                eGAFt = qml.eGAF(t, tres, Aroots, Axaf, Aeigvals, AZ00, AZ10, AZ11)
            else: # shut
                #eGAFt = np.zeros(Fxaf[0].shape)
                eGAFt = qml.eGAF(t, tres, Froots, Fxaf, Feigvals, FZ00, FZ10, FZ11)
            grouplik = np.dot(grouplik, eGAFt)
            if grouplik.max() > 1e50:
                grouplik = grouplik * 1e-100
                print 'grouplik was scaled down'
        grouplik = np.dot(grouplik, endB)
        loglik += math.log(grouplik[0])
    return -loglik, np.log(mec.unit_rates())





def printout(mec, output=sys.stdout, eff='c'):
    """
    """

    output.write('\n\nOpen\tEquilibrium\tMean life\tMean latency (ms)')
    output.write('\nstate\toccupancy\t(ms)\tto next shutting')
    output.write('\n\t\t\tgiven start in this state')

    pinf = qml.pinf(mec.Q)

    for i in range(mec.k):
        if i == 0:
            mean_life_A = ideal_subset_mean_life_time(mec.Q, 1, mec.kA)
            output.write('\nSubset A ' +
                '\t{0:.6f}'.format(np.sum(pinf[:mec.kA])) +
                '\t{0:.6f}'.format(mean_life_A * 1000) +
                '\n')
        if i == mec.kA:
            mean_life_B = ideal_subset_mean_life_time(mec.Q, mec.kA + 1, mec.kA + mec.kB)
            output.write('\n\nShut\tEquilibrium\tMean life\tMean latency (ms)')
            output.write('\nstate\toccupancy\t(ms)\tto next opening')
            output.write('\n\t\t\tgiven start in this state')
            output.write('\nSubset B ' +
                '\t{0:.6f}'.format(np.sum(pinf[mec.kA:mec.kA+mec.kB])) +
                '\t{0:.6f}'.format(mean_life_B * 1000) +
                '\n')
        if i == mec.kE:
            mean_life_C = ideal_subset_mean_life_time(mec.Q, mec.kA + mec.kB + 1, mec.k)
            output.write('\n\nSubset C ' +
                '\t{0:.6f}'.format(np.sum(pinf[mec.kA+mec.kB:mec.k])) +
                '\t{0:.6f}'.format(mean_life_C * 1000) +
                '\n')
        mean = ideal_mean_latency_given_start_state(mec, i+1)
        output.write('\n{0:d}'.format(i+1) +
            '\t{0:.6f}'.format(pinf[i]) +
            '\t{0:.6f}'.format(-1 / mec.Q[i,i] * 1000) +
            '\t{0:.6f}'.format(mean * 1000))

