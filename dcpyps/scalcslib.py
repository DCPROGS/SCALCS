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

CH87: Colquhoun D, Hawkes AG (1987)
A note on correlations in single ion channel records.
Proc R Soc Lond 230, 15-52.

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
from math import*
from decimal import*

import scipy.optimize as so
import numpy as np
from numpy import linalg as nplin

import qmatlib as qml
#import bisectHJC
import pdfs
import optimize
import dcstatslib as stl

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
    w = np.zeros(kA)
    eigs, A = qml.eigs(-QAA)
    uA = np.ones((kA, 1))
    #TODO: remove 'for'
    for i in range(kA):
        w[i] = np.dot(np.dot(np.dot(phiA, A[i]), (-QAA)), uA)

    return eigs, w

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

def asymptotic_pdf(t, tres, tau, area):
    """
    Calculate asymptotic probabolity density function.

    Parameters
    ----------
    t : ndarray.
        Time.
    tres : float
        Time resolution.
    tau : ndarray, shape(k, 1)
        Time constants.
    area : ndarray, shape(k, 1)
        Component relative area.

    Returns
    -------
    apdf : ndarray.
    """
    t1 = np.extract(t[:] < tres, t)
    t2 = np.extract(t[:] >= tres, t)
    apdf2 = t2 * pdfs.expPDF(t2 - tres, tau, area)
    apdf = np.append(t1 * 0.0, apdf2)

    return apdf

def asymptotic_roots(tres, QAA, QFF, QAF, QFA, kA, kF):
    """
    Find roots for the asymptotic probability density function (Eqs. 52-58,
    HJC92).

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    QAA : array_like, shape (kA, kA)
    QFF : array_like, shape (kF, kF)
    QAF : array_like, shape (kA, kF)
    QFA : array_like, shape (kF, kA)
        QAA, QFF, QAF, QFA - submatrices of Q.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.

    Returns
    -------
    roots : array_like, shape (1, kA)
    """

    sas = -1000000
    sbs = -0.0000001
    sro = optimize.bisect_intervals(sas, sbs, tres,
        QAA, QFF, QAF, QFA, kA, kF)

    roots = np.zeros(kA)
    for i in range(kA):
        roots[i] = so.bisect(qml.detW, sro[i,0], sro[i,1],
            args=(tres, QAA, QFF, QAF, QFA, kA, kF))
#        roots[i] = bisectHJC.bisect(sro[i,0], sro[i,1], tres,
#            QAA, QFF, QAF, QFA, kA, kF)
    return roots

def asymptotic_areas(tres, roots, QAA, QFF, QAF, QFA, kA, kF, GAF, GFA):
    """
    Find the areas of the asymptotic pdf (Eq. 58, HJC92).

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    roots : array_like, shape (1,kA)
        Roots of the asymptotic pdf.
    QAA : array_like, shape (kA, kA)
    QFF : array_like, shape (kF, kF)
    QAF : array_like, shape (kA, kF)
    QFA : array_like, shape (kF, kA)
        QAA, QFF, QAF, QFA - submatrices of Q.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.
    GAF : array_like, shape (kA, kB)
    GFA : array_like, shape (kB, kA)
        GAF, GFA- transition probabilities

    Returns
    -------
    areas : ndarray, shape (1, kA)
    """

    expQFF = qml.expQt(QFF, tres)
    expQAA = qml.expQt(QAA, tres)
    eGAF = qml.eGs(GAF, GFA, kA, kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, kF, kA, expQAA)
    phiA = qml.phiHJC(eGAF, eGFA, kA)
    R = qml.AR(roots, tres, QAA, QFF, QAF, QFA, kA, kF)
    uF = np.ones((kF,1))
    areas = np.zeros(kA)
    for i in range(kA):
        areas[i] = ((-1 / roots[i]) *
            np.dot(phiA, np.dot(np.dot(R[i], np.dot(QAF, expQFF)), uF)))

#    rowA = np.zeros((kA,kA))
#    colA = np.zeros((kA,kA))
#    for i in range(kA):
#        WA = qml.W(roots[i], tres,
#            QAA, QFF, QAF, QFA, kA, kF)
#        rowA[i] = qml.pinf(WA)
#        AW = np.transpose(WA)
#        colA[i] = qml.pinf(AW)
#
#    for i in range(kA):
#        uF = np.ones((kF,1))
#        nom = np.dot(np.dot(np.dot(np.dot(np.dot(phiA, colA[i]), rowA[i]),
#            QAF), expQFF), uF)
#        W1A = qml.dW(roots[i], tres, QAF, QFF, QFA, kA, kF)
#        denom = -roots[i] * np.dot(np.dot(rowA[i], W1A), colA[i])
#        areas[i] = nom / denom

    return areas

def exact_pdf(t, tres, roots, areas, eigvals, gamma00, gamma10, gamma11):
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
        Coeficients for the exact open/shut time pdf.

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

def exact_mean_time(tres, QAA, QFF, QAF, kA, kF, GAF, GFA):
    """
    Calculate exact mean open or shut time from HJC probability density
    function.

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    QAA : array_like, shape (kA, kA)
    QFF : array_like, shape (kF, kF)
    QAF : array_like, shape (kA, kF)
        QAA, QFF, QAF - submatrices of Q.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.
    GAF : array_like, shape (kA, kB)
    GFA : array_like, shape (kB, kA)
        GAF, GFA- transition probabilities

    Returns
    -------
    mean : float
        Apparent mean open/shut time.
    """

    expQFF = qml.expQt(QFF, tres)
    expQAA = qml.expQt(QAA, tres)
    eGAF = qml.eGs(GAF, GFA, kA, kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, kF, kA, expQAA)

    phiA = qml.phiHJC(eGAF, eGFA, kA)
    QexpQF = np.dot(QAF, expQFF)
    DARS = qml.dARSdS(tres, QAA, QFF,
        GAF, GFA, expQFF, kA, kF)
    uF = np.ones((kF, 1))
    # meanOpenTime = tres + phiA * DARS * QexpQF * uF
    mean = tres + np.dot(phiA, np.dot(np.dot(DARS, QexpQF), uF))[0]

    return mean

def exact_GAMAxx(mec, tres, open):
    """
    Calculate gama coeficients for the exact open time pdf (Eq. 3.22, HJC90).

    Parameters
    ----------
    tres : float
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    open : bool
        True for open time pdf and False for shut time pdf.

    Returns
    -------
    eigen : ndarray, shape (k,)
        Eigenvalues of -Q matrix.
    gama00, gama10, gama11 : ndarrays
        Constants for the exact open/shut time pdf.
    """

    expQFF = qml.expQt(mec.QFF, tres)
    expQAA = qml.expQt(mec.QAA, tres)
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)

    if open:
        phi = qml.phiHJC(eGAF, eGFA, mec.kA)
        eigen, Z00, Z10, Z11 = qml.Zxx(mec.Q, mec.kA,
            mec.QFF, mec.QAF, mec.QFA, expQFF, open)
        u = np.ones((mec.kF,1))
    else:
        phi = qml.phiHJC(eGFA, eGAF, mec.kF)
        eigen, Z00, Z10, Z11 = qml.Zxx(mec.Q, mec.kA,
            mec.QAA, mec.QFA, mec.QAF, expQAA, open)
        u = np.ones((mec.kA, 1))

    gama00 = (np.dot(np.dot(phi, Z00), u)).T[0]
    gama10 = (np.dot(np.dot(phi, Z10), u)).T[0]
    gama11 = (np.dot(np.dot(phi, Z11), u)).T[0]

    return eigen, gama00, gama10, gama11

def likelihood(theta, opts):
    """
    """

    mec = opts['mec']
    conc = opts['conc']
    bursts = opts['data']

    #mec.set_rateconstants(np.exp(theta))
    mec.theta_unsqueeze(np.exp(theta))
    mec.set_eff('c', conc)

    startB = qml.phiA(mec)
    endB = np.ones((mec.kF, 1))

    loglik = 0
    for ind in bursts:
        burst = bursts[ind]
        grouplik = startB
        for i in range(len(burst)):
            t = burst[i] * 0.001
            if i % 2 == 0: # open time
                GAFt = qml.iGt(t, mec.QAA, mec.QAF)
            else: # shut
                GAFt = qml.iGt(t, mec.QFF, mec.QFA)
            grouplik = np.dot(grouplik, GAFt)
            if grouplik.max() > 1e50:
                grouplik = grouplik * 1e-100
                print 'grouplik was scaled down'
        grouplik = np.dot(grouplik, endB)
        loglik += log(grouplik[0])

    #newrates = np.log(mec.unit_rates())
    newrates = np.log(mec.theta())
    return -loglik, newrates

def HJClik(theta, opts):
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
    theta : array_like
        Guesses.
    bursts : dictionary
        A dictionary containing lists of open and shut intervals.
    opts : dictionary
        opts['mec'] : instance of type Mechanism
        opts['tres'] : float
            Time resolution (dead time).
        opts['tcrit'] : float
            Ctritical time interval.
        opts['isCHS'] : bool
            True if CHS vectors should be used (Eq. 5.7, CHS96).

    Returns
    -------
    loglik : float
        Log-likelihood.
    newrates : array_like
        Updated rates/guesses.
    """
    # TODO: Errors.

    mec = opts['mec']
    conc = opts['conc']
    tres = opts['tres']
    tcrit = opts['tcrit']
    is_chsvec = opts['isCHS']
    bursts = opts['data']

    #mec.set_rateconstants(np.exp(theta))
    mec.theta_unsqueeze(np.exp(theta))
    mec.set_eff('c', conc)

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
    Aroots = asymptotic_roots(tres,
        mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)
    AR = qml.AR(Aroots, tres, mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)
    Feigvals, FZ00, FZ10, FZ11 = qml.Zxx(mec.Q, mec.kA, mec.QAA,
        mec.QFA, mec.QAF, expQAA, False)
    Froots = asymptotic_roots(tres,
        mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA)
    FR = qml.AR(Froots, tres, mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA)

    if is_chsvec:
        startB, endB = qml.CHSvec(Froots, tres, tcrit,
            mec.QFA, mec.kA, expQAA, phiF, FR)

    loglik = 0
    for ind in bursts:
        burst = bursts[ind]
        grouplik = startB
        for i in range(len(burst)):
            t = burst[i] * 0.001
            if i % 2 == 0: # open time
                eGAFt = qml.eGAF(t, tres, Aeigvals, AZ00, AZ10, AZ11, Aroots,
                AR, mec.QAF, expQFF)
            else: # shut
                eGAFt = qml.eGAF(t, tres, Feigvals, FZ00, FZ10, FZ11, Froots,
                FR, mec.QFA, expQAA)
            grouplik = np.dot(grouplik, eGAFt)
            if grouplik.max() > 1e50:
                grouplik = grouplik * 1e-100
                #print 'grouplik was scaled down'
        grouplik = np.dot(grouplik, endB)
        loglik += log(grouplik[0])

    #newrates = np.log(mec.unit_rates())
    newrates = np.log(mec.theta())
    return -loglik, newrates

def corr_variance_A(phiA, QAA, kA):
    """
    Calculate variance of open (shut) time according Eq. 2.6 (CH87).
    To calculate variance of shut time function should be called with
    parameters (phiF, QFF, kF).

    Parameters
    ----------
    phiA : array_like, shape (1, kA)
        Initial vector for openings (shuttings).
    QAA : array_like, shape (kA, kA)
        AA submatrix of Q.
    kA : int
        Number of open (shut) states.

    Returns
    -------
    var : float
        Variance.
    """

    uA = np.ones((kA))[:,np.newaxis]
    I = np.eye(kA)
    invQAA = -nplin.inv(QAA)
    M = 2 * I - np.dot(uA, phiA)
    row = np.dot(phiA, invQAA)
    col = np.dot(invQAA, uA)
    var = np.dot(np.dot(row, M), col)[0,0]
    return var

def corr_covariance_A(lag, phiA, QAA, XAA, kA):
    """
    Calculate covariance of open (shut) time according CH87.
    To calculate covariance of shut time function should be called with
    parameters (phiF, QFF, XFF, kF).

    Parameters
    ----------
    lag : int
        Lag.
    phiA : array_like, shape (1, kA)
        Initial vector for openings (shuttings).
    QAA : array_like, shape (kA, kA)
        AA submatrix of Q.
    XAA : array_like, shape (kA, kA)
        Product GAF * GFA.
    kA : int
        Number of open (shut) states.

    Returns
    -------
    covar : float
        Covariance.
    """
    
    Xn = qml.Qpow(XAA, lag)
    uA = np.ones((kA))[:,np.newaxis]
    invQAA = -nplin.inv(QAA)
    M2 = Xn - np.dot(uA, phiA)
    row = np.dot(phiA, invQAA)
    col = np.dot(invQAA, uA)
    covar = np.dot(np.dot(row, M2), col)[0,0]
    return covar

def corr_covariance_AF(lag, phiA, QAA, QFF, XAA, GAF, kA, kF):
    """
    Calculate covariance of open and nth shut times according CH87.
    
    Parameters
    ----------
    lag : int
        Lag.
    phiA : array_like, shape (1, kA)
        Initial vector for openings.
    QAA : array_like, shape (kA, kA)
        AA submatrix of Q.
    QFF : array_like, shape (kF, kF)
        FF submatrix of Q.
    XAA : array_like, shape (kA, kA)
        Product GAF * GFA.
    GAF : array_like, shape (kA, kF)
        GAF matrix.
    kA : int
        Number of open states.
    kF : int
        Number of shut states.

    Returns
    -------
    covar : float
        Covariance.
    """

    Xn = qml.Qpow(XAA, lag-1)
    uA, uF = np.ones((kA))[:,np.newaxis], np.ones((kF))[:,np.newaxis]
    invQAA, invQFF = -nplin.inv(QAA), -nplin.inv(QFF)
    MAF = Xn - np.dot(uA, phiA)
    row = np.dot(phiA, invQAA)
    col = np.dot(np.dot(GAF, invQFF),uF)
    covar = np.dot(np.dot(row, MAF), col)[0,0]
    return covar

def corr_decay_amplitude_A(phiA, QAA, XAA, kA):
    """
    Calculate scalar coefficients for correlation coefficien decay (Eq. 2.11,
    CH83).

    Parameters
    ----------

    Returns
    -------
    w : ndarray, shape (1, k)
    eigs : ndarray, shape (1, k)
    """
    
    varA = corr_variance_A(phiA, QAA, kA)
    eigs, A = qml.eigs(XAA)

    uA = np.ones((kA))[:,np.newaxis]
    invQAA = -nplin.inv(QAA)
    row = np.dot(phiA, invQAA)
    col = np.dot(invQAA, uA)

    ncA = np.rank(XAA) - 1
    w = np.zeros((ncA))
    n = 0
    for i in range(kA):
        if fabs(eigs[i]) > 1e-12 and fabs(eigs[i] - 1) > 1e-12:
            w[n] = np.dot(np.dot(row, A[i, :, :]), col)[0,0] / varA
            n += 1
    return w, eigs

def corr_limit_A(phiA, QAA, AXAA, eigXAA, kA):

    uA = np.ones((kA))[:,np.newaxis]
    invQAA = -nplin.inv(QAA)
    row = np.dot(phiA, invQAA)
    col = np.dot(invQAA, uA)
    M = np.zeros((kA, kA))
    for i in range(kA - 1):
        M += AXAA[i,:,:] * eigXAA[i] / (1 - eigXAA[i])
    cor = np.dot(np.dot(row, M), col)[0,0]
    return cor

def adjacent_open_to_shut_range_mean(u1, u2, QAA, QAF, QFF, QFA, phiA):
    """
    Calculate mean (ideal- no missed events) open times adjacent to a 
    specified shut time range.

    Parameters
    ----------
    u1, u2 : floats
        Shut time range.
    QAA, QAF, QFF, QFA : array_like
        Submatrices of Q.
    phiA : array_like, shape (1, kA)
        Initial vector for openings

    Returns
    -------
    m : float
        Mean open time.
    """
    
    kA = QAA.shape[0]
    uA = np.ones((kA))[:,np.newaxis]
    invQAA, invQFF = -nplin.inv(QAA), nplin.inv(QFF)
    expQFFr = qml.expQt(QFF, u2) - qml.expQt(QFF, u1)
    col = np.dot(np.dot(np.dot(np.dot(QAF, invQFF), expQFFr), QFA), uA)
    row1 = np.dot(phiA, qml.Qpow(invQAA, 2))
    row2 = np.dot(phiA, invQAA)
    m = np.dot(row1, col)[0, 0] / np.dot(row2, col)[0, 0]
    return m

def adjacent_open_to_shut_range_pdf_components(u1, u2, QAA, QAF, QFF, QFA, phiA):
    """
    Calculate time constants and areas for an ideal (no missed events)
    exponential probability density function of open times adjacent to a 
    specified shut time range.

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
    uA = np.ones((kA))[:,np.newaxis]
    invQAA, invQFF = -nplin.inv(QAA), nplin.inv(QFF)
    expQFFr = qml.expQt(QFF, u2) - qml.expQt(QFF, u1)
    col = np.dot(np.dot(np.dot(np.dot(QAF, invQFF), expQFFr), QFA), uA)
    w = np.zeros(kA)
    eigs, A = qml.eigs(-QAA)
    row = np.dot(phiA, invQAA)
    den = np.dot(row, col)[0, 0]
    #TODO: remove 'for'
    for i in range(kA):
        w[i] = np.dot(np.dot(phiA, A[i]), col) / den
    return eigs, w

def printout_occupancies(mec, tres, output=sys.stdout):
    """
    """

    output.write('\n\n\n*******************************************\n')
    output.write('\nOpen\tEquilibrium\tMean life\tMean latency (ms)')
    output.write('\nstate\toccupancy\t(ms)\tto next shutting')
    output.write('\n\t\t\tgiven start in this state')

    pinf = qml.pinf(mec.Q)

    for i in range(mec.k):
        if i == 0:
            mean_life_A = ideal_subset_mean_life_time(mec.Q, 1, mec.kA)
            output.write('\nSubset A ' +
                '\t{0:.5g}'.format(np.sum(pinf[:mec.kA])) +
                '\t{0:.5g}'.format(mean_life_A * 1000) +
                '\n')
        if i == mec.kA:
            mean_life_B = ideal_subset_mean_life_time(mec.Q, mec.kA + 1, mec.kA + mec.kB)
            output.write('\n\nShut\tEquilibrium\tMean life\tMean latency (ms)')
            output.write('\nstate\toccupancy\t(ms)\tto next opening')
            output.write('\n\t\t\tgiven start in this state')
            output.write('\nSubset B ' +
                '\t{0:.5g}'.format(np.sum(pinf[mec.kA:mec.kA+mec.kB])) +
                '\t{0:.5g}'.format(mean_life_B * 1000) +
                '\n')
        if i == mec.kE:
            mean_life_C = ideal_subset_mean_life_time(mec.Q, mec.kA + mec.kB + 1, mec.k)
            output.write('\n\nSubset C ' +
                '\t{0:.5g}'.format(np.sum(pinf[mec.kA+mec.kB:mec.k])) +
                '\t{0:.5g}'.format(mean_life_C * 1000) +
                '\n')
        mean = ideal_mean_latency_given_start_state(mec, i+1)
        output.write('\n{0:d}'.format(i+1) +
            '\t{0:.5g}'.format(pinf[i]) +
            '\t{0:.5g}'.format(-1 / mec.Q[i,i] * 1000) +
            '\t{0:.5g}'.format(mean * 1000))

    expQFF = qml.expQt(mec.QFF, tres)
    expQAA = qml.expQt(mec.QAA, tres)
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)
    phiA = qml.phiHJC(eGAF, eGFA, mec.kA)
    phiF = qml.phiHJC(eGFA, eGAF, mec.kF)

    output.write('\n\n\nInitial vector for HJC openings phiOp =\n')
    for i in range(phiA.shape[0]):
        output.write('\t{0:.5g}'.format(phiA[i]))
    output.write('\n\nInitial vector for ideal openings phiOp =\n')
    phiAi = qml.phiA(mec)
    for i in range(phiA.shape[0]):
        output.write('\t{0:.5g}'.format(phiAi[i]))
    output.write('\n\nInitial vector for HJC shuttings phiSh =\n')
    for i in range(phiF.shape[0]):
        output.write('\t{0:.5g}'.format(phiF[i]))
    output.write('\n\nInitial vector for ideal shuttings phiSh =\n')
    phiFi = qml.phiF(mec)
    for i in range(phiF.shape[0]):
        output.write('\t{0:.5g}'.format(phiFi[i]))


def printout_distributions(mec, tres, output=sys.stdout, eff='c'):
    """

    """

    output.write('\n\n\n*******************************************\n')
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    # OPEN TIME DISTRIBUTIONS
    open = True
    # Ideal pdf
    eigs, w = ideal_dwell_time_pdf_components(mec.QAA,
        qml.phiA(mec))
    output.write('\nIDEAL OPEN TIME DISTRIBUTION')
    pdfs.expPDF_printout(eigs, w, output)

    # Asymptotic pdf
    #roots = asymptotic_roots(mec, tres, open)
    roots = asymptotic_roots(tres,
        mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)
    #areas = asymptotic_areas(mec, tres, roots, open)
    areas = asymptotic_areas(tres, roots,
        mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF, GAF, GFA)
    output.write('\n\nASYMPTOTIC OPEN TIME DISTRIBUTION')
    output.write('\nterm\ttau (ms)\tarea (%)\trate const (1/sec)')
    for i in range(mec.kA):
        output.write('\n{0:d}'.format(i+1) +
        '\t{0:.5g}'.format(-1.0 / roots[i] * 1000) +
        '\t{0:.5g}'.format(areas[i] * 100) +
        '\t{0:.5g}'.format(- roots[i]))
    areast0 = np.zeros(mec.kA)
    for i in range(mec.kA):
        areast0[i] = areas[i] * np.exp(- tres * roots[i])
    areast0 = areast0 / np.sum(areast0)
    output.write('\nAreas for asymptotic pdf renormalised for t=0 to\
    infinity (and sum=1), so areas can be compared with ideal pdf.')
    for i in range(mec.kA):
        output.write('\n{0:d}'.format(i+1) +
        '\t{0:.5g}'.format(areast0[i] * 100))
    mean = exact_mean_time(tres,
            mec.QAA, mec.QFF, mec.QAF, mec.kA, mec.kF, GAF, GFA)
    output.write('\nMean open time (ms) = {0:.5g}'.format(mean * 1000))

    # Exact pdf
    eigvals, gamma00, gamma10, gamma11 = exact_GAMAxx(mec, tres, open)
    output.write('\n\nEXACT OPEN TIME DISTRIBUTION')
    output.write('\neigen\tg00(m)\tg10(m)\tg11(m)')
    for i in range(mec.k):
        output.write('\n{0:.5g}'.format(eigvals[i]) +
        '\t{0:.5g}'.format(gamma00[i]) +
        '\t{0:.5g}'.format(gamma10[i]) +
        '\t{0:.5g}'.format(gamma11[i]))

    output.write('\n\n\n*******************************************\n')
    # SHUT TIME DISTRIBUTIONS
    open = False
    # Ideal pdf
    eigs, w = ideal_dwell_time_pdf_components(mec.QFF, qml.phiF(mec))
    output.write('\nIDEAL SHUT TIME DISTRIBUTION')
    pdfs.expPDF_printout(eigs, w, output)

    # Asymptotic pdf
    #roots = asymptotic_roots(mec, tres, open)
    roots = asymptotic_roots(tres,
        mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA)
    #areas = asymptotic_areas(mec, tres, roots, open)
    areas = asymptotic_areas(tres, roots,
        mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA, GFA, GAF)
    output.write('\n\nASYMPTOTIC SHUT TIME DISTRIBUTION')
    output.write('\nterm\ttau (ms)\tarea (%)\trate const (1/sec)')
    for i in range(mec.kF):
        output.write('\n{0:d}'.format(i+1) +
        '\t{0:.5g}'.format(-1.0 / roots[i] * 1000) +
        '\t{0:.5g}'.format(areas[i] * 100) +
        '\t{0:.5g}'.format(- roots[i]))
    areast0 = np.zeros(mec.kF)
    for i in range(mec.kF):
        areast0[i] = areas[i] * np.exp(- tres * roots[i])
    areast0 = areast0 / np.sum(areast0)
    output.write('\nAreas for asymptotic pdf renormalised for t=0 to\
    infinity (and sum=1), so areas can be compared with ideal pdf.')
    for i in range(mec.kF):
        output.write('\n{0:d}'.format(i+1) +
        '\t{0:.5g}'.format(areast0[i] * 100))
    mean = exact_mean_time(tres,
            mec.QFF, mec.QAA, mec.QFA, mec.kF, mec.kA, GFA, GAF)
    output.write('\nMean shut time (ms) = {0:.6f}'.format(mean * 1000))

    # Exact pdf
    eigvals, gamma00, gamma10, gamma11 = exact_GAMAxx(mec, tres, open)
    output.write('\n\nEXACT SHUT TIME DISTRIBUTION')
    output.write('\neigen\tg00(m)\tg10(m)\tg11(m)')
    for i in range(mec.k):
        output.write('\n{0:.5g}'.format(eigvals[i]) +
        '\t{0:.5g}'.format(gamma00[i]) +
        '\t{0:.5g}'.format(gamma10[i]) +
        '\t{0:.5g}'.format(gamma11[i]))

    # Transition probabilities
    pi = transition_probability(mec.Q)
    output.write('\n\nProbability of transitions regardless of time:')
    for i in range(mec.k):
        output.write('\n[')
        for j in range(mec.k):
            output.write('{0:.4g}\t'.format(pi[i,j]))
        output.write(']')

    # Transition frequency
    f = transition_frequency(mec.Q)
    output.write('\n\nFrequency of transitions (per second):')
    for i in range(mec.k):
        output.write('\n[')
        for j in range(mec.k):
            output.write('{0:.4g}\t'.format(f[i,j]))
        output.write(']')

def transition_probability(Q):
    """
    """
    k = Q.shape[0]
    pi = Q.copy()
    for i in range(k):
        pi[i] = pi[i] / -Q[i,i]
        pi[i,i] = 0
    return pi

def transition_frequency(Q):
    """
    """
    k = Q.shape[0]
    pinf = qml.pinf(Q)
    f = Q.copy().transpose()
    for i in range(k):
        f[i] = f[i] * pinf
        f[i,i] = 0
    return f.transpose()

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

def printout_tcrit(mec, output=sys.stdout):
    """
    Output calculations based on division into bursts by critical time (tcrit).

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    output : output device
        Default device: sys.stdout
    """

    output.write('\n\n\n*******************************************\n')
    output.write('CALCULATIONS BASED ON DIVISION INTO BURSTS BY' +
    ' tcrit- CRITICAL TIME.\n')
    # Ideal shut time pdf
    eigs, w = ideal_dwell_time_pdf_components(mec.QFF, qml.phiF(mec))
    output.write('\nIDEAL SHUT TIME DISTRIBUTION')
    pdfs.expPDF_printout(eigs, w, output)
    taus = 1 / eigs
    areas = w /eigs
    taus, areas = sortShell2(taus, areas)

    comps = taus.shape[0]-1
    tcrits = np.empty((3, comps))
    for i in range(comps):
        output.write('\n\nCritical time between components {0:d} and {1:d}'.
            format(i+1, i+2))
        output.write('\n\nEqual % misclassified (DC criterion)')
        try:
            tcrit = so.bisect(pdfs.expPDF_tcrit_DC,
                taus[i], taus[i+1], args=(taus, areas, i+1))
            enf, ens, pf, ps = pdfs.expPDF_misclassified(tcrit, taus, areas, i+1)
            pdfs.expPDF_misclassified_printout(tcrit, enf, ens, pf, ps, output)
        except:
            output.write('\nBisection with DC criterion failed.')
            tcrit = None
        tcrits[0, i] = tcrit
        
        output.write('\nEqual # misclassified (Clapham & Neher criterion)')
        try:
            tcrit = so.bisect(pdfs.expPDF_tcrit_CN,
                taus[i], taus[i+1], args=(taus, areas, i+1))
            enf, ens, pf, ps = pdfs.expPDF_misclassified(tcrit, taus, areas, i+1)
            pdfs.expPDF_misclassified_printout(tcrit, enf, ens, pf, ps, output)
        except:
            output.write('\nBisection with Clapham & Neher criterion failed.')
            tcrit = None
        tcrits[1, i] = tcrit
        
        output.write('\nMinimum total # misclassified (Jackson et al criterion)')
        try:
            tcrit = so.bisect(pdfs.expPDF_tcrit_Jackson,
                taus[i], taus[i+1], args=(taus, areas, i+1))
            enf, ens, pf, ps = pdfs.expPDF_misclassified(tcrit, taus, areas, i+1)
            pdfs.expPDF_misclassified_printout(tcrit, enf, ens, pf, ps, output)
        except:
            output.write('\nBisection with Jackson et al criterion failed.')
            tcrit = None
        tcrits[2, i] = tcrit
        
    output.write('\n\nSUMMARY of tcrit values:')
    output.write('\nComponents  DC\tC&N\tJackson\n')
    for i in range(comps):
        output.write('{0:d} to {1:d} '.format(i+1, i+2) +
            '\t{0:.5g}'.format(tcrits[0, i] * 1000) +
            '\t{0:.5g}'.format(tcrits[1, i] * 1000) +
            '\t{0:.5g}\n'.format(tcrits[2, i] * 1000))

def printout_correlations(mec, output=sys.stdout, eff='c'):
    """

    """

    output.write('\n\n\n*************************************\n')
    output.write('CORRELATIONS\n')
    
    kA, kB, kF = mec.kA, mec.kB, mec.kF
    output.write('kA, kF = {0:d}, {1:d}\n'.format(kA, kF))    
    GAF, GFA = qml.iGs(mec.Q, kA, kF)
    rGAF, rGFA = np.rank(GAF), np.rank(GFA)
    output.write('Ranks of GAF, GFA = {0:d}, {1:d}\n'.format(rGAF, rGFA))
    XFF = np.dot(GFA, GAF)
    rXFF = np.rank(XFF)
    output.write('Rank of GFA * GAF = {0:d}\n'.format(rXFF))
    ncF = rXFF - 1
    eigXFF, AXFF = qml.eigs(XFF)
    output.write('Eigenvalues of GFA * GAF:\n')
    for i in range(kF):
        output.write('\t{0:.5g}'.format(eigXFF[i]))
    XAA = np.dot(GAF, GFA)
    rXAA = np.rank(XAA)
    output.write('\nRank of GAF * GFA = {0:d}\n'.format(rXAA))
    ncA = rXAA - 1
    eigXAA, AXAA = qml.eigs(XAA)
    output.write('Eigenvalues of GAF * GFA:\n')
    for i in range(kA):
        output.write('\t{0:.5g}'.format(eigXAA[i]))
    phiA, phiF = qml.phiA(mec).reshape((1,kA)), qml.phiF(mec).reshape((1,kF))
    varA = corr_variance_A(phiA, mec.QAA, kA)
    varF = corr_variance_A(phiF, mec.QFF, kF)
    
    #   open - open time correlations
    output.write('\n\n OPEN - OPEN TIME CORRELATIONS')
    output.write('\nVariance of open time = {0:.5g}'.format(varA))
    SDA = sqrt(varA)
    output.write('\nSD of all open times = {0:.5g} ms'.format(SDA * 1000))
    n = 50
    SDA_mean_n = SDA / sqrt(float(n))
    output.write('\nSD of means of {0:d} open times if'.format(n) + 
        'uncorrelated = {0:.5g} ms'.format(SDA_mean_n * 1000))
    covAtot = 0
    for i in range(1, n):
        covA = corr_covariance_A(i+1, phiA, mec.QAA, XAA, kA)
        ro = stl.correlation_coefficient_1(covA, varA, varA)
        covAtot += (n - i) * ro * varA
    vtot = n * varA + 2. * covAtot
    actSDA = sqrt(vtot / (n * n))
    output.write('\nActual SD of mean = {0:.5g} ms'.format(actSDA * 1000))
    pA = 100 * (actSDA - SDA_mean_n) / SDA_mean_n
    output.write('\nPercent difference as result of correlation = {0:.5g}'.
        format(pA))
    v2A = corr_limit_A(phiA, mec.QAA, AXAA, eigXAA, kA)
    pmaxA = 100 * (sqrt(1 + 2 * v2A / varA) - 1)
    output.write('\nLimiting value of percent difference for large n = {0:.5g}'.
        format(pmaxA))
    output.write('\nCorrelation coefficients, r(k), for up to lag k = 5:')
    for i in range(5):
        covA = corr_covariance_A(i+1, phiA, mec.QAA, XAA, kA)
        ro = stl.correlation_coefficient_1(covA, varA, varA)
        output.write('\nr({0:d}) = {1:.5g}'.format(i+1, ro))

    # shut - shut time correlations
    output.write('\n\n SHUT - SHUT TIME CORRELATIONS')
    output.write('\nVariance of shut time = {0:.5g}'.format(varF))
    SDF = sqrt(varF)
    output.write('\nSD of all shut times = {0:.5g} ms'.format(SDF * 1000))
    n = 50
    SDF_mean_n = SDF / sqrt(float(n))
    output.write('\nSD of means of {0:d} shut times if'.format(n) +
        'uncorrelated = {0:.5g} ms'.format(SDF_mean_n * 1000))
    covFtot = 0
    for i in range(1, n):
        covF = corr_covariance_A(i+1, phiF, mec.QFF, XFF, kF)
        ro = stl.correlation_coefficient_1(covF, varF, varF)
        covFtot += (n - i) * ro * varF
    vtotF = 50 * varF + 2. * covFtot
    actSDF = sqrt(vtotF / (50. * 50.))
    output.write('\nActual SD of mean = {0:.5g} ms'.format(actSDF * 1000))
    pF = 100 * (actSDF - SDF_mean_n) / SDF_mean_n
    output.write('\nPercent difference as result of correlation = {0:.5g}'.
        format(pF))
    v2F = corr_limit_A(phiF, mec.QFF, AXFF, eigXFF, kF)
    pmaxF = 100 * (sqrt(1 + 2 * v2F / varF) - 1)
    output.write('\nLimiting value of percent difference for large n = {0:.5g}'.
        format(pmaxF))
    output.write('\nCorrelation coefficients, r(k), for up to k = 5 lags:')
    for i in range(5):
        covF = corr_covariance_A(i+1, phiF, mec.QFF, XFF, kF)
        ro = stl.correlation_coefficient_1(covF, varF, varF)
        output.write('\nr({0:d}) = {1:.5g}'.format(i+1, ro))

    # open - shut time correlations 
    output.write('\n\n OPEN - SHUT TIME CORRELATIONS')
    output.write('\nCorrelation coefficients, r(k), for up to k= 5 lags:')
    for i in range(5):
        covAF = corr_covariance_AF(i+1, phiA, mec.QAA, mec.QFF, XAA, GAF, kA, kF)
        ro = stl.correlation_coefficient_1(covAF, varA, varF)
        output.write('\nr({0:d}) = {1:.5g}'.format(i+1, ro))
        
def printout_adjacent(mec, t1, t2, output=sys.stdout):
    """

    """

    output.write('\n\n\n*************************************\n')
    output.write(' OPEN TIMES ADJACENT TO SPECIFIED SHUT TIME RANGE')
    
    kA = mec.kA
    phiA = qml.phiA(mec).reshape((1,kA))
    
    output.write('\nPDF of open times that precede shut times between {0:.3f}\
 and {1:.3f} ms'.format(t1 * 1000, t2 * 1000))
        
    eigs, w = adjacent_open_to_shut_range_pdf_components(t1, t2, 
        mec.QAA, mec.QAF, mec.QFF, mec.QFA, phiA)
    pdfs.expPDF_printout(eigs, w, output)

    mean = adjacent_open_to_shut_range_mean(t1, t2, 
        mec.QAA, mec.QAF, mec.QFF, mec.QFA, phiA)
    output.write('\nMean from direct calculation (ms) = {0:.6f}'.format(mean * 1000))