"""A collection of functions for Q matrix manipulations.

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
The principles of the stochastic interpretation of ion channel mechanisms.
In: Single-channel recording. 2nd ed. (Eds: Sakmann B, Neher E)
Plenum Press, New York, pp. 397-482.

CH95b: Colquhoun D, Hawkes AG (1995b)
A Q-Matrix Cookbook.
In: Single-channel recording. 2nd ed. (Eds: Sakmann B, Neher E)
Plenum Press, New York, pp. 589-633.

CHS96: Colquhoun D, Hawkes AG, Srodzinski K (1996)
Joint distributions of apparent open and shut times of single-ion channels
and maximum likelihood fitting of mechanisms.
Phil Trans R Soc Lond A 354, 2555-2590.
"""

__author__="R.Lape, University College London"
__date__ ="$11-Oct-2010 10:33:07$"

import numpy as np
from numpy import linalg as nplin

import dcpypsrc

def eigs(Q):
    """
    Calculate eigenvalues and spectral matrices of a matrix Q.

    Parameters
    ----------
    Q : array_like, shape (k, k)

    Returns
    -------
    eigvals : ndarray, shape (k,)
        Eigenvalues of M.
    A : ndarray, shape (k, k, k)
        Spectral matrices of Q.
    """

    eigvals, M = nplin.eig(Q)
    N = nplin.inv(M)
    k = N.shape[0]
    A = np.zeros((k, k, k))
    # TODO: make this a one-liner avoiding loops
    for i in range(k):
        X = np.empty((k, 1))
        Y = np.empty((1, k))
        X[:, 0] = M[:, i]
        Y[0] = N[i]
        A[i] = np.dot(X, Y)
    return eigvals, A

def iGs(Q, kA, kB):
    r"""
    Calculate GBA and GAB matrices (Eq. 1.25, CH82).
    Calculate also GFA and GAF if kF is given instead of kB.

    .. math::

       \bs{G}_\cl{BA} &= -\bs{Q}_\cl{BB}^{-1} \bs{Q}_\cl{BA} \\
       \bs{G}_\cl{AB} &= -\bs{Q}_\cl{AA}^{-1} \bs{Q}_\cl{AB}

    Parameters
    ----------
    Q : array_like, shape (k, k)
    kA : int
        A number of open states in kinetic scheme.
    kB : int
        A number of short lived shut states in kinetic scheme.

    Returns
    -------
    GAB : ndarray, shape (kA, kB)
    GBA : ndarray, shape (kB, kA)
    """

    kE = kA + kB
    QBB = Q[kA:kE, kA:kE]
    QBA = Q[kA:kE, 0:kA]
    QAA = Q[0:kA, 0:kA]
    QAB = Q[0:kA, kA:kE]
    GAB = np.dot(nplin.inv(-1 * QAA), QAB)
    GBA = np.dot(nplin.inv(-1 * QBB), QBA)
    return GAB, GBA

def eGs(G12, G21, k1, k2, expQ22):
    """
    Calculate eGAF (or eGFA) for calculation of initial vectors (HJC92).
        (I - GAF * (I - expQFF) * GFA)^-1 * GAF * expQFF

    Parameters
    ----------
    G12 : array_like, shape (k1, k2)
    G21 : array_like, shape (k2, k1)
    k1 : int
        A number of open/shut states in kinetic scheme.
    k2 : int
        A number of shut/open states in kinetic scheme.

    Returns
    -------
    eG12 : array_like, shape (k1, k2)
    """

    temp = np.eye(k1) - np.dot(np.dot(G12, np.eye(k2) - expQ22), G21)
    eG12 = np.dot(np.dot(nplin.inv(temp), G12), expQ22)
    return eG12

def expQ(M, t):
    """
    Calculate exponential of a matrix M.
        expM = exp(M * t)

    Parameters
    ----------
    M : array_like, shape (k, k)
    t : float
        Time.

    Returns
    -------
    expM : ndarray, shape (k, k)
    """

    eigvals, A = eigs(-M)
    k = M.shape[0]
    expM = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            for m in range(k):
                expM[i, j] += A[m, i, j] * np.exp(-eigvals[m] * t)
    return expM

def phiHJC(eG12, eG21, k1, k2):
    """
    Calculate equilibrium probability vector phi by solving
        phi*(I-eG12*eG21)=0 (Eq. 10, HJC92)

    Parameters
    ----------
    eG12 : array_like, shape (k1, k2)
    eG21 : array_like, shape (k2, k1)
    k1 : int
        A number of open/shut states in kinetic scheme.
    k2 : int
        A number of shut/open states in kinetic scheme.

    Returns
    -------
    phi : array_like, shape (k1)
    """

    if k1 == 1:
        phi = 1
        return phi

    Qsub = np.eye(k1) - np.dot(eG12, eG21)
    u = np.ones((k1,1))
    S = np.concatenate((Qsub, u), 1)
    phi = np.dot(u.transpose(), nplin.inv(np.dot(S, S.transpose())))

    return phi

def dARSdS(tres, Q11, Q12, Q22, Q21, G12, G21, expQ22, expQ11, k1, k2):
    r"""
    Evaluate the derivative with respect to s of the Laplace transform of the
    survival function (Eq. 3.6, CHS96)

    .. math::

       \left[ -\frac{\text{d}}{\text{d}s} {^\cl{A}\!\bs{R}^*(s)} \right]_{s=0}

    SFF = I - exp(QFF * tres)
    First evaluate [dVA(s) / ds] * s = 0.
    dVAds = -inv(QAA) * GAF * SFF * GFA - GAF * SFF * inv(QFF) * GFA +
    + tres * GAF * expQFF * GFA

    Then: DARS = inv(VA) * QAA^(-2) - inv(VA) * dVAds * inv(VA) * inv(QAA) =
    = inv(VA) * [inv(QAA) - dVAds * inv(VA)] * inv(QAA)
    where VA = I - GAF * SFF * GFA

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    Q11 : array_like, shape (k1, k1)
    Q12 : array_like, shape (k1, k2)
    Q22 : array_like, shape (k2, k2)
    Q21 : array_like, shape (k1, k1)
        Q11, Q12, Q22, Q21 - submatrices of Q.
    G12 : array_like, shape (k1, k2)
    G21 : array_like, shape (k2, k1)
        G12, G21 - G matrices.
    expQ22 : array_like, shape(k2, k2)
    expQ11 : array_like, shape(k1, k1)
        expQ22, expQ11 - exponentials of submatrices Q22 and Q11.
    k1 : int
        A number of open/shut states in kinetic scheme.
    k2 : int
        A number of shut/open states in kinetic scheme.

    Returns
    -------
    DARS : array_like, shape (k1, k1)
    """

    invQ11 = nplin.inv(Q11)
    invQ22 = nplin.inv(Q22)

    #SFF = I - EXPQF
    I = np.eye(k2)
    SFF = I - expQ22

    #Q1 = tres * GAF * exp(QFF*tres) * GFA
    Q1 = tres * np.dot(G12, np.dot(expQ22, G21))
    #Q2 = GAF * SFF * inv(QFF) * GFA
    Q2 = np.dot(G12, np.dot(SFF, np.dot(invQ22, G21)))
    #Q3 = -inv(QAA) * GAF * SFF * GFA
    Q3 = np.dot(np.dot(np.dot(-invQ11, G12), SFF), G21)
    Q1 = Q1 - Q2 + Q3

    # VA = I - GAF * SFF * GFA
    I = np.eye(k1)
    VA = I - np.dot(np.dot(G12, SFF), G21)

    # DARS = inv(VA) * (QAA**-2) - inv(VA) * Q1 * inv(VA) * inv(QAA) =
    #      = inv(VA) * [inv(QAA) - Q1 * inv(VA)] * inv(QAA)

    Q3 = invQ11 + - np.dot(Q1, nplin.inv(VA))
    DARS = np.dot(np.dot(nplin.inv(VA), Q3), invQ11)

    return DARS

def hjc_mean_time(mec, tres):
    """
    Calculate exact mean open or shut time from HJC probability density
    function. 

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    tres : float
        Time resolution (dead time).

    Returns
    -------
    hmopen : float
        Apparent mean open time.
    hmshut : float
        Apparent mean shut time.
    """

    GAF, GFA = iGs(mec.Q, mec.kA, mec.kF)
    expQFF = expQ(mec.QFF, tres)
    expQAA = expQ(mec.QAA, tres)

    #Calculate Gs and initial vectors corrected for missed events.
    eGAF = eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = eGs(GFA, GAF, mec.kF, mec.kA, expQAA)
    phiA = phiHJC(eGAF, eGFA, mec.kA, mec.kF)
    phiF = phiHJC(eGFA, eGAF, mec.kF, mec.kA)

    #Recalculate QexpQA and QexpQF
    QexpQF = np.dot(mec.QAF, expQFF)
    QexpQA = np.dot(mec.QFA, expQAA)

    DARS = dARSdS(tres, mec.QAA, mec.QAF, mec.QFF, mec.QFA,
        GAF, GFA, expQFF, expQAA, mec.kA, mec.kF)
    DFRS = dARSdS(tres, mec.QFF, mec.QFA, mec.QAA, mec.QAF,
        GFA, GAF, expQAA, expQFF, mec.kF, mec.kA)

    uA = np.ones((mec.kA, 1))
    uF = np.ones((mec.kF, 1))
    # meanOpenTime = tres + phiA * DARS * QexpQF * uF
    # meanShutTime = tres + phiF * DFRS * QexpQA * uA
    hmopen = tres + np.dot(phiA, np.dot(np.dot(DARS, QexpQF), uF))
    hmshut = tres + np.dot(phiF, np.dot(np.dot(DFRS, QexpQA), uA))

    return hmopen, hmshut

def popen(mec, tres=0):
    """
    Calculate open probability for any temporal resolution, tres.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    Popen : float
        Open probability.
    """

    if tres == 0:
        p = pinf(mec)
        Popen = 0
        for i in range(mec.kA):
            Popen = Popen + p[i]
        return Popen
    else:
        hmopen, hmshut = hjc_mean_time(mec, tres)
        Popen = hmopen / (hmopen + hmshut)
        return Popen[0,0]

def pinf(mec):
    """
    Calculate ecquilibrium occupancies by adding a column of ones
    to Q matrix.
    Pinf = uT * invert((S * transpos(S))).

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    pinf : ndarray, shape (k1)
    """

    u = np.ones((mec.Q.shape[0],1))
    S = np.concatenate((mec.Q, u), 1)
    pinf = np.dot(u.transpose(), nplin.inv((np.dot(S,S.transpose()))))[0]
    return pinf

def phiO(mec):
    """
    Calculate initial vector for openings.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    phi : ndarray, shape (kA)
    """

    uA = np.ones((mec.kA,1))
    pF = pinf(mec)[mec.kA:]
    nom = np.dot(pF, mec.QFA)
    denom = np.dot(nom,uA)
    phi = nom / denom
    return phi

def phiS(mec):
    """
    Calculate inital vector for shuttings.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    phi : ndarray, shape (kF)
    """

    GAF, GFA = iGs(mec.Q, mec.kA, mec.kF)
    phi = np.dot(phiO(mec), GAF)
    return phi

def phiBurst(mec):
    """
    Calculate the start probabilities of a burst (Eq. 3.2, CH82).
    PhiB = (pCinf * (QCB * GBA + QCA)) / (pCinf * (QCB * GBA + QCA) * uA)

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    phiB : array_like, shape (1, kA)
    """

    uA = np.ones((mec.kA, 1))
    pC = pinf(mec)[mec.kE:]
    nom = np.dot(pC, (np.dot(mec.QCB, mec.GBA) + mec.QCA))
    denom = np.dot(nom, uA)
    phiB = nom / denom
    return phiB

def endBurst(mec):
    r"""
    Calculate the end vector for a burst (Eq. 3.4, CH82).

    .. math::

       \bs{e}_\text{b} = (\bs{I}-\bs{G}_\cl{AB} \bs{G}_\cl{BA}) \bs{u}_\cl{A}

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    eB : array_like, shape (kA, 1)
    """

    uA = np.ones((mec.kA, 1))
    I = np.eye(mec.kA)
    eB = np.dot((I - np.dot(mec.GAB, mec.GBA)), uA)
    return eB

def mean_burst_length(mec):
    """
    Calculate the mean burst length (Eq. 3.19, CH82).
    m = PhiB * (I - GAB * GBA)^(-1) * (-QAA^(-1)) * \
        (I - QAB * (QBB^(-1)) * GBA) * uA

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    m : float
        The mean burst length.
    """

    uA = np.ones((mec.kA, 1))
    I = np.eye(mec.kA)
    invQAA = -1 * nplin.inv(mec.QAA)
    invQBB = nplin.inv(mec.QBB)
    interm1 = nplin.inv(I - np.dot(mec.GAB, mec.GBA))
    interm2 = I - np.dot(np.dot(mec.QAB, invQBB), mec.GBA)
    m = (np.dot(np.dot(np.dot(np.dot(phiBurst(mec), interm1), invQAA),
        interm2), uA)[0])
    return m

def mean_num_burst_openings(mec):
    """
    Calculate the mean number of openings per burst (Eq. 3.7, CH82).
    mu = phiB * (I - GAB * GBA)^(-1) * uA

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    mu : float
        The mean number ofopenings per burst.
    """

    uA = np.ones((mec.kA,1))
    I = np.eye(mec.kA)
    interm = nplin.inv(I - np.dot(mec.GAB, mec.GBA))
    mu = np.dot(np.dot(phiBurst(mec), interm), uA)[0]
    return mu

def distr_num_burst_openings(mec, r):
    """
    The distribution of openings per burst (Eq. 3.5, CH82).
    P(r) = phiB * (GAB * GBA)^(r-1) * eB

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    r : int
        Number of openings per burst.

    Returns
    -------
    Pr : float
        Probability of seeing r openings per burst.
    """

    GG = np.dot(mec.GAB, mec.GBA)
    if r == 1:
        interm = np.eye(mec.kA)
    elif r == 2:
        interm = GG
    else:
        interm = GG
        for i in range(2, r):
            interm = np.dot(interm, GG)
    Pr = np.dot(np.dot(phiBurst(mec), interm), endBurst(mec))
    return Pr

def pdf_burst_length(mec, t):
    """
    Probability density function of the burst length (Eq. 3.17, CH82).
    f(t) = phiB * [PEE(t)]AA * (-QAA) * eB, where PEE(t) = exp(QEE * t)

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    f : float
    """

    expQEEA = expQ(mec.QEE, t)[:mec.kA, :mec.kA]
    f = np.dot(np.dot(np.dot(phiBurst(mec), expQEEA), -mec.QAA), endBurst(mec))
    return f

def pdf_open_time(mec, t):
    """
    Probability density function of the open time.
    f(t) = phiOp * exp(-QAA * t) * (-QAA) * uA

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    f : float
    """

    uA = np.ones((mec.kA, 1))
    expQAA = expQ(mec.QAA, t)
    f = np.dot(np.dot(np.dot(phiO(mec), expQAA), -mec.QAA), uA)
    return f

def pdf_shut_time(mec, t):
    """
    Probability density function of the shut time.
    f(t) = phiShut * exp(QFF * t) * (-QFF) * uF

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    t : float
        Time.

    Returns
    -------
    f : float
    """

    uF = np.ones((mec.kF, 1))
    expQFF = expQ(mec.QFF, t)
    f = np.dot(np.dot(np.dot(phiS(mec), expQFF), -mec.QFF), uF)
    return f

def dW(s, tres, Q12, Q22, Q21, k1, k2):
    """
    Evaluate the derivative with respect to s of the matrix W(s) at the root s
    (Eq. 56, HJC92).
    W'(s) = I + QAF * [SFF(s) * (s*I - QFF)^(-1) - tau * (I - SFF(s))] * eGFA(s)
    where SFF(s) = I - exp(-(s*I - QFF) * tau) (Eq. 17, HJC92)
    and eGFA(s) = (s*I - QFF)^(-1) * QFA (Eq. 4, HJC92).

    Parameters
    ----------
    s : float
        Laplace transform argument.
    tres : float
        Time resolution (dead time).
    Q12 : array_like, shape (k1, k2)
    Q22 : array_like, shape (k2, k2)
    Q21 : array_like, shape (k2, k1)
        Q12, Q22, Q21 - submatrices of Q.
    k1 : int
        A number of open/shut states in kinetic scheme.
    k2 : int
        A number of shut/open states in kinetic scheme.

    Returns
    -------
    dW : ndarray, shape (k2, k2)
    """

    I2 = np.eye(k2)
    I1 = np.eye(k1)
    IQ22 = s * I2 - Q22
    expIQ22 = expQ(tres, -IQ22)
    S22 = I2 - expIQ22
    eG21s = np.dot(nplin.inv(s * I2 - Q22), Q21)
    w1 = np.dot(S22, nplin.inv(s * I2 - Q22)) - tres * (I2 - S22)
    dW = I1 + np.dot(np.dot(Q12, w1), eG21s)
    return dW

def H(s, tres, Q11, Q22, Q21, Q12, k1, k2):
    """
    Evaluate H(s) funtion (Eq. 54, HJC92).
    H(s) = QAA + QAF * (s*I - QFF) ^(-1) * (I - exp(-(s*I - QFF) * tau)) * QFA

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
    H : ndarray, shape (k2, k2)
    """

    I = np.eye(k1)
    X11 = s * I - Q11
    invX11 = nplin.inv(X11)
    expX11 = expQ(tres, -X11)
    H = Q22 + np.dot(np.dot(np.dot(Q21, invX11), I - expX11), Q12)
    return H

def W(s, tres, Q11, Q22, Q21, Q12, k1, k2):
    """
    Evaluate W(s) function (Eq. 52, HJC92).
    W(s) = s * I - H(s)

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
    W : ndarray, shape (k2, k2)
    """

    I = np.eye(k2)
    W = s * I - H(s, tres, Q11, Q22, Q21, Q12, k1, k2)
    return W


def gFB(s, tres, Q11, Q22, Q21, Q12, k1, k2):
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

    h = H(s, tres, Q11, Q22, Q21, Q12, k1, k2)
    eigval, A = eigs(h)
    ng = 0
    for i in range(k2):
        if eigval[i] <= s: ng += 1
    if qmatrc.debug:
        print ('number of eigenvalues that are <= s (=', s, ') =', ng)
    return ng

def bisection_intervals(sa, sb, tres, Q11, Q22, Q21, Q12, k1, k2):
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

    nga = gFB(sa, tres, Q11, Q22, Q21, Q12, k1, k2)
    if nga > 0:
        sa = sa * 4
    ngb = gFB(sb, tres, Q11, Q22, Q21, Q12, k1, k2)
    if ngb < k2:
        sb = sb / 4

    sr = np.zeros((k2,2))
    sv = np.zeros((100,4))
    sv[0,0] = sa
    sv[0,1] = sb
    sv[0,2] = nga
    sv[0,3] = ngb
    ntodo = 0
    ndone = 0
    nsplit = 0

    while (ndone < k2) and (nsplit < 1000):
        sa = sv[ntodo, 0]
        sb = sv[ntodo, 1]
        nga = sv[ntodo, 2]
        ngb = sv[ntodo, 3]
        sa1, sb1, sa2, sb2, nga1, ngb1, nga2, ngb2 = split(sa, sb,
            nga, ngb, tres, Q11, Q22, Q21, Q12, k1, k2)
        nsplit = nsplit + 1
        ntodo = ntodo - 1

        # Check if either or both of the two subintervals output from
        # SPLIT contain only one root?
        if (ngb1 - nga1) == 1:
            sr[ndone, 0] = sa1
            sr[ndone, 1] = sb1
            ndone = ndone + 1
        else:
            ntodo = ntodo + 1
            sv[ntodo, 0] = sa1
            sv[ntodo, 1] = sb1
            sv[ntodo, 2] = nga1
            sv[ntodo, 3] = ngb1
        if (ngb2 - nga2) == 1:
            sr[ndone, 0] = sa2
            sr[ndone, 1] = sb2
            ndone = ndone + 1
        else:
            ntodo = ntodo + 1
            sv[ntodo, 0] = sa2
            sv[ntodo, 1] = sb2
            sv[ntodo, 2] = nga2
            sv[ntodo, 3] = ngb2

    if ndone < k2:
        print ('Only', ndone, 'roots out of', k2, 'were located')
    return sr

def split(sa, sb, nga, ngb, tres, Q11, Q22, Q21, Q12, k1, k2):
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
    sa1, sb1, sa2, sb2 : floats
        Limits of s value intervals.
    nga1, ngb1, nga2, ngb2 : ints
        Number of eigenvalues below corresponding s values.
    """

    ntrymax = 10000
    ntry = 0
    #nerrs = False
    end = False

    while not end:
        sc = (sa + sb) / 2.0
        ngc = gFB(sc, tres, Q11, Q22, Q21, Q12, k1, k2)
        if ngc == nga: sa = sc
        elif ngc == ngb: sb = sc
        else:
            end = True
            sa1 = sa
            sb1 = sc
            sa2 = sc
            sb2 = sb
            nga1 = nga
            ngb1 = ngc
            nga2 = ngc
            ngb2 = ngb
        ntry = ntry + 1
        if ntry > ntrymax:
            print ('ERROR: unable to split interval in BALL_ROOT')
            end = True

    return sa1, sb1, sa2, sb2, nga1, ngb1, nga2, ngb2

def bisect(s1, s2, tres, Q11, Q22, Q21, Q12, k1, k2):
    """
    Find asymptotic root (det(W) = 0) in interval [s1, s2] using bisection
    method.

    Parameters
    ----------
    s1, s2 : float
        Limits of Laplace transform argument interval to split.
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
    sout : float
        Asymptotic root at wich |W|=0.
    """

    f = nplin.det(W(s1, tres, Q11, Q22, Q21, Q12, k1, k2))
    if f > 0:
        temp = s1
        s1 = s2
        s2 = temp
    iter = 0
    solved = False
    itermax = 1000
    sout = None

    while iter < itermax and not solved:
        iter += 1
        sout = 0.5 * (s1 + s2)
        f = nplin.det(W(sout, tres, Q11, Q22, Q21, Q12, k1, k2))
        if f < 0:
            s1 = sout
        elif f > 0:
            s2 = sout
        else:    #if f == 0:
            solved = True

    #if verbose: print 'function solved in', ns, 'itterations'
    return sout

def asymptotic_roots(tres, mec, open):
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
        sro = bisection_intervals(sao, sbo, tres,
            mec.QFF, mec.QAA, mec.QAF, mec.QFA, mec.kF, mec.kA)
        roots = np.zeros(mec.kA)
        for i in range(mec.kA):
            roots[i] = bisect(sro[i,0], sro[i,1], tres,
                mec.QFF, mec.QAA, mec.QAF, mec.QFA, mec.kF, mec.kA)
        return roots
    else:
        sro = bisection_intervals(sas, sbs, tres,
            mec.QAA, mec.QFF, mec.QFA, mec.QAF, mec.kA, mec.kF)
        roots = np.zeros(mec.kF)
        for i in range(mec.kF):
            roots[i] = bisect(sro[i,0], sro[i,1], tres,
                mec.QAA, mec.QFF, mec.QFA, mec.QAF, mec.kA, mec.kF)
        return roots

def asymptotic_areas(tres, roots, mec, open):
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
        G12, G21 = iGs(mec.Q, mec.kA, mec.kF)
    else:
        k1 = mec.kF
        k2 = mec.kA
        Q11 = mec.QFF
        Q22 = mec.QAA
        Q21 = mec.QAF
        Q12 = mec.QFA
        G21, G12 = iGs(mec.Q, mec.kA, mec.kF)

    expQ22 = expQsub(tres, Q22)
    expQ11 = expQsub(tres, Q11)
    eG12 = eGs(G12, G21, k1, k2, expQ22)
    eG21 = eGs(G21, G12, k2, k1, expQ11)
    phi1 = phiHJC(eG12, eG21, k1, k2)[0]

    areas = np.zeros(k1)
    rowA = np.zeros((k1,k1))
    colA = np.zeros((k1,k1))
    for i in range(k1):
        WA = W(roots[i], tres, Q22, Q11, Q12, Q21, k2, k1)
        rowA[i] = pinf(WA)
        AW = np.transpose(WA)
        colA[i] = pinf(AW)

    for i in range(k1):
        u2 = np.ones((k2,1))
        nom = np.dot(np.dot(np.dot(np.dot(np.dot(phi1, colA[i]), rowA[i]),
            Q12), expQ22), u2)
        W1A = dW(roots[i], tres, Q12, Q22, Q21, k1, k2)
        denom = -roots[i] * np.dot(np.dot(rowA[i], W1A), colA[i])
        areas[i] = nom / denom

    return areas

def exact_HJC(tres, mec, open):
    """
    Calculate gama constants for the exact open/shut time pdf (Eq. 3.22, HJC90).

    Parameters
    ----------
    tres : float
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    open : Bool
        True for open time pdf and False for shut time pdf.

    Returns
    -------
    eigen : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    gama00, gama10, gama11 : lists of floats
        Constants for the exact open/shut time pdf.
    """

    k = mec.Q.shape[0]
    expQFF = expQsub(tres, mec.QFF)
    expQAA = expQsub(tres, mec.QAA)
    uF = np.ones((mec.kF,1))
    uA = np.ones((mec.kA, 1))
    GAF, GFA = iGs(mec.Q, mec.kA, mec.kF)
    eGAF = eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = eGs(GFA, GAF, mec.kF, mec.kA, expQAA)
    phiA = phiHJC(eGAF, eGFA, mec.kA, mec.kF)
    phiF = phiHJC(eGFA, eGAF, mec.kF, mec.kA)

    eigen, A = eigs(-mec.Q)
    # Maybe needs check for equal eigenvalues.

    # Calculate Dj (Eq. 3.16, HJC90) and Cimr (Eq. 3.18, HJC90).
    D = []
    C00 = []
    C11 = []
    C10 = []

    for i in range(k):
        if open:
            D.append(np.dot(np.dot(A[i, :mec.kA, mec.kA:], expQFF), mec.QFA))
            C00.append(A[i, :mec.kA, :mec.kA])
            C11.append(np.dot(D[i], C00[i]))
        else:
            D.append(np.dot(np.dot(A[i, mec.kA:, :mec.kA], expQAA), mec.QAF))
            C00.append(A[i, mec.kA:, mec.kA:])
            C11.append(np.dot(D[i], C00[i]))
    if open:
        for i in range(k):
            S = np.zeros((mec.kA, mec.kA))
            for j in range(k):
                if j != i:
                    S = S + ((np.dot(D[i], C00[j]) + np.dot(D[j], C00[i])) /
                        (eigen[j] - eigen[i]))
            C10.append(S)
    else:
        for i in range(k):
            S = np.zeros((mec.kF, mec.kF))
            for j in range(k):
                if j != i:
                    S = S + ((np.dot(D[i], C00[j]) + np.dot(D[j], C00[i])) /
                        (eigen[j] - eigen[i]))
            C10.append(S)

    gama00 = []
    gama10 = []
    gama11 = []
    if open:
        M1 = np.dot(np.dot(mec.QAF, expQFF), uF)
        for i in range(k):
            gama00.append(np.dot(np.dot(phiA, C00[i]), M1)[0][0])
            gama10.append(np.dot(np.dot(phiA, C10[i]), M1)[0][0])
            gama11.append(np.dot(np.dot(phiA, C11[i]), M1)[0][0])
    else:
        M1 = np.dot(np.dot(mec.QFA, expQAA), uA)
        for i in range(k):
            gama00.append(np.dot(np.dot(phiF, C00[i]), M1)[0][0])
            gama10.append(np.dot(np.dot(phiF, C10[i]), M1)[0][0])
            gama11.append(np.dot(np.dot(phiF, C11[i]), M1)[0][0])

    return eigen, gama00, gama10, gama11
