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
    Calculate eigenvalues, eigenvectors and spectral matrices
    of a matrix Q.
    Return eigenvalues and spectral matrices.

    Parameters
    ----------
    Q : array_like, shape (k, k)

    Returns
    -------
    eigvals : ndarray, shape (k,)
        Eigenvalues of M.
    A : ndarray of shape (k, k, k); length `k`
        Spectral matrices of Q.
    """

    eigvals, M = nplin.eig(Q)
    if dcpypsrc.debug: print 'eigenvalues=', eigvals
    if dcpypsrc.debug: print 'eigenvectors=', M
    N = nplin.inv(M)
    k = N.shape[0]

    A = np.zeros((k, k, k))
    # TODO: make this a one-liner avoiding loops
    for i in range(k):
        for j in range(k):
            for m in range(k):
                A[i, j, m] = M[j, i] * N[i, m]

    if dcpypsrc.debug: print 'spectral matrices =', A

    return eigvals, A

def iGs(Q, kA, kB):
    r"""
    Calculate GBA and GAB matrices (Eq. 1.25, CH82).

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
    GAB : array_like, shape (kA, kB)
    GBA : array_like, shape (kB, kA)
    """

    kE = kA + kB
    QBB = Q[kA:kE, kA:kE]
    QBA = Q[kA:kE, 0:kA]
    QAA = Q[0:kA, 0:kA]
    QAB = Q[0:kA, kA:kE]

    GAB = np.dot(nplin.inv(-1 * QAA), QAB)
    GBA = np.dot(nplin.inv(-1 * QBB), QBA)
    if dcpypsrc.debug: print 'GAB= ', GAB
    if dcpypsrc.debug: print 'GBA= ', GBA

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

def expQsub(M, t):
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
    expM : array_like, shape (k, k)
    """

    eigvals, A = eigs(-M)
    k = M.shape[0]

    expM = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            for m in range(k):
                temp = A[m][i, j] * np.exp(-eigvals[m] * t)
                expM[i, j] = expM[i, j] + temp

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

    k = mec.Q.shape[0]
    kF = k - mec.kA
    GAF, GFA = iGs(mec.Q, mec.kA, kF)
    QFF = mec.Q[mec.kA:k, mec.kA:k]
    expQFF = expQsub(QFF, tres)
    QAA = mec.Q[0:mec.kA, 0:mec.kA]
    expQAA = expQsub(QAA, tres)

    #Calculate Gs and initial vectors corrected for missed events.
    eGAF = eGs(GAF, GFA, mec.kA, kF, expQFF)
    eGFA = eGs(GFA, GAF, kF, mec.kA, expQAA)
    phiA = phiHJC(eGAF, eGFA, mec.kA, kF)
    phiF = phiHJC(eGFA, eGAF, kF, mec.kA)

    #Recalculate QexpQA and QexpQF
    QAF = mec.Q[0:mec.kA, mec.kA:k]
    QexpQF = np.dot(QAF, expQFF)
    QFA = mec.Q[mec.kA:k, 0:mec.kA]
    QexpQA = np.dot(QFA, expQAA)

    DARS = dARSdS(tres, QAA, QAF, QFF, QFA, GAF, GFA, expQFF, expQAA,
        mec.kA, kF)
    DFRS = dARSdS(tres, QFF, QFA, QAA, QAF, GFA, GAF, expQAA, expQFF,
        kF, mec.kA)

    uA = np.ones((mec.kA, 1))
    uF = np.ones((kF, 1))
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
    phi : array_like, shape (k1)
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
    phi : array_like, shape (kA)
    """

    uA = np.ones((mec.kA,1))
    p = pinf(mec)
    pF = p[mec.kA:]
    QFA = mec.Q[mec.kA:, :mec.kA]
    nom = np.dot(pF, QFA)
    denom = np.dot(nom,uA)
    phi = nom / denom
    if dcpypsrc.debug: print 'phiO=', phi
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
    phi : array_like, shape (kB+kC)
    """

    kF = mec.kB + mec.kC
    phiOp = phiO(mec)
    GAF, GFA = iGs(mec.Q, mec.kA, kF)
    phi = np.dot(phiOp, GAF)

    if dcpypsrc.debug: print 'phiS=', phi
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

    kE = mec.kA + mec.kB
    k = mec.kA + mec.kB + mec.kC
    QCB = mec.Q[kE:k, mec.kA:kE]
    QCA = mec.Q[kE:k, 0:mec.kA]
    uA = np.ones((mec.kA,1))

    p = pinf(mec)
    pC = p[kE:k]
    GAB, GBA = iGs(mec.Q, mec.kA, mec.kB)
    nom = np.dot(pC,(np.dot(QCB,GBA)+QCA))
    denom = np.dot(nom,uA)
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

    uA = np.ones((mec.kA,1))
    I = np.eye(mec.kA)
    GAB, GBA = iGs(mec.Q, mec.kA, mec.kB)
    eB = np.dot((I - np.dot(GAB, GBA)),uA)
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

    kE = mec.kA + mec.kB
    QBB = mec.Q[mec.kA:kE, mec.kA:kE]
    QAA = mec.Q[0:mec.kA, 0:mec.kA]
    QAB = mec.Q[0:mec.kA, mec.kA:kE]
    uA = np.ones((mec.kA, 1))
    I = np.eye(mec.kA)

    phiB = phiBurst(mec)
    GAB, GBA = iGs(mec.Q, mec.kA, mec.kB)
    invQAA = -1 * nplin.inv(QAA)
    invQBB = nplin.inv(QBB)
    interm1 = nplin.inv(I - np.dot(GAB, GBA))
    interm2 = I - np.dot(np.dot(QAB, invQBB), GBA)
    m = (np.dot(np.dot(np.dot(np.dot(phiB, interm1), invQAA),
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

    phiB = phiBurst(mec)
    GAB, GBA = iGs(mec.Q, mec.kA, mec.kB)
    interm = nplin.inv(I - np.dot(GAB, GBA))
    mu = np.dot(np.dot(phiB, interm), uA)[0]
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

    phiB = phiBurst(mec)
    GAB, GBA = iGs(mec.Q, mec.kA, mec.kB)
    eB = endBurst(mec)

    GG = np.dot(GAB, GBA)
    if r == 1:
        interm = np.eye(mec.kA)
    elif r == 2:
        interm = GG
    else:
        interm = GG
        for i in range(2, r):
            interm = np.dot(interm, GG)
    Pr = np.dot(np.dot(phiB, interm), eB)
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

    kE = mec.kA + mec.kB
    QEE = mec.Q[0:kE, 0:kE]
    QAA = mec.Q[0:mec.kA, 0:mec.kA]
    phiB = phiBurst(mec)
    eB = endBurst(mec)
    expQEEA = expQsub(QEE, t)[0:mec.kA, 0:mec.kA]
    f = np.dot(np.dot(np.dot(phiB, expQEEA), -QAA), eB)
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

    phiOp = phiO(mec)
    QAA = mec.Q[0:mec.kA, 0:mec.kA]
    uA = np.ones((mec.kA, 1))
    expQAA = expQsub(QAA, t)
    f = np.dot(np.dot(np.dot(phiOp, expQAA), -QAA), uA)
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

    kF = mec.kB + mec.kC
    k = mec.kA + mec.kB + mec.kC
    phiSht = phiS(mec)
    QFF = mec.Q[mec.kA:k, mec.kA:k]
    uF = np.ones((kF, 1))
    expQFF = expQsub(QFF, t)
    f = np.dot(np.dot(np.dot(phiSht, expQFF), -QFF), uF)
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
    dW : array_like, shape (k2, k2)
    """

    I2 = np.eye(k2)
    I1 = np.eye(k1)
    IQ22 = s * I2 - Q22
    expIQ22 = getExpQsub(tres, -IQ22)
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
    H : array_like, shape (k2, k2)
    """

    I = np.eye(k1)
    X11 = s * I - Q11
    invX11 = nplin.inv(X11)
    expX11 = getExpQsub(tres, -X11)
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
    W : array_like, shape (k2, k2)
    """

    I = np.eye(k2)
    W = s * I - H(s, tres, Q11, Q22, Q21, Q12, k1, k2)
    return W
