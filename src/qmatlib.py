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
"""

__author__="R.Lape, University College London"
__date__ ="$11-Oct-2010 10:33:07$"

import numpy as np
from numpy import linalg as nplin

import qmatrc

def eigs(Q):
    """
    Calculate eigenvalues, eigenvectors and spectral matrices.
    Return eigenvalues and spectral matrices.

    Parameters
    ----------
    Q : array_like, shape (k, k)

    Returns
    -------
    eigvals : ndarray, shape (k,)
        Eigenvalues of Q.
    A : list of ndarrays with shape (k, k); length `k`
        Spectral matrices of Q.
    """

    eigvals, M = nplin.eig(Q)
    if qmatrc.debug: print 'eigenvalues=', eigvals
    if qmatrc.debug: print 'eigenvectors=', M
    N = nplin.inv(M)
    k = N.shape[0]

    A = []
    for i in range(k):
        Ai = np.zeros((k, k))
        for j in range(k):
            for m in range(k):
                Ai[j, m] = M[j, i] * N[i, m]
        A.append(Ai)
    if qmatrc.debug: print 'spectral matrices =', A

    return eigvals, A

def iGs(Q, kA, kB):
    """
    Calculate GBA and GAB matrices (Eq. 1.25, CH82).
        GBA=-QBB^(-1)*QBA
        GAB=-QAA^(-1)*QAB

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
    if qmatrc.debug: print 'GAB= ', GAB
    if qmatrc.debug: print 'GBA= ', GBA

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

def expQsub(t, M):
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
    """
    Python implementation of DC's DARSDS subroutine (inside HJCMEAN.FOR).
    Evaluate -dAR(s)/ds at s=0 (HJC92).
    Result used for means etc of HJC distributions.

    SFF = I - exp(QFF * tres)
    First evaluate [dVA(s) / ds] * s = 0.
    Q1 = -inv(QAA) * GAF*SFF*GFA - GAF*SFF* inv(QFF)* GFA +
    + tres*GAF* expQFF*GFA

    Then: DARS = inv(VA)*(QAA^-2) - inv(VA)*Q1*inv(VA)*inv(QAA =
    = inv(VA) * [inv(QAA) - Q1*inv(VA)] * inv(QAA)
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

def hjc_mean_time(tres, Q, kA):
    """
    Calculate exact mean open or shut time from HJC probability density
    function. This is Python implementation of DCprogs HJCMEAN.FOR
    subroutine.

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    Q : array_like, shape (k, k)
    kA : int
        Number of open states.

    Returns
    -------
    hmopen : float
        Apparent mean open time.
    hmshut : float
        Apparent mean shut time.
    """

    k = Q.shape[0]
    kF = k - kA
    GAF, GFA = iGs(Q, kA, kF)
    QFF = Q[kA:k, kA:k]
    expQFF = expQsub(tres, QFF)
    QAA = Q[0:kA, 0:kA]
    expQAA = expQsub(tres, QAA)

    #Calculate Gs and initial vectors corrected for missed events.
    eGAF = eGs(GAF, GFA, kA, kF, expQFF)
    eGFA = eGs(GFA, GAF, kF, kA, expQAA)
    phiA = phiHJC(eGAF, eGFA, kA, kF)
    phiF = phiHJC(eGFA, eGAF, kF, kA)

    #Recalculate QexpQA and QexpQF
    QAF = Q[0:kA, kA:k]
    QexpQF = np.dot(QAF, expQFF)
    QFA = Q[kA:k, 0:kA]
    QexpQA = np.dot(QFA, expQAA)

    DARS = dARSdS(tres, QAA, QAF, QFF, QFA, GAF, GFA, expQFF, expQAA,
        kA, kF)
    DFRS = dARSdS(tres, QFF, QFA, QAA, QAF, GFA, GAF, expQAA, expQFF,
        kF, kA)

    uA = np.ones((kA, 1))
    uF = np.ones((kF, 1))
    # meanOpenTime = tres + phiA * DARS * QexpQF * uF
    # meanShutTime = tres + phiF * DFRS * QexpQA * uA
    hmopen = tres + np.dot(phiA, np.dot(np.dot(DARS, QexpQF), uF))
    hmshut = tres + np.dot(phiF, np.dot(np.dot(DFRS, QexpQA), uA))

    return hmopen, hmshut

def popen(Q, kA, tres=0):
    """
    Calculate open probability for any temporal resolution, tres.

    Parameters
    ----------
    Q : array_like, shape (k, k)
    kA : int
        Number of open states.
    tres : float
        Time resolution (dead time).

    Returns
    -------
    Popen : float
        Open probability.
    """

    if tres == 0:
        p = pinf(Q)
        Popen = 0
        for i in range(kA):
            Popen = Popen + p[i]
        return Popen
    else:
        hmopen, hmshut = hjc_mean_time(tres, Q, kA)
        Popen = hmopen / (hmopen + hmshut)
        return Popen[0,0]

def pinf(Q):
    """
    Calculate ecquilibrium occupancies by adding a column of ones
    to Q matrix.
    
    Pinf = uT*invert((S*transpos(S))).

    Parameters
    ----------
    Q : array_like, shape (k, k)

    Returns
    -------
    phi : array_like, shape (k1)
    """

    u = np.ones((Q.shape[0],1))
    S = np.concatenate((Q, u), 1)
    pinf = np.dot(u.transpose(), nplin.inv((np.dot(S,S.transpose()))))[0]
    return pinf

def phiO(Q, kA):
    """
    Calculate initial vector for openings.

    Parameters
    ----------
    Q : array_like, shape (k, k)
    kA : int
        Number of open states.

    Returns
    -------
    phi : array_like, shape (kA)
    """

    k = Q.shape[0]
    uA = np.ones((kA,1))
    p = pinf(Q)
    pF = p[kA:k]
    QFA = Q[kA:k, 0:kA]
    nom = np.dot(pF, QFA)
    denom = np.dot(nom,uA)
    phi = nom / denom
    if qmatrc.debug: print 'phiO=', phi
    return phi

def phiS(Q, kA, kB, kC):
    """
    Calculate inital vector for shuttings.

    Parameters
    ----------
    Q : array_like, shape (k, k)
    kA : int
        Number of open states.
    kB : int
        Number of short lived shut states.
    kC : int
        Number of long lived shut states.

    Returns
    -------
    phi : array_like, shape (kB+kC)
    """

    kF = kB + kC
    phiOp = phiO(Q, kA)
    GAF, GFA = iGs(Q, kA, kF)
    phi = np.dot(phiOp, GAF)

    if qmatrc.debug: print 'phiS=', phi
    return phi

def phiBurst(Q, kA, kB, kC):
    """
    Calculate the start probabilities of a burst (Eq. 3.2, CH82).
    PhiB = (pCinf * (QCB * GBA + QCA)) / (pCinf * (QCB * GBA + QCA) * uA)

    Parameters
    ----------
    Q : array_like, shape (k, k)
    kA : int
        Number of open states.
    kB : int
        Number of short lived shut states.
    kC : int
        Number of long lived shut states.

    Returns
    -------
    phiB : array_like, shape (1, kA)
    """

    kE = kA + kB
    k = kA + kB + kC
    QCB = Q[kE:k, kA:kE]
    QCA = Q[kE:k, 0:kA]
    uA = np.ones((kA,1))

    p = pinf(Q)
    pC = p[kE:k]
    GAB, GBA = iGs(Q, kA, kB)
    nom = np.dot(pC,(np.dot(QCB,GBA)+QCA))
    denom = np.dot(nom,uA)
    phiB = nom / denom
    return phiB

def endBurst(Q, kA, kB, kC):
    """
    Calculate the end vector for a burst (Eq. 3.4, CH82).
    eB = (I-GAB * GBA) * uA

    Parameters
    ----------
    Q : array_like, shape (k, k)
    kA : int
        Number of open states.
    kB : int
        Number of short lived shut states.
    kC : int
        Number of long lived shut states.

    Returns
    -------
    eB : array_like, shape (kA, 1)
    """

    uA = np.ones((kA,1))
    I = np.eye(kA)
    GAB, GBA = iGs(Q, kA, kB)
    eB = np.dot((I - np.dot(GAB, GBA)),uA)
    return eB

def mean_burst_length(Q, kA, kB, kC):
    """
    Calculate the mean burst length (Eq. 3.19, CH82).
    m = PhiB * (I - GAB * GBA)^(-1) * (-QAA^(-1)) *
        (I - QAB * (QBB^(-1)) * GBA) * uA

    Parameters
    ----------
    Q : array_like, shape (k, k)
    kA : int
        Number of open states.
    kB : int
        Number of short lived shut states.
    kC : int
        Number of long lived shut states.

    Returns
    -------
    m : float
        The mean burst length.
    """

    kE = kA + kB
    QBB = Q[kA:kE, kA:kE]
    QAA = Q[0:kA, 0:kA]
    QAB = Q[0:kA, kA:kE]
    uA = np.ones((kA, 1))
    I = np.eye(kA)

    phiB = phiBurst(Q, kA, kB, kC)
    GAB, GBA = iGs(Q, kA, kB)
    invQAA = -1 * nplin.inv(QAA)
    invQBB = nplin.inv(QBB)
    interm1 = nplin.inv(I - np.dot(GAB, GBA))
    interm2 = I - np.dot(np.dot(QAB, invQBB), GBA)
    m = (np.dot(np.dot(np.dot(np.dot(phiB, interm1), invQAA),
        interm2), uA)[0])
    return m

def mean_num_burst_openings(Q, kA, kB, kC):
    """
    Calculate the mean number of openings per burst (Eq. 3.7, CH82).
    mu = phiB * (I - GAB * GBA)^(-1) * uA

    Parameters
    ----------
    Q : array_like, shape (k, k)
    kA : int
        Number of open states.
    kB : int
        Number of short lived shut states.
    kC : int
        Number of long lived shut states.

    Returns
    -------
    mu : float
        The mean number ofopenings per burst.
    """

    uA = np.ones((kA,1))
    I = np.eye(kA)

    phiB = phiBurst(Q, kA, kB, kC)
    GAB, GBA = iGs(Q, kA, kB)
    interm = nplin.inv(I - np.dot(GAB, GBA))
    mu = np.dot(np.dot(phiB, interm), uA)[0]
    return mu

def distr_num_burst_openings(r, Q, kA, kB, kC):
    """
    The distribution of openings per burst (Eq. 3.5, CH82).
    P(r) = phiB * (GAB * GBA)^(r-1) * eB

    Parameters
    ----------
    r : int
        Number of openings per burst.
    Q : array_like, shape (k, k)
    kA : int
        Number of open states.
    kB : int
        Number of short lived shut states.
    kC : int
        Number of long lived shut states.

    Returns
    -------
    Pr : float
        Probability of seeing r openings per burst.
    """

    phiB = phiBurst(Q, kA, kB, kC)
    GAB, GBA = iGs(Q, kA, kB)
    eB = endBurst(Q, kA, kB, kC)

    GG = np.dot(GAB, GBA)
    if r == 1:
        interm = np.eye(kA)
    elif r == 2:
        interm = GG
    else:
        interm = GG
        for i in range(2, r):
            interm = np.dot(interm, GG)
    Pr = np.dot(np.dot(phiB, interm), eB)
    return Pr

def distr_burst_length(t, Q, kA, kB, kC):
    """
    Probability density function of the burst length (Eq. 3.17, CH82).
    f(t) = phiB * [PEE(t)]AA * (-QAA) * eB, where PEE(t) = exp(QEE * t)

    Parameters
    ----------
    t : float
        Time.
    Q : array_like, shape (k, k)
    kA : int
        Number of open states.
    kB : int
        Number of short lived shut states.
    kC : int
        Number of long lived shut states.

    Returns
    -------
    f : float
    """

    
    kE = kA + kB
    QEE = Q[0:kE, 0:kE]
    QAA = Q[0:kA, 0:kA]

    phiB = phiBurst(Q, kA, kB, kC)
    eB = endBurst(Q, kA, kB, kC)
    expQEEA = expQsub(t, QEE)[0:kA, 0:kA]
    f = np.dot(np.dot(np.dot(phiB, expQEEA), -QAA), eB)
    return f
