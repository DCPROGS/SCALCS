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

import numpy as np
from numpy import linalg as nplin
import math

#import dcpypsrc

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
    # DO NOT DELETE commented explicit loops for future reference
    #
    # rev. 1
    # for i in range(k):
    #     X = np.empty((k, 1))
    #     Y = np.empty((1, k))
    #     X[:, 0] = M[:, i]
    #     Y[0] = N[i]
    #     A[i] = np.dot(X, Y)
    # END DO NOT DELETE
    #
    # rev. 2 - cumulative time fastest on my system
    for i in range(k):
        A[i] = np.dot(M[:, i].reshape(k, 1), N[i].reshape(1, k))

    # rev. 3 - cumulative time not faster
    # A = np.array([
    #         np.dot(M[:, i].reshape(k, 1), N[i].reshape(1, k)) \
    #             for i in range(k)
    #         ])

    return eigvals, A

def eigs_sorted(Q):
    """
    Calculate eigenvalues and spectral matrices of a matrix Q. 
    Return eigenvalues in ascending order 

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
    for i in range(k):
        A[i] = np.dot(M[:, i].reshape(k, 1), N[i].reshape(1, k))
    sorted_indices = eigvals.real.argsort()
    eigvals = eigvals[sorted_indices]
    A = A[sorted_indices, : , : ]
    return eigvals, A

def expQt(M, t):
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

    eigvals, A = eigs(M)

    # DO NOT DELETE commented explicit loops for future reference
    # k = M.shape[0]
    # expM = np.zeros((k, k))
    # rev. 1
    # TODO: avoid loops
    #    for i in range(k):
    #        for j in range(k):
    #            for m in range(k):
    #                expM[i, j] += A[m, i, j] * math.exp(eigvals[m] * t)
    #
    # rev.2:
    # for m in range(k):
    #     expM += A[m] * math.exp(eigvals[m] * t)
    # END DO NOT DELETE

    expM = np.sum(A * np.exp(eigvals * t).reshape(A.shape[0],1,1), axis=0)
    return expM

def Qpow(M, n):
    """
    Rise matrix M to power n.

    Parameters
    ----------
    M : array_like, shape (k, k)
    n : int
        Power.

    Returns
    -------
    Mn : ndarray, shape (k, k)
    """

    k = M.shape[0]
    eig, A = eigs(M)
    Mn = np.zeros((k, k))
    for i in range(k):
        Mn += A[i, :, :] * pow(eig[i], n)
    return Mn

def pinf1(Q):
    """
    Calculate equilibrium occupancies by adding a column of ones
    to Q matrix.
    Pinf = uT * invert((S * transpos(S))).

    Parameters
    ----------
    Q : array_like, shape (k, k)

    Returns
    -------
    pinf : ndarray, shape (k1)
    """

    u = np.ones((Q.shape[0],1))
    S = np.concatenate((Q, u), 1)
    pinf = np.dot(u.transpose(), nplin.inv((np.dot(S,S.transpose()))))[0]
    return pinf

def pinf(Q):
    """
    Calculate equilibrium occupancies with the reduced Q-matrix method.

    Parameters
    ----------
    Q : array_like, shape (k, k)

    Returns
    -------
    pinf : ndarray, shape (k1)
    """

    R = (Q - Q[-1: , :])[:-1, :-1]
    r = Q[-1: , :-1]
    pinf = -np.dot(r, nplin.inv(R))
    pinf = np.append(pinf, 1 - np.sum(pinf))
    return pinf

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

def iGt(t, QAA, QAB):
    """
    GAB(t) = PAA(t) * QAB      Eq. 1.20 in CH82
    PAA(t) = exp(QAA * t)      Eq. 1.16 in CH82
    """

    GAB = np.dot(expQt(QAA, t), QAB)
    return GAB

def eGs(GAF, GFA, kA, kF, expQFF):
    """
    Calculate eGAF, probabilities from transitions from apparently open to
    shut states regardles of when the transition occurs. Thease are Laplace
    transform of eGAF(t) when s=0. Used to calculat initial HJC vectors (HJC92).
    eGAF*(s=0) = (I - GAF * (I - expQFF) * GFA)^-1 * GAF * expQFF
    To caculate eGFA exhange A by F and F by A in function call.

    Parameters
    ----------
    GAF : array_like, shape (kA, kF)
    GFA : array_like, shape (kF, kA)
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.
    
    Returns
    -------
    eGAF : array_like, shape (kA, kF)
    """

    temp = np.eye(kA) - np.dot(np.dot(GAF, np.eye(kF) - expQFF), GFA)
    eGAF = np.dot(np.dot(nplin.inv(temp), GAF), expQFF)
    return eGAF

def phiA(mec):
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
    pI = pinf(mec.Q)[mec.kA:]
    nom = np.dot(pI, mec.QIA)
    denom = np.dot(nom,uA)
    phi = nom / denom
    return phi

def phiF(mec):
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

    GAF, GFA = iGs(mec.Q, mec.kA, mec.kI)
    phi = np.dot(phiA(mec), GAF)
    return phi

def phiSub(Q, k1, k2):
    """
    Calculate initial vector for any subset.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    phi : ndarray, shape (kA)
    """

    u = np.ones((k2 - k1 + 1, 1))
    p = pinf(Q)
    p1, p2, p3 = np.hsplit(p,(k1, k2+1))
    p1c = np.hstack((p1, p3))

    #Q = Q.copy()
    Q1, Q2, Q3 = np.hsplit(Q,(k1, k2+1))
    Q21, Q22, Q23 = np.hsplit(Q2.transpose(),(k1, k2+1))
    Q22c = Q22.copy()
    Q12 = np.vstack((Q21.transpose(), Q23.transpose()))

    nom = np.dot(p1c, Q12)
    denom = np.dot(nom,u)
    phi = nom / denom
    return phi, Q22c

def phiHJC(eGAF, eGFA, kA):
    """
    Calculate initial HJC vector for openings by solving
    phi*(I-eGAF*eGFA)=0 (Eq. 10, HJC92)
    For shuttings exhange A by F and F by A in function call.

    Parameters
    ----------
    eGAF : array_like, shape (kA, kF)
    eGFA : array_like, shape (kF, kA)
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.

    Returns
    -------
    phi : array_like, shape (kA)
    """

    if kA == 1:
        phi = np.array([1])

    else:
        Qsub = np.eye(kA) - np.dot(eGAF, eGFA)
        u = np.ones((kA, 1))
        S = np.concatenate((Qsub, u), 1)
        phi = np.dot(u.transpose(), nplin.inv(np.dot(S, S.transpose())))[0]

    return phi

def H(s, tres, QAA, QFF, QAF, QFA, kF):
    """
    Evaluate H(s) funtion (Eq. 54, HJC92).
    HAA(s) = QAA + QAF * (s*I - QFF) ^(-1) * (I - exp(-(s*I - QFF) * tau)) * QFA
    To evaluate HFF(s) exhange A by F and F by A in function call.

    Parameters
    ----------
    s : float
        Laplace transform argument.
    tres : float
        Time resolution (dead time).
    QAA : array_like, shape (kA, kA)
    QFF : array_like, shape (kF, kF)
    QAF : array_like, shape (kA, kF)
    QFA : array_like, shape (kF, kA)
        QAA, QFF, QAF, QFA - submatrices of Q.
    kF : int
        A number of shut states in kinetic scheme.

    Returns
    -------
    H : ndarray, shape (kA, kA)
    """

    IF = np.eye(kF)
    XFF = s * IF - QFF
    invXFF = nplin.inv(XFF)
    expXFF = expQt(-XFF, tres)
    H = QAA + np.dot(np.dot(np.dot(QAF, invXFF), IF - expXFF), QFA)
    return H

def W(s, tres, QAA, QFF, QAF, QFA, kA, kF):
    """
    Evaluate W(s) function (Eq. 52, HJC92).
    WAA(s) = s * IA - HAA(s)
    To evaluate WFF(s) exhange A by F and F by A in function call.

    Parameters
    ----------
    s : float
        Laplace transform argument.
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
    W : ndarray, shape (k2, k2)
    """

    IA = np.eye(kA)
    W = s * IA - H(s, tres, QAA, QFF, QAF, QFA, kF)
    return W

def detW(s, tres, QAA, QFF, QAF, QFA, kA, kF):
    """
    Calculate determinant of WAA(s).
    To evaluate WFF(s) exhange A by F and F by A in function call.

    Parameters
    ----------
    s : float
        Laplace transform argument.
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
    detWAA : float
    """

    return nplin.det(W(s, tres, QAA, QFF, QAF, QFA, kA, kF))

def dW(s, tres, QAF, QFF, QFA, kA, kF):
    """
    Evaluate the derivative with respect to s of the matrix W(s) at the root s
    (Eq. 56, HJC92) for open states. For same evaluation for shut states
    exhange A by F and F by A in function call.
    W'(s) = I + QAF * [SFF(s) * (s*I - QFF)^(-1) - tau * (I - SFF(s))] * eGFA(s)
    where SFF(s) = I - exp(-(s*I - QFF) * tau) (Eq. 17, HJC92)
    and eGFA(s) = (s*I - QFF)^(-1) * QFA (Eq. 4, HJC92).

    Parameters
    ----------
    s : float
        Laplace transform argument.
    tres : float
        Time resolution (dead time).
    QAF : array_like, shape (kA, kF)
    QFF : array_like, shape (kF, kF)
    QFA : array_like, shape (kF, kA)
        QAF, QFF, QFA - submatrices of Q.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.

    Returns
    -------
    dW : ndarray, shape (kF, kF)
    """

    IF = np.eye(kF)
    IA = np.eye(kA)
    XFF = s * IF - QFF
    expXFF = expQt(-XFF, tres)
    SFF = IF - expXFF
    eGFAs = np.dot(nplin.inv(s * IF - QFF), QFA)
    w1 = np.dot(SFF, nplin.inv(s * IF - QFF)) - tres * (IF - SFF)
    dW = IA + np.dot(np.dot(QAF, w1), eGFAs)
    return dW

def dARSdS(tres, QAA, QFF, GAF, GFA, expQFF, kA, kF):
    r"""
    Evaluate the derivative with respect to s of the Laplace transform of the
    survival function (Eq. 3.6, CHS96) for open states:

    .. math::

       \left[ -\frac{\text{d}}{\text{d}s} {^\cl{A}\!\bs{R}^*(s)} \right]_{s=0}

    For same evaluation for shut states exhange A by F and F by A in function call.

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
    QAA : array_like, shape (kA, kA)
    QAF : array_like, shape (kA, kF)
    QFF : array_like, shape (kF, kF)
    QFA : array_like, shape (kF, kA)
        Q11, Q12, Q22, Q21 - submatrices of Q.
    GAF : array_like, shape (kA, kF)
    GFA : array_like, shape (kF, kA)
        GAF, GFA - G matrices.
    expQFF : array_like, shape(kF, kF)
    expQAA : array_like, shape(kA, kA)
        expQFF, expQAA - exponentials of submatrices QFF and QAA.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.

    Returns
    -------
    DARS : array_like, shape (kA, kA)
    """

    invQAA = nplin.inv(QAA)
    invQFF = nplin.inv(QFF)

    #SFF = I - EXPQF
    I = np.eye(kF)
    SFF = I - expQFF

    #Q1 = tres * GAF * exp(QFF*tres) * GFA
    Q1 = tres * np.dot(GAF, np.dot(expQFF, GFA))
    #Q2 = GAF * SFF * inv(QFF) * GFA
    Q2 = np.dot(GAF, np.dot(SFF, np.dot(invQFF, GFA)))
    #Q3 = -inv(QAA) * GAF * SFF * GFA
    Q3 = np.dot(np.dot(np.dot(-invQAA, GAF), SFF), GFA)
    Q1 = Q1 - Q2 + Q3

    # VA = I - GAF * SFF * GFA
    I = np.eye(kA)
    VA = I - np.dot(np.dot(GAF, SFF), GFA)

    # DARS = inv(VA) * (QAA**-2) - inv(VA) * Q1 * inv(VA) * inv(QAA) =
    #      = inv(VA) * [inv(QAA) - Q1 * inv(VA)] * inv(QAA)
    Q3 = invQAA + - np.dot(Q1, nplin.inv(VA))
    DARS = np.dot(np.dot(nplin.inv(VA), Q3), invQAA)

    return DARS

def AR(roots, tres, QAA, QFF, QAF, QFA, kA, kF):
    """
    
    Parameters
    ----------
    roots : array_like, shape (1, kA)
        Roots of the asymptotic pdf.
    tres : float
        Time resolution (dead time).
    QAA, QFF, QAF, QFA : array_like
        Submatrices of Q.
    kA, kF : ints
        Number of open and shut states.

    Returns
    -------
    R : ndarray, shape(kA, kA, kA)
    """

    R = np.zeros((kA, kA, kA))
    row = np.zeros((kA, kA))
    col1 = np.zeros((kA, kA))
    for i in range(kA):
        WA = W(roots[i], tres, QAA, QFF, QAF, QFA, kA, kF)
        row[i] = pinf(WA)
        AW = np.transpose(WA)
        col1[i] = pinf(AW)
    col = col1.transpose()

    for i in range(kA):
        nom = np.dot(col[:,i].reshape((kA, 1)), row[i,:].reshape((1, kA)))
        W1A = dW(roots[i], tres, QAF, QFF, QFA, kA, kF)
        denom = np.dot(np.dot(row[i,:].reshape((1, kA)), W1A),
            col[:,i].reshape((kA, 1)))
        R[i] = nom / denom

    return R

def HAF(roots, tres, tcrit, QAF, expQFF, R):
    """
    Parameters
    ----------
    roots : array_like, shape (1, kA)
        Roots of the asymptotic pdf.
    tres : float
        Time resolution (dead time).
    tcrit : float
        Critical time.
    QAF : array_like, shape(kA, kF)
    expQFF : array_like, shape(kF, kF)
    R : array_like, shape(kA, kA, kA)

    Returns
    -------
    HAF : ndarray, shape(kA, kF)
    """

    coeff = -np.exp(roots * (tcrit - tres)) / roots
    temp = np.sum(R * coeff.reshape(R.shape[0],1,1), axis=0)
    HAF = np.dot(np.dot(temp, QAF), expQFF)

    return HAF

def CHSvec(roots, tres, tcrit, QFA, kA, expQAA, phiF, R):
    """
    Calculate initial and final CHS vectors for HJC likelihood function
    (Eqs. 5.5 or 5.7, CHS96).

    Parameters
    ----------
    roots : array_like, shape (1, kA)
        Roots of the asymptotic pdf.
    tres : float
        Time resolution (dead time).
    tcrit : float
        Critical time.
    QFA : array_like, shape(kF, kA)
    kA : int
    expQAA : array_like, shape(kA, kA)
    phiF : array_like, shape(1, kF)
    R : array_like, shape(kF, kF, kF)

    Returns
    -------
    start : ndarray, shape (1, kA)
        CHS start vector (Eq. 5.11, CHS96).
    end : ndarray, shape (kF, 1)
        CHS end vector (Eq. 5.8, CHS96).
    """

    H = HAF(roots, tres, tcrit, QFA, expQAA, R)
    u = np.ones((kA, 1))
    start = np.dot(phiF, H) / np.dot(np.dot(phiF, H), u)
    end = np.dot(H, u)

    return start, end

def eGAF(t, tres, eigvals, Z00, Z10, Z11, roots, R, QAF, expQFF):
    #TODO: update documentation
    """
    Calculate transition density eGAF(t) for exact (Eq. 3.2, HJC90) and
    asymptotic (Eq. 3.24, HJC90) distribution.

    Parameters
    ----------
    t : float
        Time interval.
    tres : float
        Time resolution (dead time).
    eigvals : array_like, shape (1, k)
        Eigenvalues of -Q matrix.
    Z00, Z10, Z11 : array_like, shape (k, kA, kF)
        Z constants for the exact open time pdf.
    roots : array_like, shape (1, kA)
        Roots of the asymptotic pdf.
    R : array_like, shape(kA, kA, kA)
    QAF : array_like, shape(kA, kF)
    expQFF : array_like, shape(kF, kF)

    Returns
    -------
    eGAFt : array_like, shape(kA, kF)
    """

    if t < (tres * 2): # exact
        eGAFt = f0((t - tres), eigvals, Z00)
    elif t < (tres * 3):
        eGAFt = (f0((t - tres), eigvals, Z00) -
            f1((t - 2 * tres), eigvals, Z10, Z11))
    else: # asymptotic
        temp = np.sum(R * np.exp(roots *
            (t - tres)).reshape(R.shape[0],1,1), axis=0)
        eGAFt = np.dot(np.dot(temp, QAF), expQFF)

    return eGAFt

def f0(u, eigvals, Z00):
    """
    A component of exact time pdf (Eq. 22, HJC92).

    Parameters
    ----------
    u : float
        u = t - tres
    eigvals : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    Z00 : list of array_likes
        Constants for the exact open/shut time pdf.
        Z00 for likelihood calculation or gama00 for time distributions.

    Returns
    -------
    f : ndarray
    """

#    f = np.zeros(Z00[0].shape)
#    for i in range(len(eigvals)):
#        f += Z00[i] *  math.exp(-eigvals[i] * u)

    if Z00.ndim > 1:
        f = np.sum(Z00 *  np.exp(-eigvals * u).reshape(Z00.shape[0],1,1),
            axis=0)
    else:
        f = np.sum(Z00 *  np.exp(-eigvals * u))
    return f

def f1(u, eigvals, Z10, Z11):
    """
    A component of exact time pdf (Eq. 22, HJC92).

    Parameters
    ----------
    u : float
        u = t - tres
    eigvals : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    Z10, Z11 (or gama10, gama11) : list of array_likes
        Constants for the exact open/shut time pdf. Z10, Z11 for likelihood
        calculation or gama10, gama11 for time distributions.

    Returns
    -------
    f : ndarray
    """

#    f = np.zeros(Z10[0].shape)
#    for i in range(len(eigvals)):
#        f += (Z10[i] + Z11[i] * u) *  math.exp(-eigvals[i] * u)

    if Z10.ndim > 1:
        f = np.sum((Z10 + Z11 * u) *
            np.exp(-eigvals * u).reshape(Z10.shape[0],1,1), axis=0)
    else:
        f = np.sum((Z10 + Z11 * u) * np.exp(-eigvals * u))
    return f

def Zxx(Q, eigen, A, kopen, QFF, QAF, QFA, expQFF, open):
    """
    Calculate Z constants for the exact open time pdf (Eq. 3.22, HJC90).
    Exchange A and F for shut time pdf.

    Parameters
    ----------
    t : float
        Time.
    Q : array_like, shape (k, k)
    kopen : int
        Number of open states.
    QFF, QAF, QFA : array_like
        Submatrices of Q.
    open : bool
        True for open time pdf, False for shut time pdf.

    Returns
    -------
    eigen : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    Z00, Z10, Z11 : array_like, shape (k, kA, kF)
        Z constants for the exact open time pdf.
    """

    k = Q.shape[0]
    kA = k - QFF.shape[0]
#    eigen, A = eigs(-Q)
    # Maybe needs check for equal eigenvalues.

    # Calculate Dj (Eq. 3.16, HJC90) and Cimr (Eq. 3.18, HJC90).
    D = np.empty((k))
    if open:
        C00 = A[:, :kopen, :kopen]
        A1 = A[:, :kopen, kopen:]
    else:
        C00 = A[:, kopen:, kopen:]
        A1 = A[:, kopen:, :kopen]
    D = np.dot(np.dot(A1, expQFF), QFA)

    C11 = np.empty((k, kA, kA))
    #TODO: try to remove 'for' cycles
    for i in range(k):
        C11[i] = np.dot(D[i], C00[i])

    C10 = np.empty((k, kA, kA))
    #TODO: try to remove 'for' cycles
    for i in range(k):
        S = np.zeros((kA, kA))
        for j in range(k):
            if j != i:
                S += ((np.dot(D[i], C00[j]) + np.dot(D[j], C00[i])) /
                    (eigen[j] - eigen[i]))
        C10[i] = S

    M = np.dot(QAF, expQFF)
    Z00 = np.array([np.dot(C, M) for C in C00])
    Z10 = np.array([np.dot(C, M) for C in C10])
    Z11 = np.array([np.dot(C, M) for C in C11])

    return eigen, Z00, Z10, Z11
