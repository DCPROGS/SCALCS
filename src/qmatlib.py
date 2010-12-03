#! /usr/bin/python
"""
dc_pyps project is pure Python implementations of Q-Matrix formalism
for ion channel research. To learn more about kinetic analysis of ion
channels see the references below.

qmatlib module is a collection of functions for Q matrix manipulations.

References:

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

import math
import numpy as np
from numpy import linalg as nplin

def eigs(Q, debug=False):
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
    if debug: print 'eigenvalues=', eigvals
    if debug: print 'eigenvectors=', M
    N = nplin.inv(M)
    k = N.shape[0]

    A = []
    for i in range(0, k):
        Ai = np.zeros((k, k))
        for j in range(0, k):
            for m in range(0, k):
                Ai[j, m] = M[j, i] * N[i, m]
        A.append(Ai)
    if debug: print 'spectral matrices =', A

    return eigvals, A

def iGs(Q, kA, kB, debug=False):
    """
    Calculate and return GBA and GAB matrices (Eq. 1.25, CH82).
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
    if debug: print 'GAB= ', GAB
    if debug: print 'GBA= ', GBA

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

def expQsub(M, t, debug=False):
    """
    Calculate exponential of a matrix M.
        expM = exp(M * t)

    Parameters
    ----------
    M : array_like, shape (k, k)
    t : float

    Returns
    -------
    expM : array_like, shape (k, k)
    """

    eigvals, A = eigs(-M, debug)
    k = M.shape[0]

    expM = np.zeros((k, k))
    for i in range(0, k):
        for j in range(0, k):
            for m in range(0, k):
                temp = A[m][i, j] * math.exp(-eigvals[m] * t)
                expM[i, j] = expM[i, j] + temp

    return expM

def phiHJC(eG12, eG21, k1, k2, debug=False):
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

def phiO(Q, kA, debug=False):
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
    if debug: print 'phiO=', phi
    return phi

def phiS(Q, kA, kB, kC, debug=False):
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

    if debug: print 'phiS=', phi
    return phi