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

import math

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
    
    # Sort eigenvalues in ascending order. 
    sorted_indices = eigvals.real.argsort()
    eigvals = eigvals[sorted_indices]
    A = A[sorted_indices, : , : ]

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

    # rev. 3 - fastest for me
    expM = np.sum(A * np.exp(eigvals * t).reshape(A.shape[0],1,1), axis=0)
    
    # rev. 4 - slower on my system despite math.exp
    # expM = np.sum(A * np.array([
    #            math.exp(eigval * t) for eigval in eigvals
    #            ]).reshape(A.shape[0],1,1), 
    #               axis=0)

    return expM

def phiHJC(eGAF, eGFA, kA):
    """
    Calculate equilibrium probability vector phi by solving
        phi*(I-eG12*eG21)=0 (Eq. 10, HJC92)

    Parameters
    ----------
    eGAF : array_like, shape (kA, kF)
    eGFA : array_like, shape (kF, kA)
    kA : int
        A number of open/shut states in kinetic scheme.
    kF : int
        A number of shut/open states in kinetic scheme.

    Returns
    -------
    phi : array_like, shape (kA)
    """

    if kA == 1:
        phi = 1
        return phi

    Qsub = np.eye(kA) - np.dot(eGAF, eGFA)
    u = np.ones((kA, 1))
    S = np.concatenate((Qsub, u), 1)
    phi = np.dot(u.transpose(), nplin.inv(np.dot(S, S.transpose())))

    return phi[0]

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

def pinf(Q):
    """
    Calculate ecquilibrium occupancies by adding a column of ones
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
    pF = pinf(mec.Q)[mec.kA:]
    nom = np.dot(pF, mec.QFA)
    denom = np.dot(nom,uA)
    phi = nom / denom
    return phi

def phiA(mec, k1, k2):
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
    p = pinf(mec.Q)
    #print "pinf=", p
    p1, p2, p3 = np.hsplit(p,(k1, k2+1))
    pF = np.hstack((p1, p3))
    #print "pF=", pF

    Q = mec.Q.copy()
    #print "Q=", Q
    Q1, Q2, Q3 = np.hsplit(Q,(k1, k2+1))
    Q21, Q22, Q23 = np.hsplit(Q2.transpose(),(k1, k2+1))
    QAA = Q22.copy()
    QFA = np.vstack((Q21.transpose(), Q23.transpose()))
    #print "QFA=", QFA

    nom = np.dot(pF, QFA)
    denom = np.dot(nom,u)
    phi = nom / denom
    #print "phi=", phi
    return phi, QAA

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

#def phiBurst(mec):
#    """
#    Calculate the start probabilities of a burst (Eq. 3.2, CH82).
#    PhiB = (pCinf * (QCB * GBA + QCA)) / (pCinf * (QCB * GBA + QCA) * uA)
#
#    Parameters
#    ----------
#    mec : dcpyps.Mechanism
#        The mechanism to be analysed.
#
#    Returns
#    -------
#    phiB : array_like, shape (1, kA)
#    """
#
#    uA = np.ones((mec.kA, 1))
#    pC = pinf(mec.Q)[mec.kE:]
#    nom = np.dot(pC, (np.dot(mec.QCB, mec.GBA) + mec.QCA))
#    denom = np.dot(nom, uA)
#    phiB = nom / denom
#    return phiB
#
#def endBurst(mec):
#    r"""
#    Calculate the end vector for a burst (Eq. 3.4, CH82).
#
#    .. math::
#
#       \bs{e}_\text{b} = (\bs{I}-\bs{G}_\cl{AB} \bs{G}_\cl{BA}) \bs{u}_\cl{A}
#
#    Parameters
#    ----------
#    mec : dcpyps.Mechanism
#        The mechanism to be analysed.
#
#    Returns
#    -------
#    eB : array_like, shape (kA, 1)
#    """
#
#    uA = np.ones((mec.kA, 1))
#    I = np.eye(mec.kA)
#    eB = np.dot((I - np.dot(mec.GAB, mec.GBA)), uA)
#    return eB

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
    expIQ22 = expQt(-IQ22, tres)
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
    expX11 = expQt(-X11, tres)
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

def eGAF(t, tres, roots, XAF, eigvals, Z00, Z10, Z11):
    """

    Parameters
    ----------
    t : float
        Time interval.
    tres : float
        Time resolution (dead time).
    roots : array_like, shape (1, kA)
        Roots of the asymptotic pdf.
    XAF : array_like, shape(kA, kA, kF)
    eigvals : array_like, shape (1, k)
        Eigenvalues of -Q matrix.
    Z00, Z10, Z11 : array_like, shape (k, kA, kF)
        Z constants for the exact open time pdf.

    Returns
    -------
    eGAFt : array_like, shape(kA, kA, kF)
    """

    #eGAFt = np.zeros(XAF[0].shape)
    if t < tres * 3: # exact
        if t < tres * 2:
            eGAFt = f0((t - tres), eigvals, Z00)
        else:
            eGAFt = (f0((t - tres), eigvals, Z00) -
                f1((t - 2 * tres), eigvals, Z10, Z11))
    else: # asymptotic
#        for i in range(len(roots)):
#            eGAFt += XAF[i] * math.exp(roots[i] * (t - tres))
        eGAFt = np.sum(XAF * np.exp(roots *
            (t - tres)).reshape(XAF.shape[0],1,1), axis=0)



    return eGAFt

def XAF(tres, roots, QAA, QFF, QAF, QFA, expQFF):
    """

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    roots : array_like, shape (1, kA)
        Roots of the asymptotic open time pdf.
    QAA, QFF, QAF, QFA : array_like
        Submatrices of Q.

    Returns
    -------
    X : array_like, shape(kA, kA, kF)
    """

    kA = QAA.shape[0]
    kF = QFF.shape[0]
#    expQFF = expQt(QFF, tres)
    X = np.zeros((kA, kA, kF))
    rowA = np.zeros((kA, kA))
    colA = np.zeros((kA, kA))
    for i in range(kA):
        WAA = W(roots[i], tres, QFF, QAA, QAF, QFA, kF, kA)
        rowA[i] = pinf(WAA)
        TrWAA = np.transpose(WAA)
        colA[i] = pinf(TrWAA)
    colA = np.transpose(colA)

    for i in range(kA):
        nom = np.dot(np.dot(np.dot(colA[:,i].reshape((kA, 1)),
            rowA[i,:].reshape((1, kA))), QAF), expQFF)
        W1A = dW(roots[i], tres, QAF, QFF, QFA, kA, kF)
        denom = np.dot(np.dot(rowA[i,:].reshape((1, kA)), W1A),
            colA[:,i].reshape((kA, 1)))
        X[i] = nom / denom
    return X

def Zxx(t, Q, kopen, QFF, QAF, QFA, expQFF):
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

    Returns
    -------
    eigen : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    Z00, Z10, Z11 : array_like, shape (k, kA, kF)
        Z constants for the exact open time pdf.
    """

    open = True
    k = Q.shape[0]
    kA = k - QFF.shape[0]
    if kA != kopen:
        open = False
#    expQFF = expQt(QFF, t)
    eigen, A = eigs(-Q)
    # Maybe needs check for equal eigenvalues.

    # Calculate Dj (Eq. 3.16, HJC90) and Cimr (Eq. 3.18, HJC90).
    #D = []
    #C00 = []
    #C11 = []
    #C10 = []

    D = np.empty((k))
    if open:
        C00 = A[:, :kopen, :kopen]
        A1 = A[:, :kopen, kopen:]
        D = np.dot(np.dot(A1, expQFF), QFA)
    else:
        C00 = A[:, kopen:, kopen:]
        A1 = A[:, kopen:, :kopen]
        D = np.dot(np.dot(A1, expQFF), QFA)

    C11 = D * C00

    #for i in range(k):
        #if open:
            #D[i] = (np.dot(np.dot(A1[i], expQFF), QFA))
            #C00.append(A[i, :kopen, :kopen])
        #else:
            #D[i] = (np.dot(np.dot(A1[i], expQFF), QFA))
            #C00.append(A[i, kopen:, kopen:])
        #C11.append(np.dot(D[i], C00[i]))

    C10 = np.empty((k, kA, kA))
    for i in range(k):
        S = np.zeros((kA, kA))
        for j in range(k):
            if j != i:
                S += ((np.dot(D[i], C00[j]) + np.dot(D[j], C00[i])) /
                    (eigen[j] - eigen[i]))
        C10[i] = S

#    Z00 = []
#    Z10 = []
#    Z11 = []
    M = np.dot(QAF, expQFF)
#    for i in range(k):
#        Z00.append(np.dot(C00[i], M))
#        Z10.append(np.dot(C10[i], M))
#        Z11.append(np.dot(C11[i], M))

    Z00 = np.array([np.dot(C, M) for C in C00])
    Z10 = np.array([np.dot(C, M) for C in C10])
    Z11 = np.array([np.dot(C, M) for C in C11])

    return eigen, Z00, Z10, Z11
