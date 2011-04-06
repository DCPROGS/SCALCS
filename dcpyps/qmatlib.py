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
    k = M.shape[0]
    expM = np.zeros((k, k))
    #TODO: avoid loops
    for i in range(k):
        for j in range(k):
            for m in range(k):
                expM[i, j] += A[m, i, j] * np.exp(eigvals[m] * t)
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
    pC = pinf(mec.Q)[mec.kE:]
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
    if dcpypsrc.debug:
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
        Asymptotic root at wich \|W\|=0.
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


def f0(u, eigvals, gamma00):
    """
    A component of exact time pdf (Eq. 22, HJC92).

    Parameters
    ----------
    u : float
        u = t - tres
    eigvals : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    gama00 : list of floats
        Constants for the exact open/shut time pdf.

    Returns
    -------
    f : float
    """

    print 'gamma00=', gamma00
    print 'eigvals=', eigvals
    f = np.sum(gamma00 * np.exp(-eigvals * u))
    return f

def f1(u, eigvals, gamma10, gamma11):
    """
    A component of exact time pdf (Eq. 22, HJC92).

    Parameters
    ----------
    u : float
        u = t - tres
    eigvals : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    gama10, gama11 : lists of floats
        Constants for the exact open/shut time pdf.

    Returns
    -------
    f : float
    """
    f = np.sum((gamma10 + gamma11 * u) * np.exp(-eigvals * u))
    return f

def lf0(u, eigvals, Z00):
    """
    A component of exact time pdf (Eq. 22, HJC92).

    Parameters
    ----------
    u : float
        u = t - tres
    eigvals : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    Z00 :
        Constants for the exact open/shut time pdf.

    Returns
    -------
    f : float
    """

    f = np.zeros(Z00[0].shape)
    for i in range(len(eigvals)):
        f += np.sum(Z00[i] *  np.exp(-eigvals[i] * u))
    return f

def lf1(u, eigvals, Z10, Z11):
    """
    A component of exact time pdf (Eq. 22, HJC92).

    Parameters
    ----------
    u : float
        u = t - tres
    eigvals : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    Z10, Z11 : lists of floats
        Constants for the exact open/shut time pdf.

    Returns
    -------
    f : float
    """

    f = np.zeros(Z10[0].shape)
    for i in range(len(eigvals)):
        f += np.sum((Z10[i] + Z11[i] * u) *  np.exp(-eigvals[i] * u))
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

    eGAFt = np.zeros(XAF[0].shape)

    if t < tres * 3: # exact
        if t < tres * 2:
            eGAFt = lf0((t - tres), eigvals, Z00)
        else:
            ff0 = lf0((t - tres), eigvals, Z00)
            ff1 = lf1((t - 2 * tres), eigvals, Z10, Z11)
            eGAFt = ff0 - ff1
    else: # asymptotic
        for i in range(len(roots)):
            eGAFt += XAF[i] * math.exp(-roots[i] * (t - tres))

    return eGAFt


def XAF(tres, roots, QAA, QFF, QAF, QFA):
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
    expQFF = expQt(QFF, tres)
    X = np.zeros((kA, kA, kF))
    rowA = np.zeros((kA, kA))
    colA = np.zeros((kA, kA))
    for i in range(kA):
        WAA = W(roots[i], tres, QFF, QAA, QAF, QFA, kF, kA)
        rowA[i] = pinf(WAA)
    colA = np.transpose(nplin.inv(rowA))

    for i in range(kA):
        nom = np.dot(np.dot((colA[i].reshape((kA, 1)) * rowA[i]), QAF), expQFF)
        W1A = dW(roots[i], tres, QAF, QFF, QFA, kA, kF)
        denom = np.dot(np.dot(rowA[i], W1A), colA[i])
        X[i] = nom / denom
    return X

def Zxx(t, Q, kopen, QFF, QAF, QFA):
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
    expQFF = expQt(QFF, t)
    eigen, A = eigs(-Q)
    # Maybe needs check for equal eigenvalues.

    # Calculate Dj (Eq. 3.16, HJC90) and Cimr (Eq. 3.18, HJC90).
    D = []
    C00 = []
    C11 = []
    C10 = []

    for i in range(k):
        if open:
            D.append(np.dot(np.dot(A[i, :kopen, kopen:], expQFF), QFA))
            C00.append(A[i, :kopen, :kopen])
        else:
            D.append(np.dot(np.dot(A[i, kopen:, :kopen], expQFF), QFA))
            C00.append(A[i, kopen:, kopen:])
        C11.append(np.dot(D[i], C00[i]))

    for i in range(k):
        S = np.zeros((kA, kA))
        for j in range(k):
            if j != i:
                S += ((np.dot(D[i], C00[j]) + np.dot(D[j], C00[i])) /
                    (eigen[j] - eigen[i]))
        C10.append(S)

    Z00 = []
    Z10 = []
    Z11 = []
    M = np.dot(QAF, expQFF)
    for i in range(k):
        Z00.append(np.dot(C00[i], M))
        Z10.append(np.dot(C10[i], M))
        Z11.append(np.dot(C11[i], M))

    return eigen, Z00, Z10, Z11

def GAMAxx(Z00, Z10, Z11, phiA):
    """
    Calculate gama constants for the exact open time pdf (Eq. 3.22, HJC90).
    Exchange A and F for shut time pdf.

    Parameters
    ----------
    Z00, Z10, Z11 : array_like, shape (k, kA, kF)
        Z constants for the exact open/shut time pdf.
    phiA : array_like, shape (1, kA)
        Initial vector for openings.

    Returns
    -------
    gama00, gama10, gama11 : array_like, shape (1, k)
        Gama constants.
    """
    
    uA = np.ones((phiA.shape[0], 1))
    gama00 = []
    gama10 = []
    gama11 = []
    for i in range(len(Z00)):
        gama00.append(np.dot(np.dot(phiA, Z00[i]), uA))
        gama10.append(np.dot(np.dot(phiA, Z10[i]), uA))
        gama11.append(np.dot(np.dot(phiA, Z11[i]), uA))
    return gama00, gama10, gama11
