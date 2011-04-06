"""A collection of functions for single channel or macroscopic
current calculations.

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

import math

import numpy as np
from numpy import linalg as nplin

import qmatlib as qml

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
        DARS = qml.dARSdS(tres, mec.QAA, mec.QAF, mec.QFF, mec.QFA,
            GAF, GFA, expQFF, expQAA, mec.kA, mec.kF)
        uF = np.ones((mec.kF, 1))
        # meanOpenTime = tres + phiA * DARS * QexpQF * uF
        mean = tres + np.dot(phiA, np.dot(np.dot(DARS, QexpQF), uF))
    else:
        phiF = qml.phiHJC(eGFA, eGAF, mec.kF)
        QexpQA = np.dot(mec.QFA, expQAA)
        DFRS = qml.dARSdS(tres, mec.QFF, mec.QFA, mec.QAA, mec.QAF,
            GFA, GAF, expQAA, expQFF, mec.kF, mec.kA)
        uA = np.ones((mec.kA, 1))
        # meanShutTime = tres + phiF * DFRS * QexpQA * uA
        mean = tres + np.dot(phiF, np.dot(np.dot(DFRS, QexpQA), uA))

    return mean[0]

def popen(mec, tres, conc, eff='c'):
    """
    Calculate equilibrium open probability (Popen) and correct for
    unresolved blockages in case of presence of fast pore blocker.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    tres : float
        Time resolution (dead time).
    conc : float
        Concentration.

    Returns
    -------
    Popen : float
        Open probability value at a given concentration.
    """

    mec.set_eff(eff, conc)
    if tres == 0:
        p = qml.pinf(mec.Q)
        Popen = 0
        for i in range(mec.kA):
            Popen = Popen + p[i]
    else:
        hmopen = hjc_mean_time(mec, tres, True)
        hmshut = hjc_mean_time(mec, tres, False)
        Popen = (hmopen / (hmopen + hmshut))
    if mec.fastblk:
        Popen = Popen / (1 + conc / mec.KBlk)
    return Popen

def get_P0(mec, tres, eff='c'):
    """
    Find Popen at concentration = 0.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    P0 : float
        Open probability in absence of effector.
    """

    conc = 0
    P0 = 0
    Popen = popen(mec, 0, conc)
    if Popen < 1e-10:
        P0 = Popen
    else:
        P0 = popen(mec, tres, conc)
    return P0

def get_maxPopen(mec, tres, eff='c'):
    """
    Estimate numerically maximum equilibrium open probability.
    In case Popen curve goes through a maximum, the peak open
    probability is returned.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    maxPopen : float
        Maximum equilibrium open probability.
    conc : float
        Concentration at which Popen curve reaches maximal value.
    """

    decline = get_decline(mec, tres)
    flat = False
    monot = True

    conc = 1e-9    # start at 1 nM
    Poplast = popen(mec, tres, conc)
    fac = np.sqrt(10)
    c1 = 0
    c2 = 0

    niter = 0
    while (not flat and conc < 100 and monot):
        conc = conc * fac
        Popen = popen(mec, tres, conc)
        if decline and (np.fabs(Popen) < 1e-12):
            flat = np.fabs(Poplast) < 1e-12
        else:
            rel = (Popen - Poplast) / Popen
            if niter > 1 and Popen > 1e-5:
                if (rel * rellast) < -1e-10: # goes through min/max
                    monot = False
                    c1 = conc / fac     # conc before max
                    c2 = conc    # conc after max
                flat = ((np.fabs(rel) < 1e-5) and
                    (np.fabs(rellast) < 1e-5))
            if conc < 0.01:    # do not leave before 10 mM ?
                flat = False
            rellast = rel
        Poplast = Popen
        niter += 1

    if not monot:    # find maxPopen and cmax more accurately
        epsc =  c1 / 1000    # accuracy in concentration
        epsy = 0.0001    # accuracy in open probability
        Perr = 2 * epsy
        fac = 1.01
        maxnstep  = int(np.log10(np.fabs(c1 - c2) / epsc) / np.log10(2) + 0.5)
        nstep = 0
        while nstep <= maxnstep and np.fabs(Perr) > 0:
            conc = 0.5 * (c1 + c2)
            conc1 = conc / fac
            P1 = popen(mec, tres, conc)
            conc1 = conc * fac
            P2 = popen(mec, tres, conc)
            Perr = P2 - P1
            if Perr < 0:
                c1 = conc1
            else:
                c2 = conc1

    maxPopen = popen(mec, tres, conc)
    return maxPopen, conc

def get_decline(mec, tres, eff='c'):
    """
    Find whether open probability curve increases or decreases
    with ligand concentration. Popen may decrease if ligand is inhibitor.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    decline : bool
        True if Popen curve dectreases with concentration.
    """

    Popen = popen(mec, tres, 1)    # Popen at 1 M
    P0 = get_P0(mec, tres)    # Popen at 0 M
    decline = (Popen < P0)
    return decline

def get_EC50(mec, tres, eff='c'):
    """
    Estimate numerically the equilibrium EC50 for a specified mechanism.
    If monotonic this is unambiguous. If not monotonic then returned is
    a concentration for 50% of  the peak response to the left of the peak.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    EC50 : float
        Concentration at which Popen is 50% of its maximal value.
    """

    P0 = get_P0(mec, tres)
    maxPopen, cmax = get_maxPopen(mec, tres)

    c1 = 0
    c2 = cmax
    conc = 0
    epsy = 0.001    # accuracy in Popen
    Perr = 2 * epsy
    epsc = 0.1e-9    # accuracy in concentration 0.1 nM
    nstepmax = int(np.log10(np.fabs(c1 - c2) / epsc) / np.log10(2) + 0.5)
    nstep = 0

    while np.fabs(Perr) > epsy and nstep <= nstepmax:
        nstep += 1
        conc = (c1 + c2) / 2
        Popen = popen(mec, tres, conc)
        Popen = np.fabs((Popen - P0) / (maxPopen - P0))
        Perr = Popen - 0.5
        if Perr < 0:
            c1 = conc
        elif Perr > 0:
            c2 = conc
    EC50 = conc

    return EC50

def get_nH(mec, tres, eff='c'):
    """
    Calculate Hill slope, nH, at EC50 of a calculated Popen curve.
    This is Python implementation of DCPROGS HJC_HILL.FOR subroutine.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    nH : float
        Hill slope.
    """

    P0 = get_P0(mec, tres)
    Pmax, cmax = get_maxPopen(mec, tres)
    EC50 = get_EC50(mec, tres)
    decline = get_decline(mec, tres)
    if decline:
        temp = P0
        P0 = Pmax
        Pmax = temp

    # Calculate Popen curve
    n = 64
    dc = (np.log10(EC50 * 1.1) - np.log10(EC50 * 0.9)) / (n - 1)
    c = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        c[i] = (EC50 * 0.9) * pow(10, i * dc)
        y[i] = popen(mec, tres, c[i])

    # Find two points around EC50.
    i50 = 0
    s1 = 0
    s2 = 0
    i = 0
    while i50 ==0 and i < n-1:
        if (c[i] <= EC50) and (c[i+1] >= EC50):
            i50 = i
            y1 = np.log10(np.fabs((y[i] - P0) / (Pmax - y[i])))
            y2 = np.log10(np.fabs((y[i+1] - P0) / (Pmax - y[i+1])))
            s1 = (y2 - y1) / (np.log10(c[i+1]) - np.log10(c[i]))
            y3 = np.log10(np.fabs((y[i+1] - P0) / (Pmax - y[i+1])))
            y4 = np.log10(np.fabs((y[i+2] - P0) / (Pmax - y[i+2])))
            s2 = (y4 - y3) / (np.log10(c[i+2]) - np.log10(c[i+1]))
        i += 1

    # Interpolate linearly for Hill slope at EC50
    b = (s2 - s1) / (c[i50+1] - c[i50])
    nH = s1 + b * (EC50 - c[i50])
    return nH

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
    m = (np.dot(np.dot(np.dot(np.dot(qml.phiBurst(mec), interm1), invQAA),
        interm2), uA)[0])
    return m

def mean_open_time_burst(mec):
    """
    Calculate the mean total open time per burst (Eq. 3.26, CH82).

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    m : float
        The mean total open time per burst.
    """

    uA = np.ones((mec.kA, 1))
    VAA = mec.QAA + np.dot(mec.QAB, mec.GBA)
    m = np.dot(np.dot(qml.phiBurst(mec), -nplin.inv(VAA)), uA)[0]
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
    mu = np.dot(np.dot(qml.phiBurst(mec), interm), uA)[0]
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
    Pr = np.dot(np.dot(qml.phiBurst(mec), interm), qml.endBurst(mec))
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

    expQEEA = qml.expQt(mec.QEE, t)[:mec.kA, :mec.kA]
    f = np.dot(np.dot(np.dot(qml.phiBurst(mec), expQEEA), -mec.QAA),
        qml.endBurst(mec))
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
    expQAA = qml.expQt(mec.QAA, t)
    f = np.dot(np.dot(np.dot(qml.phiO(mec), expQAA), -mec.QAA), uA)
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
    expQFF = qml.expQt(mec.QFF, t)
    f = np.dot(np.dot(np.dot(qml.phiS(mec), expQFF), -mec.QFF), uF)
    return f

def get_ideal_pdf_components(mec, open):
    """
    Calculate time constants and areas for an ideal (no missed events)
    exponential open/shut time probability density function.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    open : bool
        True to calculate mean open time, False to calculate mean shut time.

    Returns
    -------
    taus : ndarray, shape(k, 1)
        Time constants.
    areas : ndarray, shape(k, 1)
    """

    if open:
        areas = np.zeros(mec.kA)
        eigs, A = qml.eigs(-mec.QAA)
        uA = np.ones((mec.kA, 1))
        for i in range(mec.kA):
            areas[i] = (np.dot(np.dot(np.dot(qml.phiO(mec), A[i]),
                (-mec.QAA)), uA) / eigs[i])
    else:
        areas = np.zeros(mec.kF)
        eigs, A = qml.eigs(-mec.QFF)
        uF = np.ones((mec.kF, 1))
        for i in range(mec.kF):
            areas[i] = (np.dot(np.dot(np.dot(qml.phiS(mec), A[i]),
                (-mec.QFF)), uF) / eigs[i])

    taus = 1 / eigs
    return taus, areas

def get_burst_ideal_pdf_components(mec):
    """
    Calculate time constants and areas for an ideal (no missed events)
    exponential burst length probability density function.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    taus : ndarray, shape(k, 1)
        Time constants.
    areas : ndarray, shape(k, 1)
    """

    areas = np.zeros(mec.kE)
    eigs, A = qml.eigs(-mec.QEE)
    for i in range(mec.kE):
        areas[i] = (np.dot(np.dot(np.dot(qml.phiBurst(mec),
            A[i][:mec.kA, :mec.kA]), (-mec.QAA)), qml.endBurst(mec)) / eigs[i])

    taus = 1 / eigs
    return taus, areas

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
        sro = qml.bisection_intervals(sao, sbo, tres,
            mec.QFF, mec.QAA, mec.QAF, mec.QFA, mec.kF, mec.kA)
        roots = np.zeros(mec.kA)
        for i in range(mec.kA):
            roots[i] = qml.bisect(sro[i,0], sro[i,1], tres,
                mec.QFF, mec.QAA, mec.QAF, mec.QFA, mec.kF, mec.kA)
        return roots
    else:
        sro = qml.bisection_intervals(sas, sbs, tres,
            mec.QAA, mec.QFF, mec.QFA, mec.QAF, mec.kA, mec.kF)
        roots = np.zeros(mec.kF)
        for i in range(mec.kF):
            roots[i] = qml.bisect(sro[i,0], sro[i,1], tres,
                mec.QAA, mec.QFF, mec.QFA, mec.QAF, mec.kA, mec.kF)
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
        WA = qml.W(roots[i], tres, Q22, Q11, Q12, Q21, k2, k1)
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

def pdf_exponential(t, tres, roots, areas):
    """
    Calculate exponential probabolity density function.

    Parameters
    ----------
    t : float
        Time.
    tres : float
        Time resolution (dead time).
    roots : array_like, shape (k,)
    areas : array_like, shape (k,).

    Returns
    -------
    f : float
    """

    if t < tres:
        f = 0
    else:
        f = 0
        for j in range(roots.shape[0]):
            ta = -1 / roots[j]
            ar = areas[j]
            f = f + ((ar / ta) * np.exp(-(t - tres) / ta))
    return f

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
        ff0 = qml.f0((t - tres), eigvals, gamma00)
        ff1 = qml.f1((t - 2 * tres), eigvals, gamma10, gamma11)
        f = ff0 - ff1
    else:
        f = pdf_exponential(t, tres, roots, areas)
    return f

def exact_pdf_coef(mec, tres, open):
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
    uF = np.ones((mec.kF,1))
    uA = np.ones((mec.kA, 1))
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)
    phiA = qml.phiHJC(eGAF, eGFA, mec.kA)
    phiF = qml.phiHJC(eGFA, eGAF, mec.kF)

    eigen, A = qml.eigs(-mec.Q)
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
            gama00.append(np.dot(np.dot(phiA, C00[i]), M1)[0])
            gama10.append(np.dot(np.dot(phiA, C10[i]), M1)[0])
            gama11.append(np.dot(np.dot(phiA, C11[i]), M1)[0])
    else:
        M1 = np.dot(np.dot(mec.QFA, expQAA), uA)
        for i in range(k):
            gama00.append(np.dot(np.dot(phiF, C00[i]), M1)[0])
            gama10.append(np.dot(np.dot(phiF, C10[i]), M1)[0])
            gama11.append(np.dot(np.dot(phiF, C11[i]), M1)[0])

    return eigen, gama00, gama10, gama11

def ini_vectors(mec, tres, tcrit, is_chsvec=False):
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

    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    expQFF = qml.expQt(mec.QFF, tres)
    expQAA = qml.expQt(mec.QAA, tres)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)
    uA = np.ones((mec.kA, 1))

    if is_chsvec:
        roots = asymptotic_roots(mec, tres, False)
        HFA = np.zeros((mec.kF, mec.kA))
        XFA = qml.XAF(tres, roots, mec.QFF, mec.QAA, mec.QFA,
            mec.QAF)
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

def HJClik(bursts, mec, tres, tcrit, is_chsvec=False):
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

    # TODO: Here reset rates which reached limit or are negative.
    # TODO: Make new Q from theta.
    # TODO: Errors.

    startB, endB = ini_vectors(mec, tres, tcrit, is_chsvec)
    print 'startB=', startB
    print 'endB=', endB
    Aeigvals, AZ00, AZ10, AZ11 = qml.Zxx(tres, mec.Q, mec.kA, mec.QFF,
        mec.QAF, mec.QFA)
    Aroots = asymptotic_roots(mec, tres, True)
    Axaf = qml.XAF(tres, Aroots, mec.QAA, mec.QFF, mec.QAF, mec.QFA)
    Feigvals, FZ00, FZ10, FZ11 = qml.Zxx(tres, mec.Q, mec.kA, mec.QAA,
        mec.QFA, mec.QAF)
    Froots = asymptotic_roots(mec, tres, False)
    Fxaf = qml.XAF(tres, Froots, mec.QFF, mec.QAA, mec.QFA, mec.QAF)
    print 'Fxaf=', Fxaf

    loglik = 0
    for ind in bursts:
        print '\n', ind, 'burst'
        burst = bursts[ind]
        grouplik = startB
        for i in range(len(burst)):
            t = burst[i] * 0.001
            if i % 2 == 0: # open time
                print '\nOPEN t=', t
                eGAFt = qml.eGAF(t, tres, Aroots, Axaf, Aeigvals, AZ00, AZ10, AZ11)
                print 'eGAFt=', eGAFt
            else: # shut
                print'\n SHUT t=', t
                eGAFt = qml.eGAF(t, tres, Froots, Fxaf, Feigvals, FZ00, FZ10, FZ11)
                print 'eGAFt=', eGAFt
            grouplik = np.dot(grouplik, eGAFt)
            print 'grouplik intermediate=', grouplik
        grouplik = np.dot(grouplik, endB)
        print 'grouplik=', grouplik[0]
        loglik += math.log10(grouplik[0])
        print 'loglik=', loglik

    return loglik

