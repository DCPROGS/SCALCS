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

TODO
--------
Check if it works with declining Popen curve.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 20:29:14$"

import numpy as np

import qmatlib as qml
import qmatrc

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
    mec.set_eff(eff, conc)
    P0 = 0
    Popen = qml.popen(mec.Q, mec.kA, 0)
    if Popen < 1e-10:
        P0 = Popen
    else:
        P0 = qml.popen(mec.Q, mec.kA, tres)
    if qmatrc.debug: print 'Popen(0)=', P0
    return P0

def get_maxPopen(mec, tres, fastBlk=False, KBlk=None, eff='c'):
    """
    Estimate numerically maximum equilibrium open probability.
    In case Popen curve goes through a maximum, the peak open
    probability is returned.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).
    fastBlk : bool
        True in presence of short unresolved blockages.
    KBlk : float
        Fast block equilibrium constant in M.

    Returns
    -------
    maxPopen : float
        Maximum equilibrium open probability.
    conc : float
        Concentration at which Popen curve reaches maximal value.
    """

    decline = get_decline(mec, tres, fastBlk, KBlk)
    flat = False
    monot = True

    conc = 1e-9    # start at 1 nM
    Poplast = get_Popen(mec, tres, conc, fastBlk, KBlk)
    fac = np.sqrt(10)
    c1 = 0
    c2 = 0

    niter = 0
    while (not flat and conc < 100 and monot):
        conc = conc * fac
        Popen = get_Popen(mec, tres, conc, fastBlk, KBlk)
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
            P1 = get_Popen(mec, tres, conc, fastBlk, KBlk)
            conc1 = conc * fac
            P2 = get_Popen(mec, tres, conc, fastBlk, KBlk)
            Perr = P2 - P1
            if Perr < 0:
                c1 = conc1
            else:
                c2 = conc1

    maxPopen = get_Popen(mec, tres, conc, fastBlk, KBlk)
    return maxPopen, conc

def get_Popen(mec, tres, conc, fastBlk=False, KBlk=None, eff='c'):
    """
    Calculate equilibrium open probability, Popen, and correct for
    unresolved blockages in case of presence of fast pore blocker.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).
    conc : float
        Concentration.
    fastBlk : bool
        True in presence of short unresolved blockages.
    KBlk : float
        Fast block equilibrium constant in M.

    Returns
    -------
    Popen : float
        Open probability value at a given concentration.
    """

    mec.set_eff(eff, conc)
    Popen = qml.popen(mec.Q, mec.kA, tres)
    if fastBlk:
        Popen = Popen / (1 + conc / KBlk)
    return Popen

def get_decline(mec, tres, fastBlk=False, KBlk=None, eff='c'):
    """
    Find whether open probability curve increases or decreases
    with ligand concentration. Popen may decrease if ligand is inhibitor.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).
    fastBlk : bool
        True in presence of short unresolved blockages.
    KBlk : float
        Fast block equilibrium constant in M.

    Returns
    -------
    decline : bool
        True if Popen curve dectreases with concentration.
    """

    Popen = get_Popen(mec, tres, 1, fastBlk, KBlk)    # Popen at 1 M
    P0 = get_P0(mec, tres)    # Popen at 0 M
    decline = (Popen < P0)
    return decline

def get_EC50(mec, tres, fastBlk=False, KBlk=None, eff='c'):
    """
    Estimate numerically the equilibrium EC50 for a specified mechanism.
    If monotonic this is unambiguous. If not monotonic then returned is
    a concentration for 50% of  the peak response to the left of the peak.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).
    fastBlk : bool
        True in presence of short unresolved blockages.
    KBlk : float
        Fast block equilibrium constant in M.

    Returns
    -------
    EC50 : float
        Concentration at which Popen is 50% of its maximal value.
    """

    P0 = get_P0(mec, tres)
    maxPopen, cmax = get_maxPopen(mec, tres, fastBlk, KBlk)

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
        Popen = get_Popen(mec, tres, conc, fastBlk, KBlk)
        Popen = np.fabs((Popen - P0) / (maxPopen - P0))
        Perr = Popen - 0.5
        if Perr < 0:
            c1 = conc
        elif Perr > 0:
            c2 = conc
    EC50 = conc

    return EC50

def get_nH(mec, tres, fastBlk=False, KB=0, eff='c'):
    """
    Calculate Hill slope, nH, at EC50 of a calculated Popen curve.
    This is Python implementation of DCPROGS HJC_HILL.FOR subroutine.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).
    fastBlk : bool
        True in presence of short unresolved blockages.
    KBlk : float
        Fast block equilibrium constant in M.

    Returns
    -------
    nH : float
        Hill slope.
    """

    P0 = get_P0(mec, tres)
    Pmax, cmax = get_maxPopen(mec, tres, fastBlk, KB)
    EC50 = get_EC50(mec, tres, fastBlk, KB)
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
        y[i] = get_Popen(mec, tres, c[i], fastBlk, KB)

    # Find two point around EC50.
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

def get_Popen_plot(mec, tres, cmin, cmax, fastBlk=False, KBlk=0):
    """
    Calculate Popen curve parameters and data for Popen curve plot.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).
    fastBlk : bool
        True in presence of short unresolved blockages.
    KBlk : float
        Fast block equilibrium constant in M.

    Returns
    -------
    text1 : string
        Contains parameters for Popen curve corrected for missed events.
    text2 : string
        Contains parameters for ideal Popen curve.
    c : ndarray of floats, shape (num of points,)
        Concentration in mikroM.
    pe : ndarray of floats, shape (num of points,)
        Open probability corrected for missed events.
    pi : ndarray of floats, shape (num of points,)
        Ideal open probability.
    """

    # Calculate EC50, nH and maxPopen for ideal Popen curve.
    emaxPopen, econc = get_maxPopen(mec, tres, fastBlk, KBlk)
    eEC50 = get_EC50(mec, tres, fastBlk, KBlk) * 1000000    # in mikroM
    enH = get_nH(mec, tres, fastBlk, KBlk)
    text1 = ('HJC Popen curve:\nmaxPopen = {0:.3f}; '.format(emaxPopen) +
        ' EC50 = {0:.3f} mikroM; '.format(eEC50) + ' nH = {0:.3f}'.format(enH))

    # Calculate EC50, nH and maxPopen for Popen curve
    # corrected for missed events.
    imaxPopen, iconc = get_maxPopen(mec, 0, fastBlk, KBlk)
    iEC50 = get_EC50(mec, 0, fastBlk, KBlk) * 1000000   # in mikroM
    inH = get_nH(mec, 0, fastBlk, KBlk)
    text2 = ('Ideal Popen curve:\nmaxPopen = {0:.3f}; '.format(imaxPopen)+
        ' EC50 = {0:.3f} mikroM; '.format(iEC50) + ' nH = {0:.3f}'.format(inH))

    # Plot ideal and corrected Popen curves.
    #cmin = iEC50 * 0.01
    #cmax = iEC50 * 500
    log_start = np.log10(cmin)
    log_end = np.log10(cmax)
    decade_num = int(log_end - log_start)
    log_int = 0.01    # increase this if want more points per curve
    point_num = int(decade_num / log_int + 1)

    c = np.zeros(point_num)
    pe = np.zeros(point_num)
    pi = np.zeros(point_num)
    for i in range(point_num):
        ctemp = pow(10, log_start + log_int * i)
        pe[i] = get_Popen(mec, tres, ctemp, fastBlk, KBlk)
        pi[i] = get_Popen(mec, 0, ctemp, fastBlk, KBlk)
        c[i] = ctemp * 1000000

    return text1, text2, c, pe, pi

def get_burstlen_pdf(mec, conc, tmin, tmax, fastBlk=False, KBlk=0):
    """
    Calculate the mean burst length and data for burst length distribution.

    Parameters
    ----------
    mec : instance of type Mechanism
    conc : float
        Concentration in M.
    tmin, tmax : floats
        Time range for burst length ditribution.
    fastBlk : bool
        True in presence of short unresolved blockages.
    KBlk : float
        Fast block equilibrium constant in M.

    Returns
    -------
    text1 : string
        Mean burst length.
    t : ndarray of floats, shape (num of points,)
        Time in millisec.
    fbst : ndarray of floats, shape (num of points,)
        Burst length pdf.
    """

    # Calculate mean burst length.
    mec.set_eff('c', conc)
    m = qml.mean_burst_length(mec.Q, mec.kA, mec.kB, mec.kC) * 1000
    text1 = 'Mean burst length = %f millisec' %m

    # Calculate burst length pdf.
    point_num = 1000
    dt = (np.log10(tmax) - np.log10(tmin)) / (point_num - 1)

    t = np.zeros(point_num)
    fbst = np.zeros(point_num)
    for i in range(point_num):
        temp = tmin * pow(10, (i * dt))
        fbst[i] = np.sqrt(temp * qml.pdf_burst_length(temp,
            mec.Q, mec.kA, mec.kB, mec.kC)) * 1000
        t[i] = temp * 1000

    return text1, t, fbst

def get_burstopenings_distr(mec, conc):
    """
    Calculate the mean number of openings per burst and data for the
    distribution of openings per burst.

    Parameters
    ----------
    mec : instance of type Mechanism
    conc : float
        Concentration in M.

    Returns
    -------
    text1 : string
        Mean number of openings per burst.
    r : ndarray of floats, shape (num of points,)
        Number of openings per burst.
    Pr : ndarray of floats, shape (num of points,)
        Fraction of bursts.
    """

    # Calculate mean number of openings per burst.
    mec.set_eff('c', conc)
    mu = qml.mean_num_burst_openings(mec.Q, mec.kA, mec.kB, mec.kC)
    text1 = 'Mean number of openings per burst = %f' %mu

    # Plot distribution of number of openings per burst
    n = 10
    r = np.arange(1, n+1)
    Pr = np.zeros(n)
    for i in range(n):
        Pr[i] = qml.distr_num_burst_openings(r[i],
            mec.Q, mec.kA, mec.kB, mec.kC)

    return text1, r, Pr

def get_burstlen_conc_plot(mec, cmin, cmax):
    """
    Calculate data for the plot of burst length versus concentration.

    Parameters
    ----------
    mec : instance of type Mechanism
    cmin, cmax : float
        Range of concentrations in M.

    Returns
    -------
    c : ndarray of floats, shape (num of points,)
        Concentration in mikroM
    br : ndarray of floats, shape (num of points,)
        Mean burst length in millisec.
    """
    point_num = 100
    incr = (cmax - cmin)/(point_num - 1)
    c = np.zeros(point_num)
    br = np.zeros(point_num)
    for i in range(point_num):
        ctemp = cmin + incr * i
        mec.set_eff('c', ctemp)
        br[i] = qml.mean_burst_length(mec.Q, mec.kA, mec.kB, mec.kC) * 1000
        c[i] = ctemp * 1000000
    return c, br

def get_burstlen_conc_fblk_plot(mec, cmin, cmax, fastBlk, KB):
    """
    Calculate data for the plot of burst length versus concentration.
    Returns burst length in absence and presence of short unresolved blockages.

    Parameters
    ----------
    mec : instance of type Mechanism
    cmin, cmax : float
        Range of concentrations in M.
    fastBlk : bool
        True in presence of short unresolved blockages.
    KBlk : float
        Fast block equilibrium constant in M.

    Returns
    -------
    c : ndarray of floats, shape (num of points,)
        Concentration in mikroM
    br : ndarray of floats, shape (num of points,)
        Mean burst length in millisec.
    """
    point_num = 100
    incr = (cmax - cmin)/(point_num - 1)
    c = np.zeros(point_num)
    br = np.zeros(point_num)
    brblk = np.zeros(point_num)
    for i in range(point_num):
        ctemp = cmin + incr * i
        mec.set_eff('c', ctemp)
        br[i] = qml.mean_burst_length(mec.Q, mec.kA, mec.kB, mec.kC) * 1000
        brblk[i] = br[i] * (1 + ctemp / KB)
        c[i] = ctemp * 1000000
    return c, br, brblk

def get_opentime_pdf(mec, conc, tmin, tmax):
    """
    Calculate the mean open time and data for open time distribution.

    Parameters
    ----------
    mec : instance of type Mechanism
    conc : float
        Concentration in M.
    tmin, tmax : floats
        Time range for burst length ditribution.

    Returns
    -------
    text1 : string
        Mean open time.
    t : ndarray of floats, shape (num of points,)
        Time in millisec.
    fopen : ndarray of floats, shape (num of points,)
        Open time pdf.
    """

    # Calculate mean open time.
    mec.set_eff('c', conc)
    #mopt = qml.mean_open_time(mec.Q, mec.kA, mec.kB, mec.kC) * 1000
    #text1 = 'Mean open time = %f millisec' %mopt
    # Calculate open time pdf.
    point_num = 1000
    dt = (np.log10(tmax) - np.log10(tmin)) / (point_num - 1)
    t = np.zeros(point_num)
    fopt = np.zeros(point_num)
    for i in range(point_num):
        temp = tmin * pow(10, (i * dt))
        fopt[i] = np.sqrt(temp * qml.pdf_open_time(temp,
            mec.Q, mec.kA)) * 1000
        t[i] = temp * 1000
    #return text1, t, fopt
    return t, fopt

def get_shuttime_pdf(mec, conc, tmin, tmax):
    """
    Calculate the mean shut time and data for shut time distribution.

    Parameters
    ----------
    mec : instance of type Mechanism
    conc : float
        Concentration in M.
    tmin, tmax : floats
        Time range for burst length ditribution.

    Returns
    -------
    text1 : string
        Mean shut time.
    t : ndarray of floats, shape (num of points,)
        Time in millisec.
    fbst : ndarray of floats, shape (num of points,)
        Shut time pdf.
    """

    # Calculate mean shut time.
    mec.set_eff('c', conc)
    #msht = qml.mean_shut_time(mec.Q, mec.kA, mec.kB, mec.kC) * 1000
    #text1 = 'Mean shut time = %f millisec' %msht
    # Calculate shut time pdf.
    point_num = 1000
    dt = (np.log10(tmax) - np.log10(tmin)) / (point_num - 1)
    t = np.zeros(point_num)
    fsht = np.zeros(point_num)
    for i in range(point_num):
        temp = tmin * pow(10, (i * dt))
        fsht[i] = np.sqrt(temp * qml.pdf_shut_time(temp,
            mec.Q, mec.kA, mec.kB, mec.kC)) * 1000
        t[i] = temp * 1000
    #return text1, t, fsht
    return t, fsht

