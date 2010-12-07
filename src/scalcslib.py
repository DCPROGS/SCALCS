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
The principles of the stochastic interpretation of ion channel mechanisms.
In: Single-channel recording. 2nd ed. (Eds: Sakmann B, Neher E)
Plenum Press, New York, pp. 397-482.

CH95b: Colquhoun D, Hawkes AG (1995b)
A Q-Matrix Cookbook.
In: Single-channel recording. 2nd ed. (Eds: Sakmann B, Neher E)
Plenum Press, New York, pp. 589-633.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 20:29:14$"

from math import*
import numpy as np
import qmatlib as qml

def get_P0(mec, tres, debug=False):
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
    """

    conc = 0
    mec.init_Q(conc)
    P0 = 0
    Popen = qml.popen(mec.Q, mec.kA, 0, debug)
    if Popen < 1e-10:
        P0 = Popen
    else:
        P0 = qml.popen(mec.Q, mec.kA, tres, debug)
    if debug: print 'Popen(0)=', P0
    return P0

def get_maxPopen(mec, tres, debug=False):
    """
    Find maximum value of a Popen curve.
    TODO: doesn't work for not monotonic curve.
    TODO: doesn't correct Poepen in case of very fast block.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).
    decline : logical
        True if Popen curve monotonicly declining.

    Returns
    -------
    maxPopen : float
    xA : float
        Concentration at which Popen curve reaches maximal value.
    """

    decline = get_decline(mec, tres)

    flat = False
    xA = 1e-9    # start at 1 nM
    ncyc = 1
    poplast = 0
    monot = True
    xA1 = 0
    xA2 = 0
    fac = sqrt(10)

    while (not flat and xA < 100):
        mec.init_Q(xA)
        Popen = qml.popen(mec.Q, mec.kA, tres, debug)

        if ncyc > 1:
            if decline and (fabs(Popen) < 1e-12):
                flat = True
            else:
                rel = (Popen - poplast) / Popen
                if ncyc > 2 and Popen > 1e-5:
                    # rellast not defined until ncyc = 2
                    if (rel * rellast) < -1e-10:
                        #goes through min/max
                        monot = False
                        xA1 = xA / fac     # conc before max
                        xA2 = xA    # conc after max
                    #Consider as convergence when Popen at two
                    #successive concentrations < 0.0001
                    flat = (fabs(rel)<1e-4) and (fabs(rellast) < 1e-4)

            if xA < 0.01:
                flat = False
                rellast = rel

        poplast = Popen
        ncyc += 1
        xA = xA * fac
    #end while

    mec.init_Q(xA)
    maxPopen = qml.popen(mec.Q, mec.kA, tres, debug)
    return maxPopen

def get_decline(mec, tres, debug=False):
    """
    Find whether open probability curve increases or decreases
    with ligand concentration. Popen may decrease if ligand is inhibitor.
    """

    # First find Popen at a single high conc (say 1 M)
    conc1 = 1     # concentration 1 M
    mec.init_Q(conc1)
    Popen = qml.popen(mec.Q, mec.kA, tres, debug)

    P0 = get_P0(mec, tres)
    if mec.fastblk:    # correct Popen for unresolved block
        #x1 = 1 + conc / KB
        #Popen = Popen / x1
        pass
    decline = (Popen < P0)
    return decline


def get_EC50(mec, tres, debug=False):
    """
    Find EC50 value of a Popen curve.
    If monotonic this is unambiguous. If not monotonic then EC50 is
    returned as conc to left of peak for 50% of peak response.

    Parameters
    ----------
    mec : instance of type Mechanism
    P0 : float
        Minimal open probability value.
    maxPopen : float
        Maximal open probability value.
    tres : float
        Time resolution (dead time).

    Returns
    -------
    EC50 : float
        Concentration at which open probability is 50% of its maximal value.
    """

    P0 = get_P0(mec, tres)
    maxPopen = get_maxPopen(mec, tres)

    epsy = 0.001    # accuracy in cur/curmax = 0.5
    Perr = 2 * epsy    # to start
    x1 = 0
    x2 = 100
    xout = 0

    epsx = 0.1e-9    # accuracy = 0.1 nM
    nstepmax = int(log10(fabs(x1-x2)/epsx) / log10(2) + 0.5)
    nstep = 0
    while fabs(Perr) > epsy:
        nstep += 1
        if nstep <= nstepmax:
            xout = 0.5 * (x1 + x2)
            xA = xout
            mec.init_Q(xA)
            Popen = qml.popen(mec.Q, mec.kA, tres, debug)
            pout = fabs((Popen - P0) / (maxPopen - P0))
            Perr = pout - 0.5    # Yout 0.5 for EC50
            if Perr < 0:
                x1 = xout
            elif Perr > 0:
                x2 = xout
    EC50 = xout

    return EC50

def get_nH(mec, tres, debug=False):
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
        Concentration at which open probability is 50% of its maximal value.
    """

    P0 = get_P0(mec, tres)
    Pmax = get_maxPopen(mec, tres)
    EC50 = get_EC50(mec, tres)
    decline = get_decline(mec, tres)

    # Calculate Popen curve
    n = 512
    dx = (log10(EC50*10) - log10(EC50*0.1))/(n-1)
    x = np.zeros((1,n))
    y = np.zeros((1,n))
    for i in range(n):
        x[0,i] = (EC50*0.1) * pow(10, i*dx)
        mec.init_Q(x[0,i])
        y[0,i] = qml.popen(mec.Q, mec.kA, tres, debug)

    if decline:
        temp = P0
        P0 = Pmax
        Pmax = temp

    i50 = 0
    s1 = 0
    s2 = 0
    i = 0

    while i50 ==0 and i < n-1:
        if (x[0,i] <= EC50) and (x[0,i+1] >= EC50):
            i50 = i
            y1 = log10(fabs((y[0,i]-P0)/(Pmax-y[0,i])))
            y2 = log10(fabs((y[0,i+1]-P0)/(Pmax-y[0,i+1])))
            s1 = (y2 - y1) / (log10(x[0,i+1]) - log10(x[0,i]))
            y3 = log10(fabs((y[0,i+1]-P0)/(Pmax-y[0,i+1])))
            y4 = log10(fabs((y[0,i+2]-P0)/(Pmax-y[0,i+2])))
            s2 = (y4 - y3) / (log10(x[0,i+2]) - log10(x[0,i+1]))
        i += 1

    b = (s2 - s1) / (x[0,i50+1] - x[0,i50])
    nH = s1 + b * (EC50 - x[0,i50])

    return nH
