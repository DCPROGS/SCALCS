"""A collection of functions for open probability, Popen, or dose-response
curve calculations.
"""

import math
import numpy as np

from scalcs import qmatlib as qml
from scalcs import scalcslib as scl

def Popen(mec, tres, conc=0, eff='c'):
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
        p = qml.pinf(mec.QGG)
        popen = np.sum(p[:mec.kA]) / np.sum(p)
    else:
        hmopen, hmshut = scl.exact_mean_open_shut_time(mec, tres)
        popen = (hmopen / (hmopen + hmshut))
    if mec.fastblock:
        popen = popen / (1 + conc / mec.fastKB)
    return popen

def Popen0(mec, tres, eff='c'):
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
    popen = Popen(mec, 0, conc=0) 
    return popen if popen < 1e-10 else Popen(mec, tres, conc=0)

def maxPopen(mec, tres, eff='c'):
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

    flat = False
    monot = True

    conc = 1e-9    # start at 1 nM
    poplast = Popen(mec, tres, conc)
    fac = math.sqrt(10)
    c1, c2 = 0, 0

    niter = 0
    while (not flat and conc < 100 and monot):
        conc = conc * fac
        popen = Popen(mec, tres, conc)
        if decline(mec, tres) and (math.fabs(popen) < 1e-12):
            flat = math.fabs(poplast) < 1e-12
        else:
            rel = (popen - poplast) / popen
            if niter > 1 and popen > 1e-5:
                if (rel * rellast) < -1e-10: # goes through min/max
                    monot = False
                    c1 = conc / fac     # conc before max
                    c2 = conc    # conc after max
                flat = ((math.fabs(rel) < 1e-5) and
                    (math.fabs(rellast) < 1e-5))
            if conc < 0.01:    # do not leave before 10 mM ?
                flat = False
            rellast = rel
        poplast = popen
        niter += 1

    if not monot:    # find maxPopen and cmax more accurately
        epsc =  c1 / 1000    # accuracy in concentration
        epsy = 0.0001    # accuracy in open probability
        perr = 2 * epsy
        fac = 1.01
        maxnstep  = int(math.log10(math.fabs(c1 - c2) / epsc) / math.log10(2) + 0.5)
        nstep = 0
        while nstep <= maxnstep and math.fabs(perr) > 0:
            conc = 0.5 * (c1 + c2)
            conc1 = conc / fac
            P1 = Popen(mec, tres, conc)
            conc1 = conc * fac
            P2 = Popen(mec, tres, conc)
            perr = P2 - P1
            if perr < 0:
                c1 = conc1
            else:
                c2 = conc1

    maxPopen = Popen(mec, tres, conc)
    return maxPopen, conc

def decline(mec, tres, eff='c'):
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
    return (Popen(mec, tres, conc=1) < Popen0(mec, tres))

def EC50(mec, tres, eff='c'):
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

    P0 = Popen0(mec, tres)
    maxP, c2 = maxPopen(mec, tres)
    c1 = 0
    epsy = 0.001    # accuracy in Popen
    perr = 2 * epsy
    epsc = 0.1e-9    # accuracy in concentration 0.1 nM
    nstepmax = int(math.log10(math.fabs(c1 - c2) / epsc) / math.log10(2) + 0.5)
    nstep = 0
    while math.fabs(perr) > epsy and nstep <= nstepmax:
        conc = (c1 + c2) / 2
        perr = math.fabs((Popen(mec, tres, conc) - P0) / (maxP - P0)) - 0.5
        if perr < 0:
            c1 = conc
        elif perr > 0:
            c2 = conc
        nstep += 1
    return conc

def nH(mec, tres, eff='c'):
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

    P0 = Popen0(mec, tres)
    Pmax, cmax = maxPopen(mec, tres)
    if decline(mec, tres):
        P0, Pmax = Pmax, P0
    ec50 = EC50(mec, tres)
    # Calculate Popen curve
    n = 64
    dc = (math.log10(ec50 * 1.1) - math.log10(ec50 * 0.9)) / (n - 1)
    c = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        c[i] = (ec50 * 0.9) * pow(10, i * dc)
        y[i] = Popen(mec, tres, c[i])

    # Find two points around EC50.
    i50 = 0
    s1, s2 = 0, 0
    i = 0
    while i50 ==0 and i < n-1:
        if (c[i] <= ec50) and (c[i+1] >= ec50):
            i50 = i
            y1 = math.log10(math.fabs((y[i] - P0) / (Pmax - y[i])))
            y2 = math.log10(math.fabs((y[i+1] - P0) / (Pmax - y[i+1])))
            s1 = (y2 - y1) / (math.log10(c[i+1]) - math.log10(c[i]))
            y3 = math.log10(math.fabs((y[i+1] - P0) / (Pmax - y[i+1])))
            y4 = math.log10(math.fabs((y[i+2] - P0) / (Pmax - y[i+2])))
            s2 = (y4 - y3) / (math.log10(c[i+2]) - math.log10(c[i+1]))
        i += 1

    # Interpolate linearly for Hill slope at EC50
    b = (s2 - s1) / (c[i50+1] - c[i50])
    nH = s1 + b * (ec50 - c[i50])
    return nH


def printout(mec, tres):
    """
    """
    out = ('\n*******************************************\nPopen CURVE\n' )
    if mec.fastblock:
        out += ('\nThis Popen curve was corrected for fast block ' + 
            'with KB = {0:.5g} mM.'.format(mec.fastKB * 1000))
    out += ('\nHJC Popen curve:\n' + print_pars(mec, tres))
    out += ('\nIdeal Popen curve:\n' + print_pars(mec, 0))
    return out

def print_pars(mec, tres):
    emaxPopen, conc = maxPopen(mec, tres)
    return ('maxPopen = {0:.5g}; '.format(emaxPopen) + 
           ' EC50 = {0:.5g} microM; '.format(EC50(mec, tres) * 1000000) + 
           ' nH = {0:.5g}'.format(nH(mec, tres)))