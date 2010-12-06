#! /usr/bin/python
"""
calc_Popen module is a collection of functions used for calculations of
open probality (Popen) curves.
"""
__author__="R.Lape, University College London"
__date__ ="$11-Oct-2010 10:38:34$"

from math import*
import numpy as np
import qmatlib as qml
from mechanism import Mechanism
import matplotlib.pyplot as plt


def get_maxPopen(func, mec, tres, decline, debug=False):
    """
    Find maximum value of a Popen curve.
    TODO: doesn't work for not monotonic curve.
    TODO: doesn't correct Poepen in case of very fast block.
    Parameters
    ----------
    func : function
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
        Popen = func(mec.Q, mec.kA, tres, debug)

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
    maxPopen = func(mec.Q, mec.kA, tres, debug)
    return maxPopen, xA

def get_EC50(func, mec, P0, maxPopen, tres, debug=False):
    """
    Find EC50 value of a Popen curve.
    If monotonic this is unambiguous. If not monotonic then EC50 is
    returned as conc to left of peak for 50% of peak response.

    Parameters
    ----------
    func : function
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
            Popen = func(mec.Q, mec.kA, tres, debug)
            pout = fabs((Popen - P0) / (maxPopen - P0))
            Perr = pout - 0.5    # Yout 0.5 for EC50
            if Perr < 0:
                x1 = xout
            elif Perr > 0:
                x2 = xout
    EC50 = xout

    return EC50

def get_nH(x, y, decline, EC50, Pmax, P0, debug=False):
    """
    Calculate Hill slope, nH, at EC50 of a calculated Popen curve.
    This is Python implementation of DCPROGS HJC_HILL.FOR subroutine.

    Parameters
    ----------
    x : array_like, shape (n)
        Concentration.
    y : array_like, shape (n)
        Popen values.
    n : int
        Number of values in x or y
    decline : logical
        True if Popen curve monotonicly declining.
    EC50 : float
        Concentration at which open probability is 50% of its maximal value.
    maxPopen : float
        Maximal open probability value.
    P0 : float
        Minimal open probability value.

    Returns
    -------
    nH : float
        Concentration at which open probability is 50% of its maximal value.
    """

    if decline:
        temp = P0
        P0 = Pmax
        Pmax = temp

    i50 = 0
    s1 = 0
    s2 = 0
    i = 0
    n = np.size(x)

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

def get_Popen_curve_param(cmin, cmax, mec, tres, fastblk=False, debug=False):
    """
    Estimate numerically the equilibrium EC50 and maximum equilibrium
    Popen for a specified mechanism.
    This is Python implementation of DCPROGS EC50_HJ2.FOR subroutine.

    Parameters
    ----------
    cmin : float
        Concentration to start.
    cmax : float
        Concentration to stop.
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    eEC50 : float
    emaxPopen : float
    enH : float
    iEC50 : float
    imaxPopen : float
    inH : float
    """

    # Find Popen at concentration = 0
    conc = 0
    mec.init_Q(conc)
    P0 = 0
    Popen = qml.popen(mec.Q, mec.kA, 0, debug)
    if Popen < 1e-10:
        P0 = Popen
    else:
        P0 = qml.popen(mec.Q, mec.kA, tres, debug)
    if debug: print 'Popen(0)=', P0

    # Find whether Popen increases or decreases with ligand
    # concentration. Popen may decrease if ligand is inhibitor.
    # First find Popen at a single high conc (say 1 M)
    conc1 = 1     # concentration 1 M
    mec.init_Q(conc1)
    Popen = qml.popen(mec.Q, mec.kA, tres, debug)
    if fastblk:    # correct Popen for unresolved block
        #x1 = 1 + conc / KB
        #Popen = Popen / x1
        pass
    decline = (Popen < P0)

    emaxPopen, xA = get_maxPopen(qml.popen, mec, tres, decline)
    imaxPopen, xA = get_maxPopen(qml.popen, mec, 0, decline)
    if debug: print 'HJC maxPopen={0:.4f}'.format(emaxPopen),'at conc=',xA,'M'
    if debug: print 'ideal maxPopen={0:.4f}'.format(imaxPopen),'at conc=',xA,'M'
    eEC50 = get_EC50(qml.popen, mec, P0, emaxPopen, tres)
    iEC50 = get_EC50(qml.popen, mec, P0, imaxPopen, 0)
    if debug: print 'HJC EC50 = {0:.3f} mikroM'.format(eEC50*1000000)
    if debug: print 'ideal EC50 = {0:.3f} mikroM'.format(iEC50*1000000)

    # Calculate Popen curve
    n = 512
    dx = (log10(cmax) - log10(cmin))/(n-1)
    xcal = np.zeros((1,n))
    eycal = np.zeros((1,n))
    iycal = np.zeros((1,n))
    for i in range(n):
        xcal[0,i] = cmin * pow(10, i*dx)
        mec.init_Q(xcal[0,i])
        eycal[0,i] = qml.popen(mec.Q, mec.kA, tres, debug)
        iycal[0,i] = qml.popen(mec.Q, mec.kA, 0, debug)

    enH = get_nH(xcal, eycal, decline, eEC50, emaxPopen, P0)
    inH = get_nH(xcal, iycal, decline, iEC50, imaxPopen, P0)
    if debug: print 'Hill coeficient of HJC curve =', enH
    if debug: print 'Hill coeficient of ideal curve =', inH

    return eEC50, emaxPopen, enH, iEC50, imaxPopen, inH

def plot_Popen_curve(cmin, cmax, mec, tres, debug=False):
    """
    Plot open probability, Popen, curve.

    Parameters
    ----------
    cmin : float
        Concentration to start.
    cmax : float
        Concentration to stop.
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).
    """

    log_start = log10(cmin)
    log_end = log10(cmax)
    decade_numb = int(log_end -log_start)
    if debug: print "number of decades = ", decade_numb
    log_int = 0.01    # increase this if want more points per curve
    point_num = int(decade_numb / log_int + 1)
    if debug: print "number of points = ", point_num

    conc = np.zeros((2, point_num))
    p = np.zeros((2, point_num))
    for i in range(point_num):
        conc[0,i] = pow(10, log_start + log_int*i)
        mec.init_Q(conc[0,i])
        p[0,i] = qml.popen(mec.Q, mec.kA, tres, debug)
        p[1,i] = qml.popen(mec.Q, mec.kA, 0, debug)

    line1, line2 = plt.semilogx(conc[0],p[0],'b-',conc[0],p[1],'r--')
    plt.ylabel('Popen')
    plt.xlabel('Concentration, M')
    plt.axis([cmin, cmax, 0, 1])
    
    eEC50, emaxPopen, enH, iEC50, imaxPopen, inH = get_Popen_curve_param(cmin, cmax, mec, tres, debug)
    text1 = 'ideal Popen curve:\nmaxPopen={0:.3f}'.format(imaxPopen) + '\nEC50 = {0:.2e} M'.format(iEC50) + '\nnH = {0:.3f}'.format(inH)
    plt.text(1e-8, 0.6, text1)
    text2 = 'HJC Popen curve:\nmaxPopen={0:.3f}'.format(emaxPopen) + '\nEC50 = {0:.2e} M'.format(eEC50) + '\nnH = {0:.3f}'.format(enH)
    plt.text(1e-8, 0.4, text2)
    plt.figlegend( (line1, line2),
           ('HJC Popen', 'ideal Popen'),
           'upper left' )
    plt.title('Apparent and ideal Popen curves')
    plt.show()

if __name__ == "__main__":

    debug = False
    tres = 0.00004  # resolution in seconds
    c_start = 1.e-9      # 1 nM in M
    c_end = 1.e-3        # 1 mikroM in M

    mec = Mechanism()
    mec.demoQ()
    #get_Popen_curve_param(c_start, c_end, mec, dict, tres, debug)
    plot_Popen_curve(c_start, c_end, mec, tres, debug)