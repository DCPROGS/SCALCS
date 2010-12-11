"""
Ploting utilities for single channel and macroscopic current
calculations.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 23:01:09$"

import numpy as np
import matplotlib.pyplot as plt

import qmatlib as qml
import scalcslib as scl
import qmatrc

def popen_curve(cmin, cmax, mec, tres, text1, text2, eff='c'):
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

    log_start = np.log10(cmin)
    log_end = np.log10(cmax)
    decade_num = int(log_end - log_start)
    if qmatrc.debug: print "number of decades = ", decade_num
    log_int = 0.01    # increase this if want more points per curve
    point_num = int(decade_num / log_int + 1)
    if qmatrc.debug: print "number of points = ", point_num

    c = np.zeros(point_num)
    pe = np.zeros(point_num)
    pi = np.zeros(point_num)
    for i in range(point_num):
        c[i] = pow(10, log_start + log_int * i)
        pe[i] = scl.get_Popen(mec, tres, c[i])
        pi[i] = scl.get_Popen(mec, 0, c[i])

    line1, line2 = plt.semilogx(c, pe, 'b-', c, pi, 'r--')
    plt.ylabel('Popen')
    plt.xlabel('Concentration, M')
    plt.axis([cmin, cmax, 0, 1])

    #plt.text(1e-8, 0.6, text1)
    #plt.text(1e-8, 0.4, text2)
    #plt.figlegend((line1, line2),
    #       ('HJC Popen', 'ideal Popen'),
    #       'upper left')
    plt.title('Apparent and ideal Popen curves')
    #plt.show()

def distr_burst_length(mec, conc):
    """
    Plot the burst length distribution.

    Parameters
    ----------
    mec : instance of type Mechanism
    conc : float
        Concentrtaion.
    """

    point_num = 1000
    tmin = 0.00001
    tmax = 0.100
    dt = (np.log10(tmax) - np.log10(tmin)) / (point_num - 1)

    t = np.zeros(point_num)
    fbst = np.zeros(point_num)
    for i in range(point_num):
        t[i] = tmin * pow(10, (i * dt))
        fbst[i] = np.sqrt(t[i] * qml.distr_burst_length(t[i], mec, conc))
    plt.semilogx(t, fbst, 'b-')
    plt.ylabel('fbst(t)')
    plt.xlabel('burst length, s')
    plt.title('The burst length pdf')
    #plt.show()

def distr_num_burst_openings(n, mec, conc):
    """
    Plot the distribution of number of openings per burst.

    Parameters
    ----------
    n : int
        Number of openings per burst.
    mec : instance of type Mechanism
    conc : float
        Concentrtaion.
    """

    r = np.arange(1, n+1)
    #Pr = qml.distNumOpeningsBurst(r, mec)
    Pr = np.zeros(n)
    for i in range(n):
        Pr[i] = qml.distr_num_burst_openings(r[i], mec, conc)

    plt.plot(r, Pr,'ro')
    plt.ylabel('Pr')
    plt.xlabel('Openings per burst')
    plt.title('Openings per burst')
    plt.axis([0, n+1, 0, 1])
    #plt.show()

def burst_length_versus_conc(mec, cmin, cmax):
    """
    Plot of the mean burst length versus concentration.

    Parameters
    ----------
    mec : instance of type Mechanism
    cmin : float
        Concentration to start.
    cmax : float
        Concentration to stop.
    """

    point_num = 100
    incr = (cmax - cmin)/(point_num - 1)
    c = np.zeros(point_num)
    b = np.zeros(point_num)
    blk = np.zeros(point_num)
    for i in range(point_num):
        c[i] = cmin + incr * i
        b[i] = qml.mean_burst_length(mec, c[i])
        if mec.fastblk:
            blk[i] = b[i] * (1 + c[i] / mec.KBlk)

    if mec.fastblk:
        plt.plot(c, b, 'b-', c, blk, 'g-')
    else:
        plt.plot(c, b, 'b-')
    plt.ylabel('Mean burst length, ms')
    plt.xlabel('Concentration, mM')
    plt.title('Mean burst length')
    #plt.show()

