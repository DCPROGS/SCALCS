"""
Ploting utilities for single channel and macroscopic current
calculations.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 23:01:09$"

import numpy as np
import matplotlib.pyplot as plt

import scalcslib as scl
import qmatrc

def plot_Popen_curve(cmin, cmax, mec, tres, text1, text2, eff='c'):
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

    plt.text(1e-8, 0.6, text1)
    plt.text(1e-8, 0.4, text2)
    plt.figlegend((line1, line2),
           ('HJC Popen', 'ideal Popen'),
           'upper left')
    plt.title('Apparent and ideal Popen curves')
    plt.show()
