"""
Ploting utilities for single channel and macroscopic current
calculations.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 23:01:09$"

import numpy as np
import qmatlib as qml
import matplotlib.pyplot as plt

def plot_Popen_curve(cmin, cmax, mec, tres, text1, text2, debug=False):
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
    decade_numb = int(log_end -log_start)
    if debug: print "number of decades = ", decade_numb
    log_int = 0.01    # increase this if want more points per curve
    point_num = int(decade_numb / log_int + 1)
    if debug: print "number of points = ", point_num

    conc = np.zeros((2, point_num))
    p = np.zeros((2, point_num))
    for i in range(point_num):
        conc[0,i] = pow(10, log_start + log_int*i)
        mec.c = conc[0,i]
        p[0,i] = qml.popen(mec.Q, mec.kA, tres, debug)
        p[1,i] = qml.popen(mec.Q, mec.kA, 0, debug)

    line1, line2 = plt.semilogx(conc[0],p[0],'b-',conc[0],p[1],'r--')
    plt.ylabel('Popen')
    plt.xlabel('Concentration, M')
    plt.axis([cmin, cmax, 0, 1])

    plt.text(1e-8, 0.6, text1)
    plt.text(1e-8, 0.4, text2)
    plt.figlegend( (line1, line2),
           ('HJC Popen', 'ideal Popen'),
           'upper left' )
    plt.title('Apparent and ideal Popen curves')
    plt.show()



