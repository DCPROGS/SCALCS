# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="remis"
__date__ ="$25-Sep-2011 17:53:25$"

import sys
import matplotlib.pyplot as plt
from scipy.special import erf
import numpy as np

from dcpyps import qmatlib as qml
from dcpyps import samples
from dcpyps import rcj_alt
from dcpyps import cjumps


if __name__ == "__main__":

    reclen = 0.05
    step = 0.000005
    cmax = 0.002
    cb = 0.0
    
    profile = 'rcj'

    if profile == 'rcj':
        centre = 0.01
        width = 0.01
        rise = 0.00025
        decay = 0.00020
        cargs = (cmax, centre, width, rise, decay, cb)
        cfunc = cjumps.pulse_erf
    elif profile == 'instexp':
        prepulse = 0.01
        tdec = 0.0025
        cargs = (prepulse, cmax, tdec, cb)
        cfunc = cjumps.pulse_instexp
    elif profile == 'square':
        prepulse = 0.01
        pulse = 0.01
        cargs = (prepulse, pulse, cmax, cb)
        cfunc = cjumps.pulse_square

#    t = np.arange(0, reclen, step)
#    c = cjumps.pulse_erf(t, (cmax, centre, width, rise, decay, cb))
    mec = samples.CH82()

    t, c, P, Popen = cjumps.solve_jump(mec, reclen, step, cb, cfunc, cargs)

    plt.plot(t, c, 'k-', t, Popen, 'r-')
    plt.ylabel('Concentration, mM')
    plt.xlabel('Time, ms')
    plt.title('Concentration jump')
    plt.show()
