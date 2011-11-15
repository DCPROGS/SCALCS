# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="remis"
__date__ ="$25-Sep-2011 17:53:25$"

import sys
import matplotlib.pyplot as plt
from scipy.special import erf
import numpy as np
from scipy.optimize import fsolve

from dcpyps import qmatlib as qml
from dcpyps import samples
from dcpyps import rcj_alt
from dcpyps import cjumps


if __name__ == "__main__":

    mec = samples.CH82()
    conc = 100e-9 # 100 nM
    mec.set_eff('c', conc)
    tres = 0.0001 # 100 microsec

    root = fsolve(qml.detW, [-5000, 10],
        args=(tres, mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF))

    print 'root=', root

#    qml.detW(s, tres, mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)