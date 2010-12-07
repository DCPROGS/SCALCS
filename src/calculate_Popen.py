#! /usr/bin/python
"""
This script uses scalcslib module to calculate maxPopen, EC50 and nH parameters.
"""
__author__="R.Lape, University College London"
__date__ ="$11-Oct-2010 10:38:34$"

from math import*
import scalcslib as scl
import scplotlib as scpl
from mechanism import Mechanism

if __name__ == "__main__":

    debug = False
    tres = 0.00004  # resolution in seconds
    c_start = 1.e-9      # 1 nM in M
    c_end = 1.e-3        # 1 mikroM in M

    mec = Mechanism()
    mec.demoQ()

    emaxPopen = scl.get_maxPopen(mec, tres)
    imaxPopen = scl.get_maxPopen(mec, 0)
    eEC50 = scl.get_EC50(mec, tres)
    iEC50 = scl.get_EC50(mec, 0)
    enH = scl.get_nH(mec, tres)
    inH = scl.get_nH(mec, 0)

    text1 = ('ideal Popen curve:\nmaxPopen={0:.3f}'.format(imaxPopen)
        + '\nEC50 = {0:.2e} M'.format(iEC50) + '\nnH = {0:.3f}'.format(inH))
    text2 = ('HJC Popen curve:\nmaxPopen={0:.3f}'.format(emaxPopen)
        + '\nEC50 = {0:.2e} M'.format(eEC50) + '\nnH = {0:.3f}'.format(enH))

    scpl.plot_Popen_curve(c_start, c_end, mec, tres, text1, text2)
