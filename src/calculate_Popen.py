#! /usr/bin/python
"""
This script uses scalcslib module to calculate maxPopen, EC50 and
nH parameters.
"""

import numpy as np

import scalcslib as scl
import scplotlib as scpl
import mechanism as mec

def demoQ():

    RateList = [
         mec.Rate(15.0, 4, 1), # beta_1
         mec.Rate(15000.0, 3, 2), # beta_2
         mec.Rate(3000.0, 1, 4), # alpha_1
         mec.Rate(500.0, 2, 3), # alpha_2
         mec.Rate(2000.0, 4, 5), # k_m1
         mec.Rate(2 * 2000.0, 3, 4), # k_m2
         mec.Rate(2 * 5.0e07, 5, 4, 'c'), # k_p1
         mec.Rate(5.0e08, 1, 2, 'c'), # k_star_p2
         mec.Rate(5.0e08, 4, 3, 'c'), # k_p2
         mec.Rate(2 * 1.0 / 3.0, 2, 1), # k_star_m2
         ]

    StateList = [
        mec.State(1, 'A'),
        mec.State(2, 'A'),
        mec.State(3, 'B'),
        mec.State(4, 'B'),
        mec.State(5, 'C')
        ]

    ncyc = 1
    dgamma = 50e-12
    fastblk = True
    KBlk = 0.001

    return  mec.Mechanism(RateList, StateList, ncyc, dgamma, fastblk, KBlk)

if __name__ == "__main__":

    debug = False
    tres = 0.00004  # resolution in seconds
    c_start = 1.e-9      # 1 nM in M
    c_end = 1.e-3        # 1 mikroM in M

    demomec = demoQ()

    emaxPopen = scl.get_maxPopen(demomec, tres)
    imaxPopen = scl.get_maxPopen(demomec, 0)
    eEC50 = scl.get_EC50(demomec, tres)
    iEC50 = scl.get_EC50(demomec, 0)
    enH = scl.get_nH(demomec, tres)
    inH = scl.get_nH(demomec, 0)

    text1 = ('ideal Popen curve:\nmaxPopen={0:.3f}'.format(imaxPopen)) # +
        #'\nEC50 = {0:.2e} M'.format(iEC50) + '\nnH = {0:.3f}'.format(inH))
    text2 = ('HJC Popen curve:\nmaxPopen={0:.3f}'.format(emaxPopen)) # +
        #'\nEC50 = {0:.2e} M'.format(eEC50) + '\nnH = {0:.3f}'.format(enH))

    scpl.plot_Popen_curve(c_start, c_end, demomec, tres, text1, text2)
