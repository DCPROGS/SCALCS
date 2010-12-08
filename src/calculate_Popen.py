#! /usr/bin/python
"""
This script uses scalcslib module to calculate maxPopen, EC50 and nH parameters.
"""
__author__="R.Lape, University College London"
__date__ ="$11-Oct-2010 10:38:34$"

import numpy as np

import scalcslib as scl
import scplotlib as scpl
import mechanism as mec

def demoQ():

    beta_1 = 15.0
    beta_2 = 15000.0
    alpha_1 = 3000.0
    alpha_2 = 500.0
    k_m1 = 2000.0
    k_m2 = 2 * 2000.0
    k_p1 = 2 * 5.0e07
    k_star_p2 = 5.0e08
    k_p2 = 5.0e08
    k_star_m2 = 2 * 1.0 / 3.0
    
    Q0 = np.array([[         0,  k_star_p2,        0,   alpha_1,     0],
                   [ k_star_m2,          0,  alpha_2,         0,     0],
                   [         0,     beta_2,        0,      k_m2,     0],
                   [    beta_1,          0,     k_p2,         0,  k_m1],
                   [         0,          0,        0,      k_p1,     0]],
                  dtype = np.float64)

    kA = 2
    kB = 2
    kC = 1
    kD = 0
    cdr = np.array([[1,2,1],[4,3,1],[5,4,1]])
    ncyc = 1
    dgamma = 50e-12
    fastblk = False

    return  mec.Mechanism(Q0, kA, kB, kC, kD, cdr, ncyc, dgamma,
                          fastblk)

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

    text1 = ('ideal Popen curve:\nmaxPopen={0:.3f}'.format(imaxPopen)
        + '\nEC50 = {0:.2e} M'.format(iEC50) + '\nnH = {0:.3f}'.format(inH))
    text2 = ('HJC Popen curve:\nmaxPopen={0:.3f}'.format(emaxPopen)
        + '\nEC50 = {0:.2e} M'.format(eEC50) + '\nnH = {0:.3f}'.format(enH))

    scpl.plot_Popen_curve(c_start, c_end, demomec, tres, text1, text2)
