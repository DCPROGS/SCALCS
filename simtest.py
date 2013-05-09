#! /usr/bin/python
"""
Maximum likelihood fit demo.
"""

import sys
import time
import math
import random
import numpy as np
import cProfile


from dcpyps import samples
from dcpyps import dcio
from dcpyps import dataset


def transition_probability(Q):
    """
    """
    k = Q.shape[0]
    pi = Q.copy()
    for i in range(k):
        pi[i] = pi[i] / -Q[i,i]
        pi[i,i] = 0.0
    return pi

def main():

    t = time.asctime()
    print t
    expdate = t[8:10] + '-' + t[4:7] + '-' + t[20:24] #  '00-ooo-0000'    #character*11
    print 'Experiment date= ', expdate

    print('\n\nTesting single channel simmulation...')
    # LOAD DEMO MECHANISM (C&H82 numerical example).
    mec = samples.CH82()
    mec.printout(sys.stdout)
    tres = 0.00002 * 1000 # in ms
    imposeres = 1
    tcrit = 0.004 * 1000 # in ms
    conc = 100e-9

    mec.set_eff('c', conc)

    pi = transition_probability(mec.Q)
    print '\n\npi=\n', pi
    picum = np.cumsum(pi,axis=1)
    print '\npicum=\n', picum
    # calculate mean lifetimes
    amean = -1000 / mec.Q.diagonal() # in ms
    print '\namean=\n', amean

    ktmax = 500 # Number of intervals to be simulated

    rec = dataset.TimeSeries()
    rec.simulate_record(mec, tres, conc, 5, ktmax)

    dcio.scn_write_simulated(rec.itint, rec.iampl, treso=tres/1000, tresg=tres/1000,
        Emem=-100.0, avamp = ampl, filename='c:/pySIMSCN.SCN')

try:
    cProfile.run('main()')
except KeyboardInterrupt:
    pass