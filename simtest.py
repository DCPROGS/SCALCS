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

    ktmax = 5000 # Number of intervals to be simulated
    nt = 0 # number of transitions counter
    nbmax = 20 # Maximum number of openings/burst expected
    
    # Initial state
    inst = mec.k -1
    if inst < mec.kA:
        ampl = 5
    else:
        ampl = 0
    t = - amean[inst] * math.log(random.random())
    print '\n\tstate ', inst+1, '\tt (ms)=', t, '\tampl (pA)='
    first = 1
    itint = [t]
    iampl = [ampl]

    while nt < ktmax:



        # Get next state
        u = random.random()
        next = 0
        j = -1
        bot = 0.0

        while not next:
            j += 1
            if j != inst:
                if j > 0:
                    bot = picum[inst, j - 1]
                if u > bot and u <= picum[inst, j]:
                    next = 1 # out of loop

        newst = j
        # Get lifetime for current state
        t = - amean[newst] * math.log(random.random()) # in ms
        print '\ntransition #', nt, '\tstate ', inst+1, '\tt=', t, '\tampl (pA)=', ampl

        if ((newst < mec.kA and inst < mec.kA) or (newst >= mec.kA and inst >= mec.kA)):
            itint[-1] += t

#        elif imposeres and (itint[-1] < tres):
#            itint[-2] += itint[-1]
#            itint.pop()
#            iampl.pop()
#            itint[-1] += t
#            nt -= 1
        else:
            itint.append(t)
            if newst < mec.kA:
                ampl = 5
            else:
                ampl = 0
            iampl.append(ampl)
            nt += 1

        inst = newst

#    for i in range(len(itint)):
#        print '\nint ', i+1, '\tt=', itint[i], 'ms \tampl=', iampl[i]


        # Check if level has changed
        # -if not, accumulate times for equal currents but do not increment kt
        # (for first transition always accumulates -ilast set to initial state)
#	if(icur(is,ichan).eq.ilast) then
#	   tint1(kt)=tint1(kt)+t
    

    dcio.scn_write_simulated(itint, iampl, treso=tres/1000, tresg=tres/1000,
        Emem=-100.0, avamp = ampl, filename='c:/pySIMSCN.SCN')


try:
    cProfile.run('main()')
except KeyboardInterrupt:
    pass