#! /usr/bin/python
"""
Maximum likelihood fit demo.
"""

import sys
import time
import math
import random
import numpy as np

from dcpyps import samples
from dcpyps import dcio
from dcpyps import scalcslib as scl

def next_state(present, picum, tmean, kA, opamp):
    """
    Get next state
    """
    u = random.random()
    found = 0
    next = -1
    bot = 0.0
    a = 0
    while not found:
        next += 1
        if next != present:
            if next > 0:
                bot = picum[present, next - 1]
            if u > bot and u <= picum[present, next]:
                found = 1 # out of loop
    t = - tmean[next] * math.log(random.random()) # in ms
    if next < kA:
        a = opamp
    return next, t, a

if __name__ == "__main__":

    t = time.asctime()
    print t
    expdate = t[8:10] + '-' + t[4:7] + '-' + t[20:24] #  '00-ooo-0000'    #character*11
    print 'Experiment date= ', expdate

    print('\n\nTesting single channel simmulation...')
    # LOAD DEMO MECHANISM (C&H82 numerical example).
    mec = samples.CH82()
    mec.printout(sys.stdout)
    tres = 0.00002 * 1000 # in ms
    impose_res = 1
    tcrit = 0.004 * 1000 # in ms
    conc = 100e-9

    mec.set_eff('c', conc)

    pi = scl.transition_probability(mec.Q)
    print '\n\npi=\n', pi
    picum = np.cumsum(pi,axis=1)
    print '\npicum=\n', picum
    # calculate mean lifetimes
    tmean = -1000 / mec.Q.diagonal() # in ms
    print '\ntmean=\n', tmean
    opamp = 5

    nintmax = 5000 # Number of intervals to be simulated
    nint = 0 # number of transitions counter
    ntrns = 0
    nbmax = 20 # Maximum number of openings/burst expected
    
    # Initial state
    inst = mec.k -1
    a = 0
    if inst < mec.kA:
        a = opamp
    t = - tmean[inst] * math.log(random.random())
    print '\nstarting tstate ', inst+1, '\t (ms)=', t, '\tamplitude (pA)=', a
    first = 1
    itint = [t]
    iampl = [a]

    while nint < nintmax:

        newst, t, a = next_state(inst, picum, tmean, mec.kA, opamp)
        ntrns += 1
        if t < tres:
            itint[-1] += t
            a = iampl[-1]
        else:
            if ((a != 0 and iampl[-1] != 0) or (a == 0 and iampl[-1] == 0)):
                itint[-1] += t
            else:
                itint.append(t)
                iampl.append(a)
                nint += 1
        inst = newst

    print '\n\t number of intervals=', len(itint)
    print '\n\t number of transitions=', ntrns

#    ###
#    dt = 0.00001 # 10 microsec
#    tot = np.sum(itint)
#    t = -dt
#    tsum = 0.0
#    for i in range(len(itint)):
#        tsum += itint[i]
#        while t < (tsum - dt):
#            t += dt
#            ampl.append(iampl[i])
#    trace = np.array(ampl, dtype=int16)

    

#    try:
    dcio.scn_write_simulated(itint, iampl, treso=tres/1000, tresg=tres/1000,
        Emem=-100.0, avamp = opamp, filename='./pySIMSCN.SCN')
#    dcio.ssd_save('./pyCONSAM.ssd', header, trace)
#    except:
#        print 'SCN or CONSAM file saving problem'
