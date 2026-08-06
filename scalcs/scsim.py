import math
import random
import numpy as np

def simulate_intervals(mec, tres, state, opamp=5, nintmax=5000):
    """

    """
    # Initialise simulation
    # Cummulative transition probability
    picum = np.cumsum(transition_probability(mec.Q), axis=1)
    #Mean lifetime in each state
    tmean = -1 / mec.Q.diagonal() # in s
    # interval and transition counters
    nint, ntrns = 0, 0 
    # initial state
    inst = state
    # amplitude of initial state
    a = opamp if inst < mec.kA else 0
    # length of initial interval
    t = - tmean[inst] * math.log(random.random())
    # lists to keep intervals and amplitudes
    tints, ampls = [t], [a]
    
    while nint < nintmax-1:

        newst, t, a = next_state(inst, picum, tmean, mec.kA, opamp)
        ntrns += 1
        if t < tres:
            tints[-1] += t
            a = ampls[-1]
        else:
            if ((a != 0 and ampls[-1] != 0) or (a == 0 and ampls[-1] == 0)):
                tints[-1] += t
            else:
                tints.append(t)
                ampls.append(a)
                nint += 1
        inst = newst
    
    return np.array(tints), np.array(ampls), np.zeros((len(tints)), dtype='b'), ntrns

def next_state(present, picum, tmean, kA, opamp):
    """
    Get next state, its lifetime and amplitude.
    """
    possible = np.nonzero(picum[present] >= random.random())[0]
    next = np.delete(possible, np.where(possible == present))[0]
    t = random.expovariate(1 / tmean[next])
    a = opamp if next < kA else 0
    return next, t, a

def transition_probability(Q):
    """
    """
    k = Q.shape[0]
    pi = Q.copy()
    for i in range(k):
        pi[i] = pi[i] / -Q[i,i]
        pi[i,i] = 0
    return pi