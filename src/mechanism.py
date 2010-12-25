#!/usr/bin/python
"""
Class Mechanism represents a kinetic reaction scheme.

CH82: Colquhoun D, Hawkes AG (1982)
On the stochastic properties of bursts of single ion channel openings
and of clusters of bursts. Phil Trans R Soc Lond B 300, 1-59.

"""

# TODO: impose detailed microscopic reversibility.
# TODO: impose constrains (e.g. independent binding sites).
# TODO: fix certain rate constants while fitting.
# TODO: Check state numbers for consistency

import numpy as np

class Rate(object):
    """
    Describes a rate between two states.
    """

    def __init__(self, rate, state1, state2, name='', eff=None, fixed=False, mr=False):

        self.name = name
        self.rate = rate

        if not isinstance(state1, int) or not isinstance(state2, int):
            raise TypeError("States have to be of type int")
        self.state1 = state1
        self.state2 = state2

        self.eff = eff # Effector; e.g. 'c' or 'v'; or even 'Glu', 'ACh', etc.
                       # We might need to expand this to a list if a rate
                       # depends on more than one effector.
        
        self.fixed = fixed # for future expansion (fixed while fitting)
        self.mr = mr # for future expansion (set by microscopic reversibility)

class State(object):
    """
    Describes a state.
    """
    
    def __init__(self, no, statetype='', name='', conductance=0.0):
        if not isinstance(no, int):
            raise TypeError("State number has to be of type int")
        self.no = no

        if statetype not in ['A', 'B', 'C', 'D']:
            raise RuntimeError("State has to be one of 'A', 'B', 'C' or 'D'")
        self.statetype = statetype

        self.name = name
        self.conductance = conductance

def initQ(Rates, States):
    Q = np.zeros((len(States), len(States)), dtype=np.float64)

    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            # find rate that describes i->j (if any):
            for Rate in Rates:
                if Rate.state1-1 == i and Rate.state2-1 == j:
                    Q[i,j] = Rate.rate
                    break

    return Q

class Mechanism(object):
    '''
    Represents a kinetic mechanism / scheme.
    '''

    def __init__(self, Rates, States, ncyc=0, fastblk=False, KBlk=None):

        #if len(Rates) != len(States)*2:
        #    raise RuntimeError("Not enough rates for given number of states")
        
        self.Rates = Rates
        self.States = States

        self.__Q0 = initQ(self.Rates, self.States)

        self.kA = 0
        self.kB = 0
        self.kC = 0
        self.kD = 0
        for State in self.States:
            if State.statetype=='A':
                self.kA += 1
            if State.statetype=='B':
                self.kB += 1
            if State.statetype=='C':
                self.kC += 1
            if State.statetype=='D':
                self.kD += 1

        self.ncyc = ncyc   # number of cycles; could be deduced from the rates!
        self.fastblk = fastblk
        self.KBlk = KBlk

        # Update diagonal elements
        for d in range(self.__Q0.shape[0]):
            self.__Q0[d,d] = 0
            self.__Q0[d,d] = -np.sum(self.__Q0[d])

        self.Q = self.__Q0.copy() # concentration-dependent Q

    def set_eff(self, eff, val):
        # find rates that are effector-dependent:
        for Rate in self.Rates:
            if Rate.eff == eff:
                self.Q[Rate.state1-1, Rate.state2-1] = \
                    self.__Q0[Rate.state1-1, Rate.state2-1] * val

        # Update diagonal elements
        for d in range(self.Q.shape[0]):
            self.Q[d,d] = 0
            self.Q[d,d] = -np.sum(self.Q[d])
