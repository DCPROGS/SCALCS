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
# TODO: Update docstrings

import numpy as np

import qmatlib as qml

def multiply(rate, value):
    """
    Multiply rate and value. Used as default rate function.

    Parameters
    ----------
    rate : float
        Current rate in Q matrix.
    value : float
        Effector value (typically, concentration or voltage).

    Returns
    -------
    product : float
        Product of rate and value.
    """
    
    return rate[0]*value

class State(object):
    """
    Describes a state.
    """
    
    def __init__(self, statetype='', name='', conductance=0.0):
        if statetype not in ['A', 'B', 'C', 'D']:
            raise RuntimeError("State has to be one of 'A', 'B', 'C' or 'D'")
        self.statetype = statetype

        self.name = name
        self.conductance = conductance
        self.no = None # will be assigned in Mechanism.__init__
                       # This is now ZERO-based!

class Rate(object):
    """
    Describes a rate between two states.
    """

    def __init__(self, ratepars, State1, State2, name='', eff=None, fixed=False,
                 mr=False, func=multiply):

        self.name = name
        try: 
            # test whether ratepars is a sequence:
            it = iter(ratepars)
            # test whether this is a numpy array:
            if isinstance(ratepars, np.ndarray):
                self.ratepars = ratepars
            else:
                # else, convert:
                self.ratepars = np.array(ratepars)
        except TypeError:
            # if not, convert to single-itemed list:
            self.ratepars = np.array([ratepars,])

        if not isinstance(State1, State) or not isinstance(State2, State):
            raise TypeError("DCPYPS: States have to be of class State")
        self.State1 = State1
        self.State2 = State2

        self.eff = eff # Effector; e.g. 'c' or 'v'; or even 'Glu', 'ACh', etc.
                       # We might need to expand this to a list if a rate
                       # depends on more than one effector.
        
        self.fixed = fixed # for future expansion (fixed while fitting)
        self.mr = mr # for future expansion (set by microscopic reversibility)
        self.func = func # f(ratepars, amount of effector); "Rate equation" if you wish

    def update(self, val):
        return self.func(self.ratepars, val)

def initQ(Rates, States):
    Q = np.zeros((len(States), len(States)), dtype=np.float64)

    # find rate that describes i->j (if any):
    for Rate in Rates:
        i = Rate.State1.no
        j = Rate.State2.no
        # check range:
        if i<0 or i>=Q.shape[0]:
            raise IndexError("DCPYPS: Rate.state1 is out of range")
        if j<0 or j>=Q.shape[1]:
            raise IndexError("DCPYPS: Rate.state2 is out of range")
        Q[i,j] = Rate.update(1.0)

    return Q

class Mechanism(object):
    '''
    Represents a kinetic mechanism / scheme.
    '''

    def __init__(self, Rates, ncyc=0, fastblk=False, KBlk=None):

        self.Rates = Rates
        # construct States end effectors from Rates:
        self.States = []
        self.efflist = []
        for rate in self.Rates:
            if rate.State1 not in self.States:
                self.States.append(rate.State1)
            if rate.State2 not in self.States:
                self.States.append(rate.State2)
            if rate.eff not in self.efflist:
                self.efflist.append(rate.eff)

        # REMIS: please check whether this makes sense
        # sort States according to state type:
        self.States.sort(key=lambda state: state.statetype.lower())
        # assign Q matrix indices according to sorted list:
        for no, state in enumerate(self.States):
            state.no = no # ZERO-based!

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
        self.kF = self.kB + self.kC
        self.kE = self.kA + self.kB

        self.ncyc = ncyc   # number of cycles; could be deduced from the rates!
        self.fastblk = fastblk
        self.KBlk = KBlk

        self.Q = np.zeros((len(self.States), len(self.States)), dtype=np.float64)

        for eff in self.efflist:
            # find rates that are effector-dependent:
            for Rate in self.Rates:
                if Rate.eff == eff:
                    self.Q[Rate.State1.no, Rate.State2.no] = \
                        Rate.update(1.0)

            # Update diagonal elements
            for d in range(self.Q.shape[0]):
                self.Q[d,d] = 0
                self.Q[d,d] = -np.sum(self.Q[d])

    def set_eff(self, eff, val):
        if eff not in self.efflist:
            sys.stderr.write("DCPYPS: None of the rates depends on effector %s\n" % eff)
 
        # find rates that are effector-dependent:
        for Rate in self.Rates:
            if Rate.eff == eff:
                self.Q[Rate.State1.no, Rate.State2.no] = \
                    Rate.update(val)

        # Update diagonal elements
        for d in range(self.Q.shape[0]):
            self.Q[d,d] = 0
            self.Q[d,d] = -np.sum(self.Q[d])

        # print self.Q

        self.eigenvals, self.A = qml.eigs(self.Q)
        self.GAB, self.GBA = qml.iGs(self.Q, self.kA, self.kB)
        self.QFF = self.Q[self.kA:, self.kA:]
        self.QFA = self.Q[self.kA:, :self.kA]
        self.QAF = self.Q[:self.kA, self.kA:]
        self.QAA = self.Q[:self.kA, :self.kA]
        self.QEE = self.Q[:self.kE, :self.kE]
        self.QBB = self.Q[self.kA:self.kE, self.kA:self.kE]
        self.QAB = self.Q[:self.kA, self.kA:self.kE]
        self.QCB = self.Q[self.kE:, self.kA:self.kE]
        self.QCA = self.Q[self.kE:, :self.kA]

        
