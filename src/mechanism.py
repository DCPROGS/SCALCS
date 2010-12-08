#!/usr/bin/python
"""
Class Mechanism represents a kinetic reaction scheme.

CH82: Colquhoun D, Hawkes AG (1982)
On the stochastic properties of bursts of single ion channel openings
and of clusters of bursts. Phil Trans R Soc Lond B 300, 1-59.

"""
__author__="R.Lape, University College London"
__date__ ="$6-Dec-2010 21:23:07$"

# TODO: impose detailed microscopic reversibility.
# TODO: impose constrains (e.g. independent binding sites).
# TODO: fix certain rate constants while fitting.
# TODO: Check state numbers for consistency

import numpy as np

class Mechanism(object):
    '''
    Represents a kinetic mechanism / scheme.
    '''

    def __init__(self, Q0, kA, kB, kC, kD, cdrs, ncyc, dgamma,
                 fastblk):

        if not isinstance(Q0, np.ndarray):
            raise TypeError("Q0 has to be a NumPy array")

        if len(Q0.shape) != 2:
            raise RuntimeError("Q0 is not 2D")

        if Q0.shape[0] != Q0.shape[1]:
            raise RuntimeError("Q0 doesn't have square shape")

        self.Q0 = Q0

        self.kA = kA
        self.kB = kB
        self.kC = kC
        self.kD = kD
        self.cdrs = cdrs    # numpy array; concentration dependent rates
        self.ncyc = ncyc   # number of cycles
        self.dgamma = dgamma
        self.fastblk = fastblk

        # Update diagonal elements
        for d in range(self.Q0.shape[0]):
            self.Q0[d,d]=0
            self.Q0[d,d] = -np.sum(self.Q0[d])

        self.Q = self.Q0.copy() # concentration-dependent Q

        self.__c = 0

    def getc(self):
        return self.__c

    def setc(self, c):
        for cdr in self.cdrs:
            self.Q[cdr[0]-1, cdr[1]-1] = self.Q0[cdr[0]-1, cdr[1]-1]*c

        for d in range(self.Q.shape[0]):
            self.Q[d,d]=0
            self.Q[d,d] = -np.sum(self.Q[d])

        self.__c = c

    c = property(getc, setc)
