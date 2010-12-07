#!/usr/bin/python
"""
Class Mechanism represents a kinetic reaction scheme.
Currently it can generate Q matrix used for numerical example in DC82
and other papers.

CH82: Colquhoun D, Hawkes AG (1982)
On the stochastic properties of bursts of single ion channel openings
and of clusters of bursts. Phil Trans R Soc Lond B 300, 1-59.
"""
__author__="R.Lape, University College London"
__date__ ="$6-Dec-2010 21:23:07$"

import numpy as np

class Mechanism:
    '''
    Represents a kinetic mechanism / scheme.
    '''

    global k
    global kA
    global kB
    global kC
    global kD
    global QT
    global Q

    global ncdep    # number of concentration dependent rates
    global cdr    # numpay array; concentration dependent rates
    global ncyc    # number of cycles
    global dgamma
    global fastblk

    def __init__(self):
	print "Mechanism initialised"

    def __del__(self):
	print "Mechanism deleted"

    def init_Q(self, c, debug=False):
        """
        Calculate Q matrix.
        Multiply concentration dependent rate constants by concentration.
        TODO: impose detailed microscopic reversibility.
        TODO: impose constrains (e.g. independent binding sites).
        TODO: fix certain rate constants while fitting.
        """

        self.Q = np.zeros((self.k, self.k), 'float64')
        QD = np.zeros((self.k, self.k), 'float64')
        for i in range(self.k):
            for j in range(self.k):
                QD[i,j] = self.QT[i,j]

        for i in range(self.ncdep):
            QD[self.cdr[i,0]-1, self.cdr[i,1]-1] = QD[self.cdr[i,0]-1, self.cdr[i,1]-1] * c

        for i in range(self.k):
            sum = 0
            for j in range(self.k):
                self.Q[i, j] = QD[i, j]
                sum = sum + QD[i, j]
            self.Q[i, i] = -sum

    def demoQ(self):
        """
        Create Q matrix as in CH82 numerical example.
        """

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

        self.QT = np.array([[   0,  k_star_p2,        0,   alpha_1,     0],
                      [ k_star_m2,          0,  alpha_2,         0,     0],
                      [         0,     beta_2,        0,      k_m2,     0],
                      [    beta_1,          0,     k_p2,         0,  k_m1],
                      [         0,          0,        0,      k_p1,    0]])

        self.kA = 2
        self.kB = 2
        self.kC = 1
        self.kD = 0
        self.k = 5

        self.ncdep = 3
        self.cdr = np.array([[1,2,1],[4,3,1],[5,4,1]])
        self.ncyc = 1
        self.dgamma = 50e-12
        self.fastblk = False
