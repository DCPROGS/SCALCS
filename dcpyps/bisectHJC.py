"""dc-pyps functions related to HJC root search."""

import sys
import math

import numpy as np
from numpy import linalg as nplin

import dcpypsrc

import qmatlib as qml

def gFB(s, tres, Q11, Q22, Q12, Q21, k1, k2):
    """
    Find number of eigenvalues of H(s) that are equal to or less than s.

    Parameters
    ----------
    s : float
        Laplace transform argument.
    tres : float
        Time resolution (dead time).
    Q11 : array_like, shape (k1, k1)
    Q22 : array_like, shape (k2, k2)
    Q21 : array_like, shape (k2, k1)
    Q12 : array_like, shape (k1, k2)
        Q11, Q12, Q22, Q21 - submatrices of Q.
    k1 : int
        A number of open/shut states in kinetic scheme.
    k2 : int
        A number of shut/open states in kinetic scheme.

    Returns
    -------
    ng : int
    """

    h = qml.H(s, tres, Q11, Q22, Q12, Q21, k2)
    eigval, A = qml.eigs(h)
    ng = 0
    for i in range(k1):
        if eigval[i] <= s: ng += 1
    if dcpypsrc.debug:
        print ('number of eigenvalues that are <= s (=', s, ') =', ng)
    return ng

def bisection_intervals(sa, sb, tres, Q11, Q22, Q12, Q21, k1, k2):
    """
    Find, according to Frank Ball's method, suitable starting guesses for
    each HJC root- the upper and lower limits for bisection. Exactly one root
    should be between those limits.

    Parameters
    ----------
    sa, sb : float
        Laplace transform arguments.
    tres : float
        Time resolution (dead time).
    Q11 : array_like, shape (k1, k1)
    Q22 : array_like, shape (k2, k2)
    Q21 : array_like, shape (k2, k1)
    Q12 : array_like, shape (k1, k2)
        Q11, Q12, Q22, Q21 - submatrices of Q.
    k1, k2 : int
        Numbers of open/shut states in kinetic scheme.

    Returns
    -------
    sr : array_like, shape (k2, 2)
        Limits of s value intervals containing exactly one root.
    """

    nga = gFB(sa, tres, Q11, Q22, Q12, Q21, k1, k2)
    if nga > 0:
        sa = sa * 4
    ngb = gFB(sb, tres, Q11, Q22, Q12, Q21, k1, k2)
    if ngb < k2:
        sb = sb / 4

    sr = np.zeros((k1, 2))
    sv = np.empty((101, 4))
    sv[0,0] = sa
    sv[0,1] = sb
    sv[0,2] = nga
    sv[0,3] = ngb
    ntodo = 0
    ndone = 0
    nsplit = 0

    while (ndone < k1) and (nsplit < 1000):
        sa = sv[ntodo, 0]
        sb = sv[ntodo, 1]
        nga = sv[ntodo, 2]
        ngb = sv[ntodo, 3]
        sa1, sc, sb2, nga1, ngc, ngb2 = split(sa, sb,
            nga, ngb, tres, Q11, Q22, Q12, Q21, k1, k2)
        nsplit = nsplit + 1
        ntodo = ntodo - 1

        # Check if either or both of the two subintervals output from
        # SPLIT contain only one root?
        if (ngc - nga1) == 1:
            sr[ndone, 0] = sa1
            sr[ndone, 1] = sc
            ndone = ndone + 1
            if ndone == k1:
                break
        else:
            ntodo = ntodo + 1
            sv[ntodo, 0] = sa1
            sv[ntodo, 1] = sc
            sv[ntodo, 2] = nga1
            sv[ntodo, 3] = ngc
        if (ngb2 - ngc) == 1:
            sr[ndone, 0] = sc
            sr[ndone, 1] = sb2
            ndone = ndone + 1
        else:
            ntodo = ntodo + 1
            sv[ntodo, 0] = sc
            sv[ntodo, 1] = sb2
            sv[ntodo, 2] = ngc
            sv[ntodo, 3] = ngb2

    if ndone < k1:
        sys.stderr.write(
            "bisectHJC: Warning: Only {0:d} roots out of {1:d} were located.".
            format(ndone, k1))

    return sr

def split(sa, sb, nga, ngb, tres, Q11, Q22, Q12, Q21, k1, k2):
    """
    Split interval [sa, sb] into two subintervals, each of which contains
    at least one root.

    Parameters
    ----------
    sa, sb : float
        Limits of Laplace transform argument interval.
    nga, ngb : int
        Number of eigenvalues (roots) below sa or sb, respectively.
    tres : float
        Time resolution (dead time).
    Q11 : array_like, shape (k1, k1)
    Q22 : array_like, shape (k2, k2)
    Q21 : array_like, shape (k2, k1)
    Q12 : array_like, shape (k1, k2)
        Q11, Q12, Q22, Q21 - submatrices of Q.
    k1, k2 : int
        Numbers of open/shut states in kinetic scheme.

    Returns
    -------
    sa, sc, sb : floats
        Limits of s value intervals.
    nga, ngc, ngb : ints
        Number of eigenvalues below corresponding s values.
    """

    ntrymax = 1000
    ntry = 0
    #nerrs = False
    end = False

    while (not end) and (ntry < ntrymax):
        sc = (sa + sb) / 2.0
        ngc = gFB(sc, tres, Q11, Q22, Q12, Q21, k1, k2)
        if ngc == nga: sa = sc
        elif ngc == ngb: sb = sc
        else:
            end = True
        ntry += 1
    if not end:
        sys.stderr.write(
        "bisectHJC: Warning: unable to split intervals for bisection.")

    return sa, sc, sb, nga, ngc, ngb

def bisect(s1, s2, tres, Q11, Q22, Q12, Q21, k1, k2):
    """
    Find asymptotic root (det(W) = 0) in interval [s1, s2] using bisection
    method.

    Parameters
    ----------
    s1, s2 : float
        Limits of Laplace transform argument interval to split.
    tres : float
        Time resolution (dead time).
    Q11 : array_like, shape (k1, k1)
    Q22 : array_like, shape (k2, k2)
    Q21 : array_like, shape (k2, k1)
    Q12 : array_like, shape (k1, k2)
        Q11, Q12, Q22, Q21 - submatrices of Q.
    k1, k2 : int
        Numbers of open/shut states in kinetic scheme.

    Returns
    -------
    sout : float
        Asymptotic root at wich \|W\|=0.
    """

    epsy = 1e-10
    f = nplin.det(qml.W(s1, tres, Q11, Q22, Q12, Q21, k1, k2))
    if f > 0:
        temp = s1
        s1 = s2
        s2 = temp
    iter = 0
    solved = False
    itermax = 100
    sout = None
    flast = 0

    while iter < itermax and not solved:
        iter += 1
        sout = 0.5 * (s1 + s2)
        f = nplin.det(qml.W(sout, tres, Q11, Q22, Q12, Q21, k1, k2))
        if f < 0:
            s1 = sout
        elif f > 0:
            s2 = sout
        else:    #if f == 0:
            solved = True

        if math.fabs(flast - f) < epsy:
            solved = True
        else:
            flast = f

    #if verbose: print 'function solved in', ns, 'itterations'
    return sout

