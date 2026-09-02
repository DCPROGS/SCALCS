"""A collection of functions for dwell time ideal, asymptotic and exact
probabulity density function calculations.

Notes
-----
DC_PyPs project are pure Python implementations of Q-Matrix formalisms
for ion channel research. To learn more about kinetic analysis of ion
channels see the references below.

References
----------
CH82: Colquhoun D, Hawkes AG (1982)
On the stochastic properties of bursts of single ion channel openings
and of clusters of bursts. Phil Trans R Soc Lond B 300, 1-59.

CH87: Colquhoun D, Hawkes AG (1987)
A note on correlations in single ion channel records.
Proc R Soc Lond 230, 15-52.

HJC92: Hawkes AG, Jalali A, Colquhoun D (1992)
Asymptotic distributions of apparent open times and shut times in a
single channel record allowing for the omission of brief events.
Phil Trans R Soc Lond B 337, 383-404.

CH95a: Colquhoun D, Hawkes AG (1995a)
The principles of the stochastic interpretation of ion channel
mechanisms. In: Single-channel recording. 2nd ed. (Eds: Sakmann B,
Neher E) Plenum Press, New York, pp. 397-482.

CH95b: Colquhoun D, Hawkes AG (1995b)
A Q-Matrix Cookbook. In: Single-channel recording. 2nd ed. (Eds:
Sakmann B, Neher E) Plenum Press, New York, pp. 589-633.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 20:29:14$"

import sys
from math import*
from decimal import*
import random

import scipy.optimize as so
import numpy as np
from numpy import linalg as nplin

from scalcs import qmatlib as qml
#import bisectHJC
from scalcs import pdfs
#import optimize

def ideal_dwell_time_pdf(t, QAA, phiA):
    """
    Probability density function of the open time.
    f(t) = phiOp * exp(-QAA * t) * (-QAA) * uA
    For shut time pdf A by F in function call.

    Parameters
    ----------
    t : float
        Time (sec).
    QAA : array_like, shape (kA, kA)
        Submatrix of Q.
    phiA : array_like, shape (1, kA)
        Initial vector for openings

    Returns
    -------
    f : float
    """

    kA = QAA.shape[0]
    uA = np.ones((kA, 1))
    expQAA = qml.expQt(QAA, t)
    f = np.dot(np.dot(np.dot(phiA, expQAA), -QAA), uA)
    return np.asarray(f).item()

def ideal_dwell_time_pdf_components(QAA, phiA):
    """
    Calculate time constants and areas for an ideal (no missed events)
    exponential open time probability density function.
    For shut time pdf A by F in function call.

    Parameters
    ----------
    t : float
        Time (sec).
    QAA : array_like, shape (kA, kA)
        Submatrix of Q.
    phiA : array_like, shape (1, kA)
        Initial vector for openings

    Returns
    -------
    taus : ndarray, shape(k, 1)
        Time constants.
    areas : ndarray, shape(k, 1)
        Component relative areas.
    """

    kA = QAA.shape[0]
    eigs, A = qml.eigs_sorted(-QAA)
    uA = np.ones((kA, 1))
    # w[i] = phiA @ A[i] @ (-QAA) @ uA  for each i.
    # Vectorised: contract phiA over A's second axis, then matmul (-QAA) and uA.
    # np.einsum('j,ijk->ik', phiA, A) has shape (kA, kA); @ (-QAA) → (kA, kA);
    # @ uA → (kA, 1); squeeze to (kA,).
    tmp = np.einsum('j,ijk->ik', phiA, A)   # (kA, kA)
    w = (tmp @ (-QAA) @ uA).ravel()         # (kA,)

    return eigs, w

def ideal_subset_time_pdf(Q, k1, k2, t):
    """
    
    """
    
    u = np.ones((k2 - k1 + 1, 1))
    phi, QSub = qml.phiSub(Q, k1, k2)
    expQSub = qml.expQt(QSub, t)
    f = np.dot(np.dot(np.dot(phi, expQSub), -QSub), u)
    return f

def ideal_subset_mean_life_time(Q, state1, state2):
    """
    Calculate mean life time in a specified subset. Add all rates out of subset
    to get total rate out. Skip rates within subset.

    Parameters
    ----------
    mec : instance of type Mechanism
    state1,state2 : int
        State numbers (counting origin 1)

    Returns
    -------
    mean : float
        Mean life time.
    """

    k = Q.shape[0]
    p = qml.pinf(Q)
    # Total occupancy for subset.
    pstot = np.sum(p[state1-1 : state2])

    # Total rate out
    if pstot == 0:
        mean = 0.0
    else:
        # Sum Q[i,j] * p[i] / pstot for rows i in subset, columns j outside subset.
        rows = np.arange(state1 - 1, state2)
        mask = np.ones(k, dtype=bool)
        mask[state1 - 1:state2] = False          # True for columns outside subset
        s = float(np.sum(Q[np.ix_(rows, mask)] * (p[rows] / pstot)[:, np.newaxis]))

        mean = 1 / s
    return mean

def ideal_mean_latency_given_start_state(mec, state):
    """
    Calculate mean latency to next opening (shutting), given starting in
    specified shut (open) state.

    mean latency given starting state = pF(0) * inv(-QFF) * uF

    F- all shut states (change to A for mean latency to next shutting
    calculation), p(0) = [0 0 0 ..1.. 0] - a row vector with 1 for state in
    question and 0 for all other states.

    Parameters
    ----------
    mec : instance of type Mechanism
    state : int
        State number (counting origin 1)

    Returns
    -------
    mean : float
        Mean latency.
    """

    if state <= mec.kA:
        # for calculating mean latency to next shutting
        p = np.zeros((mec.kA))
        p[state-1] = 1
        u = np.ones((mec.kA, 1))
        invQ = nplin.inv(-mec.QAA)
    else:
        # for calculating mean latency to next opening
        p = np.zeros((mec.kI))
        p[state-mec.kA-1] = 1
        u = np.ones((mec.kI, 1))
        invQ = nplin.inv(-mec.QII)

    mean = np.dot(np.dot(p, invQ), u)[0]

    return mean

def asymptotic_pdf(t, tres, tau, area):
    """
    Calculate asymptotic probabolity density function.

    Parameters
    ----------
    t : ndarray.
        Time.
    tres : float
        Time resolution.
    tau : ndarray, shape(k, 1)
        Time constants.
    area : ndarray, shape(k, 1)
        Component relative area.

    Returns
    -------
    apdf : ndarray.
    """
    t1 = np.extract(t[:] < tres, t)
    t2 = np.extract(t[:] >= tres, t)
    apdf2 = t2 * pdfs.expPDF(t2 - tres, tau, area)
    apdf = np.append(t1 * 0.0, apdf2)

    return apdf

def asymptotic_roots(tres, QAA, QFF, QAF, QFA, kA, kF):
    """Find roots for the asymptotic probability density function (Eqs. 52-58,
    HJC92).

    Solves det[W(s)] = 0 for the kA roots that lie in the range
    (−10⁶, −10⁻⁷).  Each root is refined with Brent's method.

    Parameters
    ----------
    tres : float
        Time resolution (dead time) in seconds.
    QAA : array_like, shape (kA, kA)
    QFF : array_like, shape (kF, kF)
    QAF : array_like, shape (kA, kF)
    QFA : array_like, shape (kF, kA)
        Submatrices of the full Q matrix.
    kA : int
        Number of open states in the kinetic scheme.
    kF : int
        Number of shut states in the kinetic scheme.

    Returns
    -------
    roots : ndarray, shape (kA,)
        Negative real roots, one per open state.

    Notes
    -----
    **Numerical stability** — The matrix exponential
    ``exp((QFF − s·I)·tres)`` is evaluated during the root-count sweep.
    When ``|s| × tres > ln(float64_max) ≈ 709`` this overflows.  The lower
    search bound is therefore clamped adaptively::

        sas = max(−1_000_000, −700 / tres)

    This only tightens the bound when the default −10⁶ would cause overflow
    (i.e. when ``tres > 0.7 ms``); otherwise the full range is used so that
    bisection can correctly count and bracket all roots.

    At ``tres = 0`` the matrix H(s) becomes constant, making the root-counting
    function a step function.  ``bisect_intervals`` handles this correctly by
    discarding zero-root sub-intervals rather than re-queuing them.
    """

    # The lower search bound must satisfy |sas| * tres < ln(float64_max) ≈ 709
    # to prevent exp((QFF - s·I)·tres) from overflowing during the sweep.
    # Only tighten the bound when the default −10⁶ would cause overflow; keep
    # it at −10⁶ otherwise so that the bisection counting algorithm has enough
    # room to correctly bracket all roots.
    _exp_bound = -700.0 / max(tres, 1e-20)   # exp(700) safely < float64 max
    sas = max(-1_000_000, _exp_bound)         # less-negative = tighter bound
    sbs = -0.0000001
    sro = bisect_intervals(sas, sbs, tres,
        QAA, QFF, QAF, QFA, kA, kF)

    roots = np.zeros(kA)
    for i in range(kA):
        roots[i] = _refine_root(sro[i, 0], sro[i, 1], tres,
                                QAA, QFF, QAF, QFA, kA, kF)

    # A root of det[W(s)] = 0 is an s that is its own eigenvalue of H(s).
    # Checking that is cheap next to the bisection that found it, and it is
    # the only thing standing between a caller and a plausible-looking number
    # produced in a regime where H(s) could not be evaluated at all.
    #
    # That regime is reachable by ordinary means: a mechanism whose
    # concentration has never been set keeps its association rates
    # unscaled, so CH82 straight from scalcs.samples has Q eigenvalues near
    # -5e8 rather than -3e3. H(s) evaluates exp((QFF - s*I)*tres), so at
    # tres = 10 us that asks for exp(5000). The old code returned whatever
    # came back; this says what went wrong.
    for i, root in enumerate(roots):
        h = qml.H(root, tres, QAA, QFF, QAF, QFA, kF)
        residual = np.min(np.abs(nplin.eigvals(h).real - root))
        if residual > 1e-6 * max(abs(root), 1.0):
            raise RuntimeError(
                "asymptotic_roots: root {0} of {1} did not converge "
                "(s = {2:.6g}, |eig(H(s)) - s| = {3:.3g}, |s|*tres = {4:.3g}). "
                "The search reached a range where H(s) cannot be evaluated "
                "usefully. Check that the mechanism's rates are set: an unset "
                "concentration leaves association rates unscaled, which pushes "
                "the Q eigenvalues orders of magnitude too fast and the roots "
                "with them. Above |s|*tres of about 700 the matrix exponential "
                "in H(s) overflows outright."
                .format(i + 1, kA, root, residual, abs(root) * tres))

    return roots


def _refine_root(sa, sb, tres, QAA, QFF, QAF, QFA, kA, kF):
    """Locate the single root of det[W(s)] = 0 bracketed by [sa, sb].

    Brent's method on det[W(s)] where that is trustworthy, and bisection on
    the root count where it is not.

    det[W(s)] is not usable at large |s|. It carries a factor
    exp((QFF - s*I)*tres), so at s = -1.5e6 with tres = 60 us the determinant
    is of order 1e53 -- far outside the range where its sign means anything.
    Sampled across such a bracket it changes sign hundreds of times, all of it
    noise, and at other resolutions it changes sign not at all while a root
    provably sits inside. Handing either case to brentq gives "f(a) and f(b)
    must have different signs", which is what CH82 did at most resolutions,
    with the set of failing ones depending on the scipy version.

    bisect_gFB is well behaved wherever detW is not: it counts eigenvalues of
    H(s) that do not exceed s, and increments by one exactly at the root. That
    count is what bracketed the root in the first place, so bisecting on it
    cannot disagree with the bracket.
    """

    fa = qml.detW(sa, tres, QAA, QFF, QAF, QFA, kA, kF)
    fb = qml.detW(sb, tres, QAA, QFF, QAF, QFA, kA, kF)

    if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
        return so.brentq(qml.detW, sa, sb,
                         args=(tres, QAA, QFF, QAF, QFA, kA, kF))

    # Bisect on the root count. The count at sa is some n; inside the bracket
    # it becomes n + 1. Converge on the step.
    nga = bisect_gFB(sa, tres, QAA, QFF, QAF, QFA, kA, kF)
    lo, hi = sa, sb
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:          # exhausted double precision
            break
        if bisect_gFB(mid, tres, QAA, QFF, QAF, QFA, kA, kF) > nga:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) <= 1e-12 * max(abs(lo), abs(hi)):
            break
    return 0.5 * (lo + hi)

def bisect_gFB(s, tres, Q11, Q22, Q12, Q21, k1, k2):
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
    eigval = nplin.eigvals(h)
    ng = (eigval <= s).sum()
    return ng

def bisect_intervals(sa, sb, tres, Q11, Q22, Q12, Q21, k1, k2):
    """Find bracket intervals that each contain exactly one HJC root.

    Uses Frank Ball's bisection method to partition the search range
    [sa, sb] into sub-intervals, each of which contains exactly one root of
    det[W(s)] = 0.

    Parameters
    ----------
    sa, sb : float
        Lower and upper Laplace-transform argument bounds.  ``sa`` must be
        more negative than the most negative root; ``sb`` must be between
        the smallest-magnitude root and zero.
    tres : float
        Time resolution (dead time) in seconds.
    Q11 : array_like, shape (k1, k1)
    Q22 : array_like, shape (k2, k2)
    Q21 : array_like, shape (k2, k1)
    Q12 : array_like, shape (k1, k2)
        Submatrices of the full Q matrix.
    k1, k2 : int
        Numbers of open / shut states in the kinetic scheme.

    Returns
    -------
    sr : ndarray, shape (k2, 2)
        Each row is [s_low, s_high] bracketing exactly one root.

    Notes
    -----
    **Infinite-loop guard** — Sub-intervals that contain zero roots are
    discarded silently (not re-queued).  This is essential when
    ``tres = 0``: H(s) is then constant (= Q11) and the root-counting
    function ``bisect_gFB`` is a step function, so the left half of any
    bisection step always has zero roots and would loop forever if
    re-queued unconditionally.
    """

    # The whole algorithm depends on nga and ngb being the root counts *at the
    # current endpoints*. Widening a bound without recounting leaves a stale
    # number behind, and the top-level interval is then entered claiming fewer
    # roots than it holds -- which is how CH82 at its default concentration
    # produced one bracket for two roots, and an IndexError several frames
    # later in asymptotic_roots.
    #
    # Widening also has to repeat. A single multiplication by four is not
    # enough whenever the outermost root lies more than four times below the
    # starting bound, and nothing guaranteed that it did not.
    #
    # The lower bound is capped: H(s) evaluates exp((Q22 - s*I)*tres), so
    # |s|*tres must stay under ln(float64 max) ~ 709.
    floor = -700.0 / max(tres, 1e-20)

    nga = bisect_gFB(sa, tres, Q11, Q22, Q12, Q21, k1, k2)
    while nga > 0 and sa > floor:
        sa = max(sa * 4.0, floor)
        nga = bisect_gFB(sa, tres, Q11, Q22, Q12, Q21, k1, k2)

    # k1, not k2: the number of roots of det[W(s)] = 0 is the size of the
    # partition being solved for, which is k1. Comparing against k2 made the
    # test vacuously true whenever k2 > k1 -- for CH82, ngb < 3 always holds
    # when there are only 2 roots -- so sb was moved on every call for no
    # reason, again without recounting.
    ngb = bisect_gFB(sb, tres, Q11, Q22, Q12, Q21, k1, k2)
    while ngb < k1 and sb < -1e-30:
        sb = sb / 4.0
        ngb = bisect_gFB(sb, tres, Q11, Q22, Q12, Q21, k1, k2)

    done = []
    todo = [[sa, sb, nga, ngb]]
#    nsplit = 0

#    while (len(done) < k1) and (nsplit < 1000):
    while todo:
        svv = todo.pop()

        # An interval known to contain exactly one root cannot be split any
        # further: bisect_split terminates only when the mid-point count is a
        # *third* value strictly between the two endpoint counts, which never
        # occurs once a single root is isolated.  Handing such an interval to
        # bisect_split makes it spin the full ntrymax loop and emit a spurious
        # "unable to split intervals" warning.  This is the k1 == 1 case
        # (single-state partition: e.g. the del Castillo-Katz / CCO open and
        # within-burst-gap survivors, or any monoliganded kB == 1 gap), where
        # the top-level interval holds the lone root from the outset.  Bracket
        # it directly instead.
        if (svv[3] - svv[2]) == 1:
            done.append([svv[0], svv[1]])
            continue

        sa1, sc, sb2, nga1, ngc, ngb2 = bisect_split(svv[0], svv[1], svv[2], svv[3],
            tres, Q11, Q22, Q12, Q21, k1, k2)
#        nsplit += 1

        # Check if either or both of the two subintervals output from
        # SPLIT contain only one root?
        # Only push back onto todo if the subinterval actually contains
        # roots (count difference > 0).  Pushing a zero-root interval
        # back causes an infinite loop because it keeps splitting into
        # two more zero-root halves.
        if (ngc - nga1) == 1:
            done.append([sa1, sc])
        elif (ngc - nga1) > 1:
            todo.append([sa1, sc, nga1, ngc])
        # ngc - nga1 == 0: no roots in left half, discard silently
        if (ngb2 - ngc) == 1:
            done.append([sc, sb2])
        elif (ngb2 - ngc) > 1:
            todo.append([sc, sb2, ngc, ngb2])
        # ngb2 - ngc == 0: no roots in right half, discard silently

    if len(done) < k1:
        # This used to warn to stderr and return a short array, leaving the
        # caller to fail with an IndexError several frames away, or to hand
        # brentq an interval with no sign change. Neither message named the
        # problem. Fail here, where the cause is known.
        raise RuntimeError(
            "bisectHJC: bracketed {0:d} of {1:d} roots of det[W(s)] = 0 in "
            "[{2:.6g}, {3:.6g}] at tres = {4:g} s. The search range does not "
            "contain them all, or roots are too close together to separate."
            .format(len(done), k1, sa, sb, tres))
    return np.array(done)

def bisect_split(sa, sb, nga, ngb, tres, Q11, Q22, Q12, Q21, k1, k2):
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
        ngc = bisect_gFB(sc, tres, Q11, Q22, Q12, Q21, k1, k2)
        if ngc == nga: sa = sc
        elif ngc == ngb: sb = sc
        else:
            end = True
        ntry += 1
    if not end:
        sys.stderr.write(
        "bisectHJC: Warning: unable to split intervals for bisection.")

    return sa, sc, sb, nga, ngc, ngb

def asymptotic_areas(tres, roots, QAA, QFF, QAF, QFA, kA, kF, GAF, GFA):
    """
    Find the areas of the asymptotic pdf (Eq. 58, HJC92).

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    roots : array_like, shape (1,kA)
        Roots of the asymptotic pdf.
    QAA : array_like, shape (kA, kA)
    QFF : array_like, shape (kF, kF)
    QAF : array_like, shape (kA, kF)
    QFA : array_like, shape (kF, kA)
        QAA, QFF, QAF, QFA - submatrices of Q.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.
    GAF : array_like, shape (kA, kB)
    GFA : array_like, shape (kB, kA)
        GAF, GFA- transition probabilities

    Returns
    -------
    areas : ndarray, shape (1, kA)
    """

    expQFF = qml.expQt(QFF, tres)
    expQAA = qml.expQt(QAA, tres)
    eGAF = qml.eGs(GAF, GFA, kA, kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, kF, kA, expQAA)
    phiA = qml.phiHJC(eGAF, eGFA, kA)
    R = qml.AR(roots, tres, QAA, QFF, QAF, QFA, kA, kF)
    uF = np.ones((kF, 1))
    # areas[i] = (-1/roots[i]) * phiA @ R[i] @ QAF @ expQFF @ uF
    # Let v = QAF @ expQFF @ uF  shape (kA, 1).
    # R @ v via matmul: R shape (kA,kA,kA), v shape (kA,1) → (kA,kA,1).
    # einsum 'j,ijk,k->i' over phiA, R, v.ravel() gives shape (kA,).
    v = (QAF @ expQFF @ uF).ravel()                      # (kA,)
    areas = (-1.0 / roots) * np.einsum('j,ijk,k->i', phiA, R, v)

#    rowA = np.zeros((kA,kA))
#    colA = np.zeros((kA,kA))
#    for i in range(kA):
#        WA = qml.W(roots[i], tres,
#            QAA, QFF, QAF, QFA, kA, kF)
#        rowA[i] = qml.pinf(WA)
#        AW = np.transpose(WA)
#        colA[i] = qml.pinf(AW)
#
#    for i in range(kA):
#        uF = np.ones((kF,1))
#        nom = np.dot(np.dot(np.dot(np.dot(np.dot(phiA, colA[i]), rowA[i]),
#            QAF), expQFF), uF)
#        W1A = qml.dW(roots[i], tres, QAF, QFF, QFA, kA, kF)
#        denom = -roots[i] * np.dot(np.dot(rowA[i], W1A), colA[i])
#        areas[i] = nom / denom

    return areas

def exact_pdf(t, tres, roots, areas, eigvals, gamma00, gamma10, gamma11):
    r"""
    Calculate exponential probabolity density function with exact solution for
    missed events correction (Eq. 21, HJC92).

    .. math::
       :nowrap:

       \begin{align*}
       f(t) =
       \begin{cases}
       f_0(t)                          & \text{for}\; 0 \leq t \leq t_\text{res} \\
       f_0(t) - f_1(t - t_\text{res})  & \text{for}\; t_\text{res} \leq t \leq 2 t_\text{res}
       \end{cases}
       \end{align*}

    Parameters
    ----------
    t : float
        Time.
    tres : float
        Time resolution (dead time).
    roots : array_like, shape (k,)
    areas : array_like, shape (k,)
    eigvals : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    gama00, gama10, gama11 : lists of floats
        Coeficients for the exact open/shut time pdf.

    Returns
    -------
    f : float
    """

    if t < tres:
        f = 0
    elif ((tres < t) and (t < (2 * tres))):
        f = qml.f0((t - tres), eigvals, gamma00)
    elif ((tres * 2) < t) and (t < (3 * tres)):
        f = (qml.f0((t - tres), eigvals, gamma00) -
            qml.f1((t - 2 * tres), eigvals, gamma10, gamma11))
    else:
        f = pdfs.expPDF(t - tres, -1 / roots, areas)
    return f

def exact_mean_open_shut_time(mec, tres):
    """
    Calculate exact mean open or shut time from HJC probability density
    function.

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    QAA : array_like, shape (kA, kA)
    QFF : array_like, shape (kF, kF)
    QAF : array_like, shape (kA, kF)
        QAA, QFF, QAF - submatrices of Q.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.
    GAF : array_like, shape (kA, kB)
    GFA : array_like, shape (kB, kA)
        GAF, GFA- transition probabilities

    Returns
    -------
    mean : float
        Apparent mean open/shut time.
    """
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    expQFF = qml.expQt(mec.QFF, tres)
    expQAA = qml.expQt(mec.QAA, tres)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)

    phiA = qml.phiHJC(eGAF, eGFA, mec.kA)
    phiF = qml.phiHJC(eGFA, eGAF, mec.kF)
    QexpQF = np.dot(mec.QAF, expQFF)
    QexpQA = np.dot(mec.QFA, expQAA)
    DARS = qml.dARSdS(tres, mec.QAA, mec.QFF, GAF, GFA, expQFF, mec.kA, mec.kF)
    DFRS = qml.dARSdS(tres, mec.QFF, mec.QAA, GFA, GAF, expQAA, mec.kF, mec.kA)
    uF, uA = np.ones((mec.kF, 1)), np.ones((mec.kA, 1))
    # meanOpenTime = tres + phiA * DARS * QexpQF * uF
    meanA = tres + np.dot(phiA, np.dot(np.dot(DARS, QexpQF), uF))[0]
    meanF = tres + np.dot(phiF, np.dot(np.dot(DFRS, QexpQA), uA))[0]

    return meanA, meanF


def exact_mean_time(tres, QAA, QFF, QAF, kA, kF, GAF, GFA):
    """
    Calculate exact mean open or shut time from HJC probability density
    function.

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    QAA : array_like, shape (kA, kA)
    QFF : array_like, shape (kF, kF)
    QAF : array_like, shape (kA, kF)
        QAA, QFF, QAF - submatrices of Q.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.
    GAF : array_like, shape (kA, kB)
    GFA : array_like, shape (kB, kA)
        GAF, GFA- transition probabilities

    Returns
    -------
    mean : float
        Apparent mean open/shut time.
    """
    
    expQFF = qml.expQt(QFF, tres)
    expQAA = qml.expQt(QAA, tres)
    eGAF = qml.eGs(GAF, GFA, kA, kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, kF, kA, expQAA)

    phiA = qml.phiHJC(eGAF, eGFA, kA)
    QexpQF = np.dot(QAF, expQFF)
    DARS = qml.dARSdS(tres, QAA, QFF,
        GAF, GFA, expQFF, kA, kF)
    uF = np.ones((kF, 1))
    # meanOpenTime = tres + phiA * DARS * QexpQF * uF
    mean = tres + np.dot(phiA, np.dot(np.dot(DARS, QexpQF), uF))[0]

    return mean

def exact_GAMAxx(mec, tres, open):
    """
    Calculate gama coeficients for the exact open time pdf (Eq. 3.22, HJC90).

    Parameters
    ----------
    tres : float
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    open : bool
        True for open time pdf and False for shut time pdf.

    Returns
    -------
    eigen : ndarray, shape (k,)
        Eigenvalues of -Q matrix.
    gama00, gama10, gama11 : ndarrays
        Constants for the exact open/shut time pdf.
    """

    expQFF = qml.expQt(mec.QII, tres)
    expQAA = qml.expQt(mec.QAA, tres)
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kI)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kI, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kI, mec.kA, expQAA)
    eigs, A = qml.eigs_sorted(-mec.Q)

    if open:
        phi = qml.phiHJC(eGAF, eGFA, mec.kA)
        eigen, Z00, Z10, Z11 = qml.Zxx(mec.Q, eigs, A, mec.kA,
            mec.QII, mec.QAI, mec.QIA, expQFF, open)
        u = np.ones((mec.kI,1))
    else:
        phi = qml.phiHJC(eGFA, eGAF, mec.kI)
        eigen, Z00, Z10, Z11 = qml.Zxx(mec.Q, eigs, A, mec.kA,
            mec.QAA, mec.QIA, mec.QAI, expQAA, open)
        u = np.ones((mec.kA, 1))

    gama00 = (np.dot(np.dot(phi, Z00), u)).T[0]
    gama10 = (np.dot(np.dot(phi, Z10), u)).T[0]
    gama11 = (np.dot(np.dot(phi, Z11), u)).T[0]

    return eigen, gama00, gama10, gama11

def likelihood(theta, opts):
    """
    Calculate likelihood for a series of open and shut times using ideal
    probability density functions.
    """

    mec = opts['mec']
    conc = opts['conc']
    bursts = opts['data']

    #mec.set_rateconstants(np.exp(theta))
    mec.theta_unsqueeze(np.exp(theta))
    mec.set_eff('c', conc)

    startB = qml.phiA(mec)
    endB = np.ones((mec.kF, 1))

    loglik = 0
    for ind in bursts:
        burst = bursts[ind]
        grouplik = startB
        for i in range(len(burst)):
            t = burst[i]
            if i % 2 == 0: # open time
                GAFt = qml.iGt(t, mec.QAA, mec.QAF)
            else: # shut
                GAFt = qml.iGt(t, mec.QFF, mec.QFA)
            grouplik = np.dot(grouplik, GAFt)
            if grouplik.max() > 1e50:
                grouplik = grouplik * 1e-100
                print ('grouplik was scaled down')
        grouplik = np.dot(grouplik, endB)
        loglik += log(grouplik[0])

    newrates = np.log(mec.theta())
    return -loglik, newrates

def HJClik(theta, opts):
    """
    Calculate likelihood for a series of open and shut times using HJC missed
    events probability density functions (first two dead time intervals- exact
    solution, then- asymptotic).

    Lik = phi * eGAF(t1) * eGFA(t2) * eGAF(t3) * ... * eGAF(tn) * uF
    where t1, t3,..., tn are open times; t2, t4,..., t(n-1) are shut times.

    Gaps > tcrit are treated as unusable (e.g. contain double or bad bit of
    record, or desens gaps that are not in the model, or gaps so long that
    next opening may not be from the same channel). However this calculation
    DOES assume that all the shut times predicted by the model are present
    within each group. The series of multiplied likelihoods is terminated at
    the end of the opening before an unusable gap. A new series is then
    started, using appropriate initial vector to give Lik(2), ... At end
    these are multiplied to give final likelihood.

    Parameters
    ----------
    theta : array_like
        Guesses.
    bursts : dictionary
        A dictionary containing lists of open and shut intervals.
    opts : dictionary
        opts['mec'] : instance of type Mechanism
        opts['tres'] : float
            Time resolution (dead time).
        opts['tcrit'] : float
            Ctritical time interval.
        opts['isCHS'] : bool
            True if CHS vectors should be used (Eq. 5.7, CHS96).

    Returns
    -------
    loglik : float
        Log-likelihood.
    newrates : array_like
        Updated rates/guesses.
    """
    # TODO: Errors.

    mec = opts['mec']
    conc = opts['conc']
    tres = opts['tres']
    tcrit = opts['tcrit']
    is_chsvec = opts['isCHS']
    bursts = opts['data']

    mec.theta_unsqueeze(np.exp(theta))
    mec.set_eff('c', conc)

    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    expQFF = qml.expQt(mec.QFF, tres)
    expQAA = qml.expQt(mec.QAA, tres)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)
    phiF = qml.phiHJC(eGFA, eGAF, mec.kF)
    startB = qml.phiHJC(eGAF, eGFA, mec.kA)
    endB = np.ones((mec.kF, 1))

    eigen, A = qml.eigs(-mec.Q)
    Aeigvals, AZ00, AZ10, AZ11 = qml.Zxx(mec.Q, eigen, A, mec.kA, mec.QFF,
        mec.QAF, mec.QFA, expQFF, True)
    Aroots = asymptotic_roots(tres,
        mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)
    AR = qml.AR(Aroots, tres, mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)
    Feigvals, FZ00, FZ10, FZ11 = qml.Zxx(mec.Q, eigen, A, mec.kA, mec.QAA,
        mec.QFA, mec.QAF, expQAA, False)
    Froots = asymptotic_roots(tres,
        mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA)
    FR = qml.AR(Froots, tres, mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA)

    if is_chsvec:
        startB, endB = qml.CHSvec(Froots, tres, tcrit,
            mec.QFA, mec.kA, expQAA, phiF, FR)

    loglik = 0
    for ind in range(len(bursts)):
        burst = bursts[ind]
        grouplik = startB
        for i in range(len(burst)):
            t = burst[i]
            if i % 2 == 0: # open time
                eGAFt = qml.eGAF(t, tres, Aeigvals, AZ00, AZ10, AZ11, Aroots,
                AR, mec.QAF, expQFF)
            else: # shut
                eGAFt = qml.eGAF(t, tres, Feigvals, FZ00, FZ10, FZ11, Froots,
                FR, mec.QFA, expQAA)
            grouplik = np.dot(grouplik, eGAFt)
            if grouplik.max() > 1e50:
                grouplik = grouplik * 1e-100
                #print 'grouplik was scaled down'
        grouplik = np.dot(grouplik, endB)
        try:
            loglik += log(grouplik[0])
        except:
            print ('HJClik: Warning: likelihood has been set to 0')
            print ('likelihood=', grouplik[0])
            print ('rates=', mec.unit_rates())
            loglik = 0
            break

    newrates = np.log(mec.theta())
    return -loglik, newrates

def corr_variance_A(phiA, QAA, kA):
    """
    Calculate variance of open (shut) time according Eq. 2.6 (CH87).
    To calculate variance of shut time function should be called with
    parameters (phiF, QFF, kF).

    Parameters
    ----------
    phiA : array_like, shape (1, kA)
        Initial vector for openings (shuttings).
    QAA : array_like, shape (kA, kA)
        AA submatrix of Q.
    kA : int
        Number of open (shut) states.

    Returns
    -------
    var : float
        Variance.
    """

    uA = np.ones((kA))[:,np.newaxis]
    I = np.eye(kA)
    invQAA = -nplin.inv(QAA)
    M = 2 * I - np.dot(uA, phiA)
    row = np.dot(phiA, invQAA)
    col = np.dot(invQAA, uA)
    var = np.dot(np.dot(row, M), col)[0,0]
    return var

def corr_covariance_A(lag, phiA, QAA, XAA, kA):
    """
    Calculate covariance of open (shut) time according CH87.
    To calculate covariance of shut time function should be called with
    parameters (phiF, QFF, XFF, kF).

    Parameters
    ----------
    lag : int
        Lag.
    phiA : array_like, shape (1, kA)
        Initial vector for openings (shuttings).
    QAA : array_like, shape (kA, kA)
        AA submatrix of Q.
    XAA : array_like, shape (kA, kA)
        Product GAF * GFA.
    kA : int
        Number of open (shut) states.

    Returns
    -------
    covar : float
        Covariance.
    """
    
    Xn = qml.Qpow(XAA, lag)
    uA = np.ones((kA))[:,np.newaxis]
    invQAA = -nplin.inv(QAA)
    M2 = Xn - np.dot(uA, phiA)
    row = np.dot(phiA, invQAA)
    col = np.dot(invQAA, uA)
    covar = np.dot(np.dot(row, M2), col)[0,0]
    return covar

def corr_covariance_AF(lag, phiA, QAA, QFF, XAA, GAF, kA, kF):
    """
    Calculate covariance of open and nth shut times according CH87.
    
    Parameters
    ----------
    lag : int
        Lag.
    phiA : array_like, shape (1, kA)
        Initial vector for openings.
    QAA : array_like, shape (kA, kA)
        AA submatrix of Q.
    QFF : array_like, shape (kF, kF)
        FF submatrix of Q.
    XAA : array_like, shape (kA, kA)
        Product GAF * GFA.
    GAF : array_like, shape (kA, kF)
        GAF matrix.
    kA : int
        Number of open states.
    kF : int
        Number of shut states.

    Returns
    -------
    covar : float
        Covariance.
    """

    Xn = qml.Qpow(XAA, lag-1)
    uA, uF = np.ones((kA))[:,np.newaxis], np.ones((kF))[:,np.newaxis]
    invQAA, invQFF = -nplin.inv(QAA), -nplin.inv(QFF)
    MAF = Xn - np.dot(uA, phiA)
    row = np.dot(phiA, invQAA)
    col = np.dot(np.dot(GAF, invQFF),uF)
    covar = np.dot(np.dot(row, MAF), col)[0,0]
    return covar

def corr_decay_amplitude_A(phiA, QAA, XAA, kA):
    """
    Calculate scalar coefficients for correlation coefficien decay (Eq. 2.11,
    CH83).

    Parameters
    ----------

    Returns
    -------
    w : ndarray, shape (1, k)
    eigs : ndarray, shape (1, k)
    """
    
    varA = corr_variance_A(phiA, QAA, kA)
    eigs, A = qml.eigs(XAA)

    uA = np.ones((kA))[:,np.newaxis]
    invQAA = -nplin.inv(QAA)
    row = np.dot(phiA, invQAA)
    col = np.dot(invQAA, uA)

    ncA = np.linalg.matrix_rank(XAA) - 1
    w = np.zeros((ncA))
    n = 0
    for i in range(kA):
        if fabs(eigs[i]) > 1e-12 and fabs(eigs[i] - 1) > 1e-12:
            w[n] = np.dot(np.dot(row, A[i, :, :]), col)[0,0] / varA
            n += 1
    return w, eigs

def corr_limit_A(phiA, QAA, AXAA, eigXAA, kA):

    uA = np.ones((kA))[:,np.newaxis]
    invQAA = -nplin.inv(QAA)
    row = np.dot(phiA, invQAA)
    col = np.dot(invQAA, uA)
    # M = sum_{i=0}^{kA-2} AXAA[i] * eigXAA[i] / (1 - eigXAA[i])
    coeff = eigXAA[:-1] / (1.0 - eigXAA[:-1])            # (kA-1,)
    M = np.sum(AXAA[:-1] * coeff[:, np.newaxis, np.newaxis], axis=0)
    cor = np.dot(np.dot(row, M), col)[0,0]
    return cor

def adjacent_open_to_shut_range_mean(u1, u2, QAA, QAF, QFF, QFA, phiA):
    """
    Calculate mean (ideal- no missed events) open times adjacent to a 
    specified shut time range.

    Parameters
    ----------
    u1, u2 : floats
        Shut time range.
    QAA, QAF, QFF, QFA : array_like
        Submatrices of Q.
    phiA : array_like, shape (1, kA)
        Initial vector for openings

    Returns
    -------
    m : float
        Mean open time.
    """
    
    kA = QAA.shape[0]
    uA = np.ones((kA))[:,np.newaxis]
    invQAA, invQFF = -nplin.inv(QAA), nplin.inv(QFF)
    expQFFr = qml.expQt(QFF, u2) - qml.expQt(QFF, u1)
    col = np.dot(np.dot(np.dot(np.dot(QAF, invQFF), expQFFr), QFA), uA)
    row1 = np.dot(phiA, qml.Qpow(invQAA, 2))
    row2 = np.dot(phiA, invQAA)
    m = np.dot(row1, col)[0, 0] / np.dot(row2, col)[0, 0]
    return m

def HJC_dependency(top, tsh, tres, Q, QAA, QAF, QFF, QFA):
    """
    Calculate normalised joint distribution (CHS96, Eq. 3.22) of an open time
    and the following shut time as proposed by Magleby & Song 1992. 
    
    Parameters
    ----------
    top, tsh : array_like of floats
        Open and shut tims.
    tres : float
        Time resolution.
    Q : array, shape (k,k)
        Q matrix. 
    QAA, QAF, QFF, QFA : array_like
        Submatrices of Q.

    Returns
    -------
    dependency : ndarray
    """
    
    kA, kF = QAA.shape[0], QFF.shape[0]
    uA = np.ones((kA))[:,np.newaxis]
    uF = np.ones((kF))[:,np.newaxis]
    expQFF = qml.expQt(QFF, tres)
    expQAA = qml.expQt(QAA, tres)
    GAF, GFA = qml.iGs(Q, kA, kF)
    eGAF = qml.eGs(GAF, GFA, kA, kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, kF, kA, expQAA)
    phiA = qml.phiHJC(eGAF, eGFA, kA)
    phiF = qml.phiHJC(eGFA, eGAF, kF)
    eigs, A = qml.eigs(-Q)
    Feigvals, FZ00, FZ10, FZ11 = qml.Zxx(Q, eigs, A, kA, QAA, QFA, QAF, expQAA, False)
    Froots = asymptotic_roots(tres, QFF, QAA, QFA, QAF, kF, kA)
    FR = qml.AR(Froots, tres, QFF, QAA, QFA, QAF, kF, kA)
    Aeigvals, AZ00, AZ10, AZ11 = qml.Zxx(Q, eigs, A, kA, QFF, QAF, QFA, expQFF, True)
    Aroots = asymptotic_roots(tres, QAA, QFF, QAF, QFA, kA, kF)
    AR = qml.AR(Aroots, tres, QAA, QFF, QAF, QFA, kA, kF)

    dependency = np.zeros((top.shape[0], tsh.shape[0]))
    
    for i in range(top.shape[0]):
        eGAFt = qml.eGAF(top[i], tres, Aeigvals, AZ00, AZ10, AZ11, Aroots,
                AR, QAF, expQFF)
        fo = np.dot(np.dot(phiA, eGAFt), uF)[0]
        
        for j in range(tsh.shape[0]):
            eGFAt = qml.eGAF(tsh[j], tres, Feigvals, FZ00, FZ10, FZ11, Froots,
                FR, QFA, expQAA)
            fs = np.dot(np.dot(phiF, eGFAt), uA)[0]
            fos = np.dot(np.dot(np.dot(phiA, eGAFt), eGFAt), uA)[0]
            dependency[i, j] = (fos - (fo * fs)) / (fo * fs)
    return dependency

def HJC_adjacent_mean_open_to_shut_time_pdf(sht, tres, Q, QAA, QAF, QFF, QFA):
    """
    Calculate theoretical HJC (with missed events correction) mean open time
    given previous/next gap length (continuous function; CHS96 Eq.3.5). 

    Parameters
    ----------
    sht : array of floats
        Shut time interval.
    tres : float
        Time resolution.
    Q : array, shape (k,k)
        Q matrix.
    QAA, QAF, QFF, QFA : array_like
        Submatrices of Q.

    Returns
    -------
    mp : ndarray of floats
        Mean open time given previous gap length.
    mn : ndarray of floats
        Mean open time given next gap length.
    """
    
    kA, kF = QAA.shape[0], QFF.shape[0]
    uA = np.ones((kA))[:,np.newaxis]
    uF = np.ones((kF))[:,np.newaxis]
    expQFF = qml.expQt(QFF, tres)
    expQAA = qml.expQt(QAA, tres)
    GAF, GFA = qml.iGs(Q, kA, kF)
    eGAF = qml.eGs(GAF, GFA, kA, kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, kF, kA, expQAA)
    phiA = qml.phiHJC(eGAF, eGFA, kA)
    phiF = qml.phiHJC(eGFA, eGAF, kF)
    DARS = qml.dARSdS(tres, QAA, QFF, GAF, GFA, expQFF, kA, kF)
    eigs, A = qml.eigs(-Q)
    Feigvals, FZ00, FZ10, FZ11 = qml.Zxx(Q, eigs, A, kA, QAA, QFA, QAF, expQAA, False)
    Froots = asymptotic_roots(tres, QFF, QAA, QFA, QAF, kF, kA)
    FR = qml.AR(Froots, tres, QFF, QAA, QFA, QAF, kF, kA)
    Q1 = np.dot(np.dot(DARS, QAF), expQFF)
    col1 = np.dot(Q1, uF)
    row1 = np.dot(phiA, Q1)
    
    mp = []
    mn = []
    for t in sht:
        eGFAt = qml.eGAF(t, tres, Feigvals, FZ00, FZ10, FZ11, Froots,
                    FR, QFA, expQAA)
        denom = np.dot(np.dot(phiF, eGFAt), uA)[0]
        nom1 = np.dot(np.dot(phiF, eGFAt), col1)[0]
        nom2 = np.dot(np.dot(row1, eGFAt), uA)[0]
        mp.append(nom1 / denom)
        mn.append(nom2 / denom)
    
    return np.array(mp), np.array(mn)

def adjacent_open_to_shut_range_pdf_components(u1, u2, QAA, QAF, QFF, QFA, phiA):
    """
    Calculate time constants and areas for an ideal (no missed events)
    exponential probability density function of open times adjacent to a 
    specified shut time range.

    Parameters
    ----------
    t : float
        Time (sec).
    QAA : array_like, shape (kA, kA)
        Submatrix of Q.
    phiA : array_like, shape (1, kA)
        Initial vector for openings

    Returns
    -------
    taus : ndarray, shape(k, 1)
        Time constants.
    areas : ndarray, shape(k, 1)
        Component relative areas.
    """

    kA = QAA.shape[0]
    uA = np.ones((kA))[:,np.newaxis]
    invQAA, invQFF = -nplin.inv(QAA), nplin.inv(QFF)
    expQFFr = qml.expQt(QFF, u2) - qml.expQt(QFF, u1)
    col = np.dot(np.dot(np.dot(np.dot(QAF, invQFF), expQFFr), QFA), uA)
    w = np.zeros(kA)
    eigs, A = qml.eigs(-QAA)
    row = np.dot(phiA, invQAA)
    den = np.dot(row, col)[0, 0]
    #TODO: remove 'for'
    for i in range(kA):
        w[i] = (np.dot(np.dot(phiA, A[i]), col) / den).item()
    return eigs, w

def simulate_intervals(mec, tres, state, opamp=5, nintmax=5000):
    """

    """
    picum = np.cumsum(transition_probability(mec.Q), axis=1)
    tmean = -1 / mec.Q.diagonal() # in s
    ints = [random.expovariate(1 / tmean[state])]
    amps = [opamp] if state < mec.kA else [0]
    while len(ints) < nintmax:
        state, t, a = next_state(state, picum, tmean, mec.kA, opamp)
        if t < tres or a == amps[-1]:
            ints[-1] += t
        else:
            ints.append(t)
            amps.append(a)
    return np.array(ints), np.array(amps), np.zeros((len(ints)), dtype='b')

def next_state(present, picum, tmean, kA, opamp):
    """
    Get next state, its lifetime and amplitude.
    """
    possible = np.nonzero(picum[present] >= random.random())[0]
    next = np.delete(possible, np.where(possible == present))[0]
    t = random.expovariate(1 / tmean[next])
    a = opamp if next < kA else 0
    return next, t, a

def printout_occupancies(mec, tres):
    """
    """

    str = ('\n\n\n*******************************************\n\n' +
        'Open\tEquilibrium\tMean life\tMean latency (ms)\n' +
        'state\toccupancy\t(ms)\tto next shutting\n' +
        '\t\t\tgiven start in this state\n')

    pinf = qml.pinf(mec.Q)

    for i in range(mec.k):
        if i == 0:
            mean_life_A = ideal_subset_mean_life_time(mec.Q, 1, mec.kA)
            str += ('Subset A ' +
                '\t{0:.5g}'.format(np.sum(pinf[:mec.kA])) +
                '\t{0:.5g}\n'.format(mean_life_A * 1000))
        if i == mec.kA:
            mean_life_B = ideal_subset_mean_life_time(mec.Q, mec.kA + 1, mec.kE)
            str += ('\nShut\tEquilibrium\tMean life\tMean latency (ms)\n' +
                'state\toccupancy\t(ms)\tto next opening\n' +
                '\t\t\tgiven start in this state\n' +
                'Subset B ' +
                '\t{0:.5g}'.format(np.sum(pinf[mec.kA : mec.kE])) +
                '\t{0:.5g}\n'.format(mean_life_B * 1000))
        if i == mec.kE:
            mean_life_C = ideal_subset_mean_life_time(mec.Q, mec.kE + 1, mec.kG)
            str += ('\nSubset C ' +
                '\t{0:.5g}'.format(np.sum(pinf[mec.kE : mec.kG])) +
                '\t{0:.5g}\n'.format(mean_life_C * 1000))
        if i == mec.kG:
            mean_life_D = ideal_subset_mean_life_time(mec.Q, mec.kG + 1, mec.k)
            str += ('\nSubset D ' +
                '\t{0:.5g}'.format(np.sum(pinf[mec.kG : mec.k])) +
                '\t{0:.5g}\n'.format(mean_life_D * 1000))

        mean = ideal_mean_latency_given_start_state(mec, i+1)
        str += ('{0:d}'.format(i+1) +
            '\t{0:.5g}'.format(pinf[i]) +
            '\t{0:.5g}'.format(-1 / mec.Q[i,i] * 1000) +
            '\t{0:.5g}\n'.format(mean * 1000))

    expQFF = qml.expQt(mec.QFF, tres)
    expQAA = qml.expQt(mec.QAA, tres)
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)
    phiA = qml.phiHJC(eGAF, eGFA, mec.kA)
    phiF = qml.phiHJC(eGFA, eGAF, mec.kF)

    str += ('\n\nInitial vector for HJC openings phiOp =\n')
    for i in range(phiA.shape[0]):
        str += ('\t{0:.5g}'.format(phiA[i]))
    str += ('\nInitial vector for ideal openings phiOp =\n')
    phiAi = qml.phiA(mec)
    for i in range(phiA.shape[0]):
        str += ('\t{0:.5g}'.format(phiAi[i]))
    str += ('\nInitial vector for HJC shuttings phiSh =\n')
    for i in range(phiF.shape[0]):
        str += ('\t{0:.5g}'.format(phiF[i]))
    str += ('\nInitial vector for ideal shuttings phiSh =\n')
    phiFi = qml.phiF(mec)
    for i in range(phiF.shape[0]):
        str += ('\t{0:.5g}'.format(phiFi[i]))
    str += '\n'
    
    return str

def printout_distributions(mec, tres, eff='c'):
    """

    """

    str = '\n*******************************************\n'
    GAI, GIA = qml.iGs(mec.Q, mec.kA, mec.kI)
    # OPEN TIME DISTRIBUTIONS
    open = True
    # Ideal pdf
    eigs, w = ideal_dwell_time_pdf_components(mec.QAA,
        qml.phiA(mec))
    str += 'IDEAL OPEN TIME DISTRIBUTION\n'
    str += pdfs.expPDF_printout(eigs, w)

    # Asymptotic pdf
    #roots = asymptotic_roots(mec, tres, open)
    roots = asymptotic_roots(tres,
        mec.QAA, mec.QII, mec.QAI, mec.QIA, mec.kA, mec.kI)
    #areas = asymptotic_areas(mec, tres, roots, open)
    areas = asymptotic_areas(tres, roots,
        mec.QAA, mec.QII, mec.QAI, mec.QIA, mec.kA, mec.kI, GAI, GIA)
    str += '\nASYMPTOTIC OPEN TIME DISTRIBUTION\n'
    str += 'term\ttau (ms)\tarea (%)\trate const (1/sec)\n'
    for i in range(mec.kA):
        str += ('{0:d}'.format(i+1) +
        '\t{0:.5g}'.format(-1.0 / roots[i] * 1000) +
        '\t{0:.5g}'.format(areas[i] * 100) +
        '\t{0:.5g}\n'.format(- roots[i]))
    areast0 = np.zeros(mec.kA)
    for i in range(mec.kA):
        areast0[i] = areas[i] * np.exp(- tres * roots[i])
    areast0 = areast0 / np.sum(areast0)
    str += ('Areas for asymptotic pdf renormalised for t=0 to\
    infinity (and sum=1), so areas can be compared with ideal pdf.\n')
    for i in range(mec.kA):
        str += ('{0:d}'.format(i+1) +
        '\t{0:.5g}\n'.format(areast0[i] * 100))
    mean = exact_mean_time(tres,
            mec.QAA, mec.QII, mec.QAI, mec.kA, mec.kI, GAI, GIA)
    str += ('Mean open time (ms) = {0:.5g}\n'.format(mean * 1000))

    # Exact pdf
    eigvals, gamma00, gamma10, gamma11 = exact_GAMAxx(mec, tres, open)
    str += ('\nEXACT OPEN TIME DISTRIBUTION\n')
    str += ('eigen\tg00(m)\tg10(m)\tg11(m)\n')
    for i in range(mec.k):
        str += ('{0:.5g}'.format(eigvals[i]) +
        '\t{0:.5g}'.format(gamma00[i]) +
        '\t{0:.5g}'.format(gamma10[i]) +
        '\t{0:.5g}\n'.format(gamma11[i]))

    str += ('\n\n*******************************************\n')
    # SHUT TIME DISTRIBUTIONS
    open = False
    # Ideal pdf
    eigs, w = ideal_dwell_time_pdf_components(mec.QII, qml.phiF(mec))
    str += ('IDEAL SHUT TIME DISTRIBUTION\n')
    str += pdfs.expPDF_printout(eigs, w)

    # Asymptotic pdf
    #roots = asymptotic_roots(mec, tres, open)
    roots = asymptotic_roots(tres,
        mec.QII, mec.QAA, mec.QIA, mec.QAI, mec.kI, mec.kA)
    #areas = asymptotic_areas(mec, tres, roots, open)
    areas = asymptotic_areas(tres, roots,
        mec.QII, mec.QAA, mec.QIA, mec.QAI, mec.kI, mec.kA, GIA, GAI)
    str += ('\nASYMPTOTIC SHUT TIME DISTRIBUTION\n')
    str += ('term\ttau (ms)\tarea (%)\trate const (1/sec)\n')
    for i in range(mec.kI):
        str += ('{0:d}'.format(i+1) +
        '\t{0:.5g}'.format(-1.0 / roots[i] * 1000) +
        '\t{0:.5g}'.format(areas[i] * 100) +
        '\t{0:.5g}\n'.format(- roots[i]))
    areast0 = np.zeros(mec.kI)
    for i in range(mec.kI):
        areast0[i] = areas[i] * np.exp(- tres * roots[i])
    areast0 = areast0 / np.sum(areast0)
    str += ('Areas for asymptotic pdf renormalised for t=0 to\
    infinity (and sum=1), so areas can be compared with ideal pdf.\n')
    for i in range(mec.kI):
        str += ('{0:d}'.format(i+1) +
        '\t{0:.5g}\n'.format(areast0[i] * 100))
    mean = exact_mean_time(tres,
            mec.QII, mec.QAA, mec.QIA, mec.kI, mec.kA, GIA, GAI)
    str += ('Mean shut time (ms) = {0:.6f}\n'.format(mean * 1000))

    # Exact pdf
    eigvals, gamma00, gamma10, gamma11 = exact_GAMAxx(mec, tres, open)
    str += ('\nEXACT SHUT TIME DISTRIBUTION\n' + 
        'eigen\tg00(m)\tg10(m)\tg11(m)\n')
    for i in range(mec.k):
        str += ('{0:.5g}'.format(eigvals[i]) +
        '\t{0:.5g}'.format(gamma00[i]) +
        '\t{0:.5g}'.format(gamma10[i]) +
        '\t{0:.5g}\n'.format(gamma11[i]))

    # Transition probabilities
    pi = transition_probability(mec.Q)
    str += ('\nProbability of transitions regardless of time:\n')
    for i in range(mec.k):
        str1 = '['
        for j in range(mec.k):
            str1 += '{0:.4g}\t'.format(pi[i,j])
        str1 += ']\n'
        str += str1

    # Transition frequency
    f = transition_frequency(mec.Q)
    str += ('\nFrequency of transitions (per second):\n')
    for i in range(mec.k):
        str1 = '['
        for j in range(mec.k):
            str1 += '{0:.4g}\t'.format(f[i,j])
        str1 += ']\n'
        str += str1
        
    return str

def transition_probability(Q):
    """
    """
    k = Q.shape[0]
    pi = Q.copy()
    for i in range(k):
        pi[i] = pi[i] / -Q[i,i]
        pi[i,i] = 0
    return pi

def transition_frequency(Q):
    """
    """
    k = Q.shape[0]
    pinf = qml.pinf(Q)
    f = Q.copy().transpose()
    for i in range(k):
        f[i] = f[i] * pinf
        f[i,i] = 0
    return f.transpose()

def correlation_coefficient(cov, var1, var2):
    """
    Correlation coefficient.

    Parameters
    ----------
    cov : float
        Covariance.
    var1, var2 : floats
        Variances.

    Returns
    -------
    ro : float
        Correlation coefficient.
    """

    ro = cov / sqrt(var1 * var2)
    return ro


def sortShell2(vals, simp):
    """
    Shell sort using Shell's (original) gap sequence: n/2, n/4, ..., 1.
    """
    n = np.size(vals)
    gap = n // 2
    while gap > 0:
         # do the insertion sort
         for i in range(gap, n):
             val = vals[i]
             tsimp = simp[i]
             j = i
             while j >= gap and vals[j - gap] > val:
                 vals[j] = vals[j - gap]
                 simp[j] = simp[j - gap]
                 j -= gap
             vals[j] = val
             simp[j] = tsimp
         gap //= 2
    return vals, simp

def printout_tcrit(mec):
    """
    Output calculations based on division into bursts by critical time (tcrit).

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    output : output device
        Default device: sys.stdout
    """

    str = ('\n\n*******************************************\n' +
        'CALCULATIONS BASED ON DIVISION INTO BURSTS BY' +
        ' tcrit- CRITICAL TIME.\n')
    # Ideal shut time pdf
    eigs, w = ideal_dwell_time_pdf_components(mec.QII, qml.phiF(mec))
    str += ('\nIDEAL SHUT TIME DISTRIBUTION\n')
    str += pdfs.expPDF_printout(eigs, w)
    taus = 1 / eigs
    areas = w /eigs
    taus, areas = sortShell2(taus, areas)

    comps = taus.shape[0]-1
    tcrits = np.empty((3, comps))
    for i in range(comps):
        str += ('\nCritical time between components {0:d} and {1:d}\n'.
            format(i+1, i+2) + '\nEqual % misclassified (DC criterion)\n')
        try:
            tcrit = so.bisect(pdfs.expPDF_tcrit_DC,
                taus[i], taus[i+1], args=(taus, areas, i+1))
            enf, ens, pf, ps = pdfs.expPDF_misclassified(tcrit, taus, areas, i+1)
            str += pdfs.expPDF_misclassified_printout(tcrit, enf, ens, pf, ps)
        except:
            str += ('Bisection with DC criterion failed.\n')
            tcrit = None
        tcrits[0, i] = tcrit
        
        str += ('\nEqual # misclassified (Clapham & Neher criterion)\n')
        try:
            tcrit = so.bisect(pdfs.expPDF_tcrit_CN,
                taus[i], taus[i+1], args=(taus, areas, i+1))
            enf, ens, pf, ps = pdfs.expPDF_misclassified(tcrit, taus, areas, i+1)
            str += pdfs.expPDF_misclassified_printout(tcrit, enf, ens, pf, ps)
        except:
            str += ('Bisection with Clapham & Neher criterion failed.\n')
            tcrit = None
        tcrits[1, i] = tcrit
        
        str += ('\nMinimum total # misclassified (Jackson et al criterion)')
        try:
            tcrit = so.bisect(pdfs.expPDF_tcrit_Jackson,
                taus[i], taus[i+1], args=(taus, areas, i+1))
            enf, ens, pf, ps = pdfs.expPDF_misclassified(tcrit, taus, areas, i+1)
            str += pdfs.expPDF_misclassified_printout(tcrit, enf, ens, pf, ps)
        except:
            str += ('\nBisection with Jackson et al criterion failed.')
            tcrit = None
        tcrits[2, i] = tcrit
        
    str += ('\nSUMMARY of tcrit values:\n' +
        'Components  DC\tC&N\tJackson\n')
    for i in range(comps):
        str += ('{0:d} to {1:d} '.format(i+1, i+2) +
            '\t{0:.5g}'.format(tcrits[0, i] * 1000) +
            '\t{0:.5g}'.format(tcrits[1, i] * 1000) +
            '\t{0:.5g}\n'.format(tcrits[2, i] * 1000))
            
    return str

def printout_correlations(mec, output=sys.stdout, eff='c'):
    """

    """

    str = ('\n\n*************************************\n' +
        'CORRELATIONS\n')
    
    kA, kI = mec.kA, mec.kI
    str += ('kA, kF = {0:d}, {1:d}\n'.format(kA, kI))
    GAF, GFA = qml.iGs(mec.Q, kA, kI)
    rGAF, rGFA = np.linalg.matrix_rank(GAF), np.linalg.matrix_rank(GFA)
    str += ('Ranks of GAF, GFA = {0:d}, {1:d}\n'.format(rGAF, rGFA))
    XFF = np.dot(GFA, GAF)
    rXFF = np.linalg.matrix_rank(XFF)
    str += ('Rank of GFA * GAF = {0:d}\n'.format(rXFF))
    ncF = rXFF - 1
    eigXFF, AXFF = qml.eigs(XFF)
    str += ('Eigenvalues of GFA * GAF:\n')
    str1 = ''
    for i in range(kI):
        str1 += '\t{0:.5g}'.format(eigXFF[i])
    str += str1 + '\n'
    XAA = np.dot(GAF, GFA)
    rXAA = np.linalg.matrix_rank(XAA)
    str += ('Rank of GAF * GFA = {0:d}\n'.format(rXAA))
    ncA = rXAA - 1
    eigXAA, AXAA = qml.eigs(XAA)
    str += ('Eigenvalues of GAF * GFA:\n')
    str1 = ''
    for i in range(kA):
        str1 += '\t{0:.5g}'.format(eigXAA[i])
    str += str1 + '\n'
    phiA, phiF = qml.phiA(mec).reshape((1,kA)), qml.phiF(mec).reshape((1,kI))
    varA = corr_variance_A(phiA, mec.QAA, kA)
    varF = corr_variance_A(phiF, mec.QII, kI)
    
    #   open - open time correlations
    str += ('\n OPEN - OPEN TIME CORRELATIONS')
    str += ('Variance of open time = {0:.5g}\n'.format(varA))
    SDA = sqrt(varA)
    str += ('SD of all open times = {0:.5g} ms\n'.format(SDA * 1000))
    n = 50
    SDA_mean_n = SDA / sqrt(float(n))
    str += ('SD of means of {0:d} open times if'.format(n) + 
        'uncorrelated = {0:.5g} ms\n'.format(SDA_mean_n * 1000))
    covAtot = 0
    for i in range(1, n):
        covA = corr_covariance_A(i+1, phiA, mec.QAA, XAA, kA)
        ro = correlation_coefficient(covA, varA, varA)
        covAtot += (n - i) * ro * varA
    vtot = n * varA + 2. * covAtot
    actSDA = sqrt(vtot / (n * n))
    str += ('Actual SD of mean = {0:.5g} ms\n'.format(actSDA * 1000))
    pA = 100 * (actSDA - SDA_mean_n) / SDA_mean_n
    str += ('Percent difference as result of correlation = {0:.5g}\n'.
        format(pA))
    v2A = corr_limit_A(phiA, mec.QAA, AXAA, eigXAA, kA)
    pmaxA = 100 * (sqrt(1 + 2 * v2A / varA) - 1)
    str += ('Limiting value of percent difference for large n = {0:.5g}\n'.
        format(pmaxA))
    str += ('Correlation coefficients, r(k), for up to lag k = 5:\n')
    for i in range(5):
        covA = corr_covariance_A(i+1, phiA, mec.QAA, XAA, kA)
        ro = correlation_coefficient(covA, varA, varA)
        str += ('r({0:d}) = {1:.5g}\n'.format(i+1, ro))

    # shut - shut time correlations
    str += ('\n SHUT - SHUT TIME CORRELATIONS\n')
    str += ('Variance of shut time = {0:.5g}\n'.format(varF))
    SDF = sqrt(varF)
    str += ('SD of all shut times = {0:.5g} ms\n'.format(SDF * 1000))
    n = 50
    SDF_mean_n = SDF / sqrt(float(n))
    str += ('SD of means of {0:d} shut times if'.format(n) +
        'uncorrelated = {0:.5g} ms\n'.format(SDF_mean_n * 1000))
    covFtot = 0
    for i in range(1, n):
        covF = corr_covariance_A(i+1, phiF, mec.QII, XFF, kI)
        ro = correlation_coefficient(covF, varF, varF)
        covFtot += (n - i) * ro * varF
    vtotF = 50 * varF + 2. * covFtot
    actSDF = sqrt(vtotF / (50. * 50.))
    str += ('Actual SD of mean = {0:.5g} ms\n'.format(actSDF * 1000))
    pF = 100 * (actSDF - SDF_mean_n) / SDF_mean_n
    str += ('Percent difference as result of correlation = {0:.5g}\n'.
        format(pF))
    v2F = corr_limit_A(phiF, mec.QII, AXFF, eigXFF, kI)
    pmaxF = 100 * (sqrt(1 + 2 * v2F / varF) - 1)
    str += ('Limiting value of percent difference for large n = {0:.5g}\n'.
        format(pmaxF))
    str += ('Correlation coefficients, r(k), for up to k = 5 lags:\n')
    for i in range(5):
        covF = corr_covariance_A(i+1, phiF, mec.QII, XFF, kI)
        ro = correlation_coefficient(covF, varF, varF)
        str += ('r({0:d}) = {1:.5g}\n'.format(i+1, ro))

    # open - shut time correlations 
    str += ('\n OPEN - SHUT TIME CORRELATIONS\n')
    str += ('Correlation coefficients, r(k), for up to k= 5 lags:\n')
    for i in range(5):
        covAF = corr_covariance_AF(i+1, phiA, mec.QAA, mec.QII, XAA, GAF, kA, kI)
        ro = correlation_coefficient(covAF, varA, varF)
        str += ('r({0:d}) = {1:.5g}\n'.format(i+1, ro))
    return str
        
def printout_adjacent(mec, t1, t2):
    """

    """

    str = ('\n*************************************\n' +
        ' OPEN TIMES ADJACENT TO SPECIFIED SHUT TIME RANGE\n')
    
    kA = mec.kA
    phiA = qml.phiA(mec).reshape((1,kA))
    
    str += ('PDF of open times that precede shut times between {0:.3f}\
 and {1:.3f} ms\n'.format(t1 * 1000, t2 * 1000))
        
    eigs, w = adjacent_open_to_shut_range_pdf_components(t1, t2, 
        mec.QAA, mec.QAF, mec.QFF, mec.QFA, phiA)
    str += pdfs.expPDF_printout(eigs, w)

    mean = adjacent_open_to_shut_range_mean(t1, t2, 
        mec.QAA, mec.QAF, mec.QFF, mec.QFA, phiA)
    str += ('Mean from direct calculation (ms) = {0:.6f}\n'.format(mean * 1000))
    return str

