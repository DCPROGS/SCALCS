"""
Plotting utilities for single channel currents.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 23:01:09$"

import sys
import math

import numpy as np
try:
    from matplotlib import scale as mscale
    from matplotlib import transforms as mtransforms
    from matplotlib import ticker
except:
    raise ImportError("matplotlib module is missing")

import qmatlib as qml
import scalcslib as scl
import scburst
import popen
import pdfs

def Popen(mec, tres):
    """
    Calculate Popen curve parameters and data for Popen curve plot.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    c : ndarray of floats, shape (num of points,)
        Concentration in mikroM.
    pe : ndarray of floats, shape (num of points,)
        Open probability corrected for missed events.
    pi : ndarray of floats, shape (num of points,)
        Ideal open probability.
    """

    iEC50 = popen.EC50(mec, 0)   # in mikroM

    # Plot ideal and corrected Popen curves.
    cmin = iEC50 / 20
    cmax = iEC50 * 500
    log_start = int(np.log10(cmin)) - 1
    log_end = int(np.log10(cmax)) - 1
    points = 512

    c = np.logspace(log_start, log_end, points)
    pe = np.zeros(points)
    pi = np.zeros(points)
    for i in range(points):
        pe[i] = popen.Popen(mec, tres, c[i])
        pi[i] = popen.Popen(mec, 0, c[i])

    c = c * 1000000

    return c, pe, pi

def burst_length_pdf(mec, conditional=False, tmin=0.00001, tmax=1000, points=512):
    """
    Calculate the mean burst length and data for burst length distribution.

    Parameters
    ----------
    mec : instance of type Mechanism
    conditional : bool
        True if conditional distribution is plotted.
    tmin, tmax : floats
        Time range for burst length ditribution.
    points : int
        Number of points per plot.

    Returns
    -------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    fbst : ndarray of floats, shape (num of points)
        Burst length pdf.
    cfbrst : ndarray of floats, shape (num of open states, num of points)
        Conditional burst length pdf.
    """

    eigs, w = scburst.length_pdf_components(mec)
    tmax = 20 / min(eigs)
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    fbst = np.zeros(points)
    for i in range(points):
        fbst[i] = t[i] * scburst.length_pdf(mec, t[i])

    if conditional:
        cfbst = np.zeros((points, mec.kA))
        for i in range(points):
            cfbst[i] = t[i] * scburst.length_cond_pdf(mec, t[i])
        cfbrst = cfbst.transpose()
        return t * 1000, fbst, cfbrst

    t = t * 1000 # x axis in millisec

    return t, fbst

def burst_openings_pdf(mec, n, conditional=False):
    """
    Calculate the mean number of openings per burst and data for the
    distribution of openings per burst.

    Parameters
    ----------
    mec : instance of type Mechanism
    n  : int
        Number of openings.
    conditional : bool
        True if conditional distribution is plotted.

    Returns
    -------
    r : ndarray of ints, shape (num of points,)
        Number of openings per burst.
    Pr : ndarray of floats, shape (num of points,)
        Fraction of bursts.
    cPr : ndarray of floats, shape (num of open states, num of points)
        Fraction of bursts for conditional distribution.
    """

    r = np.arange(1, n+1)
    Pr = np.zeros(n)
    for i in range(n):
        Pr[i] = scburst.openings_distr(mec, r[i])

    if conditional:
        cPr = np.zeros((n, mec.kA))
        for i in range(n):
            cPr[i] = scburst.openings_cond_distr_depend_on_start_state(mec, r[i])
        cPr = cPr.transpose()

        return r, Pr, cPr

    return r, Pr

def burst_length_versus_conc_plot(mec, cmin, cmax):
    """
    Calculate data for the plot of burst length versus concentration.

    Parameters
    ----------
    mec : instance of type Mechanism
    cmin, cmax : float
        Range of concentrations in M.

    Returns
    -------
    c : ndarray of floats, shape (num of points,)
        Concentration in mikroM
    br : ndarray of floats, shape (num of points,)
        Mean burst length in millisec.
    brblk : ndarray of floats, shape (num of points,)
        Mean burst length in millisec corrected for fast block.
    """

    points = 100
    step = (cmax - cmin)/(points - 1)
    c = np.zeros(points)
    br = np.zeros(points)
    brblk = np.zeros(points)

    for i in range(points):
        c[i] = cmin + step * i
        mec.set_eff('c', c[i])
        br[i] = scburst.length_mean(mec)
        if mec.fastblk:
            brblk[i] = br[i] * (1 + c[i] / mec.KBlk)
        else:
            brblk[i] = br[i]
    c = c * 1000000 # x axis scale in mikroMoles
    br = br * 1000
    brblk= brblk * 1000

    return c, br, brblk

def open_time_pdf(mec, tres, tmin=0.00001, tmax=1000, points=512, unit='ms'):
    """
    Calculate ideal asymptotic and exact open time distributions.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution.
    tmin, tmax : floats
        Time range for burst length ditribution.
    points : int
        Number of points per plot.
    unit : str
        'ms'- milliseconds.

    Returns
    -------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    ipdf, epdf, apdf : ndarrays of floats, shape (num of points)
        Ideal, exact and asymptotic open time distributions.
    """

    open = True

    # Asymptotic pdf
    roots = scl.asymptotic_roots(tres,
        mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)

    tmax = (-1 / roots.max()) * 20
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    # Ideal pdf.
    eigs, w = scl.ideal_dwell_time_pdf_components(mec.QAA, qml.phiA(mec))
    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
    ipdf = np.zeros(points)
    for i in range(points):
        ipdf[i] = t[i] * scl.ideal_dwell_time_pdf(t[i],
            mec.QAA, qml.phiA(mec)) * fac

    # Asymptotic pdf
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    areas = scl.asymptotic_areas(tres, roots,
        mec.QAA, mec.QFF, mec.QAF, mec.QFA,
        mec.kA, mec.kF, GAF, GFA)

    apdf = np.zeros(points)
    for i in range(points):
        apdf[i] = t[i] * pdfs.expPDF(t[i] - tres, -1 / roots, areas)

    # Exact pdf
    eigvals, gamma00, gamma10, gamma11 = scl.exact_GAMAxx(mec,
        tres, open)
    epdf = np.zeros(points)
    for i in range(points):
        epdf[i] = (t[i] * scl.exact_pdf(t[i], tres,
            roots, areas, eigvals, gamma00, gamma10, gamma11))
            
    if unit == 'ms':
        t = t * 1000 # x scale in millisec

    return t, ipdf, epdf, apdf

def shut_time_pdf(mec, tres, tmin=0.00001, tmax=1000, points=512, unit='ms'):
    """
    Calculate ideal asymptotic and exact shut time distributions.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution.
    tmin, tmax : floats
        Time range for burst length ditribution.
    points : int
        Number of points per plot.
    unit : str
        'ms'- milliseconds.

    Returns
    -------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    ipdf, epdf, apdf : ndarrays of floats, shape (num of points)
        Ideal, exact and asymptotic shut time distributions.
    """

    open = False

    # Asymptotic pdf
    roots = scl.asymptotic_roots(tres, mec.QFF, mec.QAA, mec.QFA, mec.QAF,
        mec.kF, mec.kA)

    tmax = (-1 / roots.max()) * 20
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    # Ideal pdf.
    eigs, w = scl.ideal_dwell_time_pdf_components(mec.QFF, qml.phiF(mec))
    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor

    ipdf = np.zeros(points)
    for i in range(points):
        ipdf[i] = t[i] * scl.ideal_dwell_time_pdf(t[i],
            mec.QFF, qml.phiF(mec)) * fac

    # Asymptotic pdf
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    areas = scl.asymptotic_areas(tres, roots,
        mec.QFF, mec.QAA, mec.QFA, mec.QAF,
        mec.kF, mec.kA, GFA, GAF)
    apdf = np.zeros(points)
    for i in range(points):
        apdf[i] = t[i] * pdfs.expPDF(t[i] - tres, -1 / roots, areas)

    # Exact pdf
    eigvals, gamma00, gamma10, gamma11 = scl.exact_GAMAxx(mec, tres, open)
    epdf = np.zeros(points)
    for i in range(points):
        epdf[i] = (t[i] * scl.exact_pdf(t[i], tres,
            roots, areas, eigvals, gamma00, gamma10, gamma11))

    if unit == 'ms':
        t = t * 1000 # x scale in millisec

    return t, ipdf, epdf, apdf

def subset_time_pdf(mec, tres, state1, state2,
    tmin=0.00001, tmax=1000, points=512, unit='ms'):
    """
    Calculate ideal pdf of any subset dwell times.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution.
    state1, state2 : ints
    tmin, tmax : floats
        Time range for burst length ditribution.
    points : int
        Number of points per plot.
    unit : str
        'ms'- milliseconds.

    Returns
    -------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    spdf : ndarray of floats, shape (num of points)
        Subset dwell time pdf.
    """

    open = False
    if open:
        tau, area = scl.ideal_dwell_time_pdf_components(mec.QAA, qml.phiA(mec))
    else:
        tau, area = scl.ideal_dwell_time_pdf_components(mec.QFF, qml.phiF(mec))

    tmax = tau.max() * 20
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    # Ideal pdf.
    f = 0.0
    for i in range(mec.kF):
        f += area[i] * np.exp(-tres / tau[i])
    fac = 1 / f # Scale factor.
    ipdf = np.zeros(points)
    spdf = np.zeros(points)
    for i in range(points):
        ipdf[i] = t[i] * scl.ideal_dwell_time_pdf(t[i],
            mec.QFF, qml.phiF(mec)) * fac
        spdf[i] = t[i] * scl.ideal_subset_time_pdf(mec.Q,
            state1, state2, t[i]) * fac

    if unit == 'ms':
        t = t * 1000 # x scale in millisec

    return t, ipdf, spdf


class SquareRootScale(mscale.ScaleBase):
    """
    Class for generating square root scaled axis for probability density
    function plots.
    """

    name = 'sqrtscale'
    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)
    def get_transform(self):
        """
        Set the actual transform for the axis coordinates.
        """
        return self.SqrTransform()
    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to reasonable defaults.
        """
        axis.set_major_formatter(ticker.ScalarFormatter())

    class SqrTransform(mtransforms.Transform):
        """
        """
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mtransforms.Transform.__init__(self)
        def transform(self, a):
            """
            Take numpy array and return transformed copy.
            """
            return np.sqrt(a)
        def inverted(self):
            """
            Get inverse transform.
            """
            return SquareRootScale.InvertedSqrTransform()

    class InvertedSqrTransform(mtransforms.Transform):
        """
        """
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mtransforms.Transform.__init__(self)
        def transform(self, a):
            return np.power(a, 2)
        def inverted(self):
            return SquareRootScale.SqrTransform()
