"""
Plotting utilities for single channel currents.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 23:01:09$"

import sys

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

def get_Popen_plot(mec, tres, cmin, cmax):
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

    iEC50 = popen.EC50(mec, 0) * 1000000   # in mikroM

    # Plot ideal and corrected Popen curves.
    cmin = iEC50 / 20000000.0
    cmax = iEC50 * 500 / 1000000.0
    log_start = int(np.log10(cmin)) - 1
    log_end = int(np.log10(cmax)) - 1
    decades = int(log_end - log_start)
    log_int = 0.01    # increase this if want more points per curve
    points = int(decades / log_int + 1)

    c = np.zeros(points)
    pe = np.zeros(points)
    pi = np.zeros(points)
    for i in range(points):
        ctemp = pow(10, log_start + log_int * i)
        pe[i] = popen.Popen(mec, tres, ctemp)
        pi[i] = popen.Popen(mec, 0, ctemp)
        c[i] = ctemp * 1000000

    return c, pe, pi

def get_burstlen_pdf(mec, conc, tmin, tmax):
    """
    Calculate the mean burst length and data for burst length distribution.

    Parameters
    ----------
    mec : instance of type Mechanism
    conc : float
        Concentration in M.
    tmin, tmax : floats
        Time range for burst length ditribution.

    Returns
    -------
    t : ndarray of floats, shape (num of points,)
        Time in millisec.
    fbst : ndarray of floats, shape (num of points,)
        Burst length pdf.
    """

    mec.set_eff('c', conc)

    # Calculate burst length pdf.
    points = 1000
    dt = (np.log10(tmax) - np.log10(tmin)) / (points - 1)

    t = np.zeros(points)
    fbst = np.zeros(points)
    for i in range(points):
        temp = tmin * pow(10, (i * dt))
        fbst[i] = np.sqrt(temp * scburst.length_pdf(mec, temp)) * 1000
        t[i] = temp * 1000

    return t, fbst

def get_burstopenings_distr(mec, conc):
    """
    Calculate the mean number of openings per burst and data for the
    distribution of openings per burst.

    Parameters
    ----------
    mec : instance of type Mechanism
    conc : float
        Concentration in M.

    Returns
    -------
    r : ndarray of floats, shape (num of points,)
        Number of openings per burst.
    Pr : ndarray of floats, shape (num of points,)
        Fraction of bursts.
    """


    mec.set_eff('c', conc)

    # Plot distribution of number of openings per burst
    n = 10
    r = np.arange(1, n+1)
    Pr = np.zeros(n)
    for i in range(n):
        Pr[i] = scburst.openings_distr(mec, r[i])

    return r, Pr

def get_burstlen_conc_plot(mec, cmin, cmax):
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
    """

    points = 100
    dc = (cmax - cmin)/(points - 1)
    c = np.zeros(points)
    br = np.zeros(points)
    for i in range(points):
        ctemp = cmin + dc * i
        mec.set_eff('c', ctemp)
        br[i] = scburst.length_mean(mec) * 1000
        c[i] = ctemp * 1000000
    return c, br

def get_burstlen_conc_fblk_plot(mec, cmin, cmax):
    """
    Calculate data for the plot of burst length versus concentration.
    Returns burst length in absence and presence of short unresolved blockages.

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
    """

    points = 100
    dc = (cmax - cmin)/(points - 1)
    c = np.zeros(points)
    br = np.zeros(points)
    brblk = np.zeros(points)
    for i in range(points):
        ctemp = cmin + dc * i
        mec.set_eff('c', ctemp)
        br[i] = scburst.length_mean(mec) * 1000
        brblk[i] = br[i] * (1 + ctemp / mec.KBlk)
        c[i] = ctemp * 1000000
    return c, br, brblk

def open_time_pdf(mec, tres, conc, axes, output=sys.stdout):
    """
    """

    output.write('\n\n\t===== OPEN TIME PDF =====')
    output.write('\nAgonist concentration = {0:.6f} mikroM'.
        format(conc * 1000000))
    output.write('\nResolution = {0:.2f} mikrosec'.
        format(tres * 1000000))
    output.write('\nIdeal pdf- red dashed line.')
    output.write('\nExact pdf- blue solid line.')
    output.write('\nAsymptotic pdf- green solid line.')

    open = True

    # Asymptotic pdf
    #roots = scl.asymptotic_roots(self.mec, self.tres, open)
    roots = scl.asymptotic_roots(tres,
        mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)

    tmax = (-1 / roots.max()) * 20
    tmin = 0.00001 # 10 mikrosec
    points = 512
    step = (np.log10(tmax) - np.log10(tmin)) / (points - 1)
    t = np.zeros(points)

    # Ideal pdf.
    f = 0.0
    tau, area = scl.ideal_dwell_time_pdf_components(mec.QAA,
        qml.phiA(mec))
    for i in range(mec.kA):
        f += area[i] * np.exp(-tres / tau[i])
    fac = 1 / f # Scale factor.
    ipdf = np.zeros(points)
    for i in range(points):
        t[i] = tmin * pow(10, (i * step))
        ipdf[i] = t[i] * scl.ideal_dwell_time_pdf(t[i],
            mec.QAA, qml.phiA(mec)) * fac

    # Asymptotic pdf
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    #areas = scl.asymptotic_areas(self.mec, self.tres, roots, open)
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

    t = t * 1000 # x scale in millisec
    axes.clear()
    axes.semilogx(t, ipdf, 'r--', t, epdf, 'b-', t, apdf, 'g-')
    axes.set_yscale('sqrtscale')
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')

    return axes

def get_opentime_pdf(mec, tres, tmin, tmax):
    """
    Calculate the mean open time and data for open time distribution.

    Parameters
    ----------
    mec : instance of type Mechanism
    conc : float
        Concentration in M.
    tmin, tmax : floats
        Time range for burst length ditribution.

    Returns
    -------
    t : ndarray of floats, shape (num of points,)
        Time in millisec.
    fopen : ndarray of floats, shape (num of points,)
        Open time pdf.
    """

    tau, area = scl.ideal_dwell_time_pdf_components(mec.QAA, qml.phiA(mec))
    tmax = tau.max() * 20

    # Scale factor.
    f = 0.0
    for i in range(mec.kA):
        f += area[i] * np.exp(-tres / tau[i])
    fac = 1 / f

    points = 512
    dt = (np.log10(tmax) - np.log10(tmin)) / (points - 1)
    t = np.zeros(points)
    f = np.zeros(points)
    for i in range(points):
        t[i] = tmin * pow(10, (i * dt))
        f[i] = np.sqrt(t[i] * scl.ideal_dwell_time_pdf(t[i],
            mec.QAA, qml.phiA(mec)) * fac)

    return t * 1000, f * 1000

def get_shuttime_pdf(mec, conc, tres, tmin, tmax, open):
    """
    Calculate the mean shut time and data for shut time distribution.

    Parameters
    ----------
    mec : instance of type Mechanism
    conc : float
        Concentration in M.
    tmin, tmax : floats
        Time range for burst length ditribution.

    Returns
    -------
    text1 : string
        Mean shut time.
    t : ndarray of floats, shape (num of points,)
        Time in millisec.
    fbst : ndarray of floats, shape (num of points,)
        Shut time pdf.
    """

    mec.set_eff('c', conc)
    tau, area = scl.get_ideal_pdf_components(mec, open)
    tmax = tau.max() * 20

    # Scale factor.
    f = 0.0
    for i in range(mec.kF):
        f += area[i] * np.exp(-tres / tau[i])
    fac = 1 / f

    points = 512
    dt = (np.log10(tmax) - np.log10(tmin)) / (points - 1)
    t = np.zeros(points)
    f = np.zeros(points)
    for i in range(points):
        t[i] = tmin * pow(10, (i * dt))
        f[i] = np.sqrt(t[i] * scl.pdf_shut_time(mec, t[i]) * fac)
    #return text1, t, fsht
    return t * 1000, f * 1000

def get_asymptotic_pdf(mec, conc, tres, tmin, open):
    """

    """

    mec.set_eff('c', conc)
    roots = scl.asymptotic_roots(mec, tres, open)
    areas = scl.asymptotic_areas(mec, tres, roots, open)

    tmax = (-1 / roots.max()) * 20

    points = 1000
    dt = (np.log10(tmax) - np.log10(tmin)) / (points - 1)
    t = np.zeros(points)
    f = np.zeros(points)
    for i in range(points):
        t[i] = tmin * pow(10, (i * dt))
        f[i] = np.sqrt(t[i] * scl.pdf_exponential(t[i], tres, roots, areas))

    return t * 1000, f * 1000

def get_exact_pdf(mec, conc, tres, tmin, open):
    """

    """

    mec.set_eff('c', conc)
    roots = scl.asymptotic_roots(mec, tres, open)
    areas = scl.asymptotic_areas(mec, tres, roots, open)
    eigvals, gamma00, gamma10, gamma11 = scl.exact_pdf_coef(mec, tres, open)
    tmax = (-1 / roots.max()) * 20

    points = 1000
    dt = (np.log10(tmax) - np.log10(tmin)) / (points - 1)
    t = np.zeros(points)
    f = np.zeros(points)
    for i in range(points):
        t[i] = tmin * pow(10, (i * dt))
        f[i] = np.sqrt(t[i] * scl.pdf_exact(t[i], tres,
            roots, areas, eigvals, gamma00, gamma10, gamma11))

    return t * 1000, f * 1000


class SquareRootScale(mscale.ScaleBase):
    """
    Class for generating sqare root scaled axis for probability density
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