"""
Ploting utilities for single channel and macroscopic current
calculations.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 23:01:09$"

import numpy as np

import qmatlib as qml
import scalcslib as scl

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
    text1 : string
        Contains parameters for Popen curve corrected for missed events.
    text2 : string
        Contains parameters for ideal Popen curve.
    c : ndarray of floats, shape (num of points,)
        Concentration in mikroM.
    pe : ndarray of floats, shape (num of points,)
        Open probability corrected for missed events.
    pi : ndarray of floats, shape (num of points,)
        Ideal open probability.
    """

    # Calculate EC50, nH and maxPopen for ideal Popen curve.
    emaxPopen, econc = scl.get_maxPopen(mec, tres)
    eEC50 = scl.get_EC50(mec, tres) * 1000000    # in mikroM
    enH = scl.get_nH(mec, tres)
    text1 = ('HJC Popen curve:\nmaxPopen = {0:.3f}; '.format(emaxPopen) +
        ' EC50 = {0:.3f} mikroM; '.format(eEC50) + ' nH = {0:.3f}'.format(enH))

    # Calculate EC50, nH and maxPopen for Popen curve
    # corrected for missed events.
    imaxPopen, iconc = scl.get_maxPopen(mec, 0)
    iEC50 = scl.get_EC50(mec, 0) * 1000000   # in mikroM
    inH = scl.get_nH(mec, 0)
    text2 = ('\nIdeal Popen curve:\nmaxPopen = {0:.3f}; '.format(imaxPopen)+
        ' EC50 = {0:.3f} mikroM; '.format(iEC50) + ' nH = {0:.3f}'.format(inH))

    # Plot ideal and corrected Popen curves.
    cmin = iEC50 / 20000000.0
    cmax = iEC50 * 500 / 1000000.0
    log_start = int(np.log10(cmin)) - 1
    log_end = int(np.log10(cmax)) - 1
    decade_num = int(log_end - log_start)
    log_int = 0.01    # increase this if want more points per curve
    point_num = int(decade_num / log_int + 1)

    c = np.zeros(point_num)
    pe = np.zeros(point_num)
    pi = np.zeros(point_num)
    for i in range(point_num):
        ctemp = pow(10, log_start + log_int * i)
        pe[i] = scl.popen(mec, tres, ctemp)
        pi[i] = scl.popen(mec, 0, ctemp)
        c[i] = ctemp * 1000000

    return text1, text2, c, pe, pi

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
    text1 : string
        Mean burst length.
    t : ndarray of floats, shape (num of points,)
        Time in millisec.
    fbst : ndarray of floats, shape (num of points,)
        Burst length pdf.
    """

    # Calculate mean burst length.
    mec.set_eff('c', conc)
    m = scl.mean_burst_length(mec) * 1000
    text1 = 'Mean burst length = %f millisec' %m

    # Calculate burst length pdf.
    point_num = 1000
    dt = (np.log10(tmax) - np.log10(tmin)) / (point_num - 1)

    t = np.zeros(point_num)
    fbst = np.zeros(point_num)
    for i in range(point_num):
        temp = tmin * pow(10, (i * dt))
        fbst[i] = np.sqrt(temp * scl.pdf_burst_length(mec, temp)) * 1000
        t[i] = temp * 1000

    return text1, t, fbst

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
    text1 : string
        Mean number of openings per burst.
    r : ndarray of floats, shape (num of points,)
        Number of openings per burst.
    Pr : ndarray of floats, shape (num of points,)
        Fraction of bursts.
    """

    # Calculate mean number of openings per burst.
    mec.set_eff('c', conc)
    mu = scl.mean_num_burst_openings(mec)
    text1 = 'Mean number of openings per burst = %f' %mu

    # Plot distribution of number of openings per burst
    n = 10
    r = np.arange(1, n+1)
    Pr = np.zeros(n)
    for i in range(n):
        Pr[i] = scl.distr_num_burst_openings(mec, r[i])

    return text1, r, Pr

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
    point_num = 100
    incr = (cmax - cmin)/(point_num - 1)
    c = np.zeros(point_num)
    br = np.zeros(point_num)
    for i in range(point_num):
        ctemp = cmin + incr * i
        mec.set_eff('c', ctemp)
        br[i] = scl.mean_burst_length(mec) * 1000
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

    point_num = 100
    incr = (cmax - cmin)/(point_num - 1)
    c = np.zeros(point_num)
    br = np.zeros(point_num)
    brblk = np.zeros(point_num)
    for i in range(point_num):
        ctemp = cmin + incr * i
        mec.set_eff('c', ctemp)
        br[i] = scl.mean_burst_length(mec) * 1000
        brblk[i] = br[i] * (1 + ctemp / mec.KB)
        c[i] = ctemp * 1000000
    return c, br, brblk

def get_opentime_pdf(mec, conc, tres, tmin, tmax, open):
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
    text1 : string
        Mean open time.
    t : ndarray of floats, shape (num of points,)
        Time in millisec.
    fopen : ndarray of floats, shape (num of points,)
        Open time pdf.
    """

    mec.set_eff('c', conc)
    tau, area = scl.get_ideal_pdf_components(mec, open)
    tmax = tau.max() * 20

    # Scale factor.
    f = 0.0
    for i in range(mec.kA):
        f += area[i] * np.exp(-tres / tau[i])
    fac = 1 / f

    point_num = 512
    dt = (np.log10(tmax) - np.log10(tmin)) / (point_num - 1)
    t = np.zeros(point_num)
    f = np.zeros(point_num)
    for i in range(point_num):
        t[i] = tmin * pow(10, (i * dt))
        f[i] = np.sqrt(t[i] * scl.pdf_open_time(mec, t[i]) * fac)
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

    point_num = 512
    dt = (np.log10(tmax) - np.log10(tmin)) / (point_num - 1)
    t = np.zeros(point_num)
    f = np.zeros(point_num)
    for i in range(point_num):
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

    nPoint = 1000
    dt = (np.log10(tmax) - np.log10(tmin)) / (nPoint - 1)
    t = np.zeros(nPoint)
    f = np.zeros(nPoint)
    for i in range(nPoint):
        t[i] = tmin * pow(10, (i * dt))
        f[i] = np.sqrt(t[i] * scl.pdf_asymptotic(t[i], tres, roots, areas))

    return t * 1000, f * 1000

def get_exact_pdf(mec, conc, tres, tmin, open):
    """

    """

    mec.set_eff('c', conc)
    roots = scl.asymptotic_roots(mec, tres, open)
    areas = scl.asymptotic_areas(mec, tres, roots, open)
    eigvals, gamma00, gamma10, gamma11 = scl.exact_pdf_coef(mec, tres, open)


    tmax = (-1 / roots.max()) * 20

    nPoint = 1000
    dt = (np.log10(tmax) - np.log10(tmin)) / (nPoint - 1)
    t = np.zeros(nPoint)
    f = np.zeros(nPoint)
    for i in range(nPoint):
        t[i] = tmin * pow(10, (i * dt))
        f[i] = np.sqrt(t[i] * scl.pdf_exact(t[i], tres,
            roots, areas, eigvals, gamma00, gamma10, gamma11))

    return t * 1000, f * 1000

