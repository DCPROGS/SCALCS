import math
import numpy as np
from pylab import figure, semilogx, savefig

from scalcs import qmatlib as qml
from scalcs import scalcslib as scl
from scalcs import pdfs
from scalcs import cjumps



def conc_jump_on_off_taus_versus_conc_plot(mec, cmin, cmax, width):
    """
    Calculate data for the plot of square concentration pulse evoked current 
    (occupancy) weighted on and off time constants versus concentration.

    Parameters
    ----------
    mec : instance of type Mechanism
    cmin, cmax : float
        Range of concentrations in M.

    Returns
    -------
    c : ndarray of floats, shape (num of points,)
        Concentration in mikroM
    ton, toff : floats
        On and off weighted time constants.
    """

    points = 100
    c = np.logspace(int(np.log10(cmin)), int(np.log10(cmax)), points)
    
    wton = np.zeros(points)
    wtoff = np.zeros(points)
    ton = np.zeros((points, mec.k-1))
    toff = np.zeros((points, mec.k-1))
    for i in range(points):
        mec.set_eff('c', c[i])
        wton[i], ton[i], wtoff[i], toff[i] = cjumps.weighted_taus(mec, c[i], width)

    ton = ton.transpose()
    toff = toff.transpose()

    return c * 1000, wton * 1000, ton * 1000, wtoff * 1000, toff * 1000

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
        mec.QAA, mec.QII, mec.QAI, mec.QIA, mec.kA, mec.kI)

    tmax = (-1 / roots.max()) * 20
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    # Ideal pdf.
    eigs, w = scl.ideal_dwell_time_pdf_components(mec.QAA, qml.phiA(mec))
    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
    ipdf = t * pdfs.ExpPDF(1 / eigs, w / eigs).calculate(t) * fac
    

    # Asymptotic pdf
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kI)
    areas = scl.asymptotic_areas(tres, roots,
        mec.QAA, mec.QII, mec.QAI, mec.QIA,
        mec.kA, mec.kI, GAF, GFA)
    apdf = scl.asymptotic_pdf(t, tres, -1 / roots, areas)

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


def scaled_pdf(t, pdf, dt, n):
    """
    Scale pdf to the data histogram.

    Parameters
    ----------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    pdf : ndarray of floats, shape (num of points)
        pdf to scale.
    dt : float
        Histogram bin width in log10 units.
    n : int
        Total number of events.

    Returns
    -------
    spdf : ndarray of floats, shape (num of points)
        Scaled pdf.
    """

    spdf = n * dt * 2.30259 * pdf
    #spdf = n * dt * pdf
    return spdf

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
    roots = scl.asymptotic_roots(tres, mec.QII, mec.QAA, mec.QIA, mec.QAI,
        mec.kI, mec.kA)

    tmax = (-1 / roots.max()) * 20
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    # Ideal pdf.
    eigs, w = scl.ideal_dwell_time_pdf_components(mec.QII, qml.phiF(mec))
    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
    ipdf = t * pdfs.ExpPDF(1 / eigs, w / eigs).calculate(t) *fac #pdfs.expPDF(t, 1 / eigs, w / eigs) * fac

    # Asymptotic pdf
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kI)
    areas = scl.asymptotic_areas(tres, roots,
        mec.QII, mec.QAA, mec.QIA, mec.QAI,
        mec.kI, mec.kA, GFA, GAF)
    apdf = scl.asymptotic_pdf(t, tres, -1 / roots, areas)

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
        eigs, w = scl.ideal_dwell_time_pdf_components(mec.QAA, qml.phiA(mec))
    else:
        eigs, w = scl.ideal_dwell_time_pdf_components(mec.QII, qml.phiF(mec))

    tmax = tau.max() * 20
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    # Ideal pdf.
    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
    ipdf = t * pdfs.expPDF(t, 1 / eigs, w / eigs) * fac

    spdf = np.zeros(points)
    for i in range(points):
        spdf[i] = t[i] * scl.ideal_subset_time_pdf(mec.Q,
            state1, state2, t[i]) * fac

    if unit == 'ms':
        t = t * 1000 # x scale in millisec

    return t, ipdf, spdf

def png_save_pdf_fig(outfile, ints, mec, conc, tres, type):
    x, y, dx = prepare_hist(ints, tres)
    mec.set_eff('c', conc)
    if type == 'open':
        t, ipdf, epdf, apdf = open_time_pdf(mec, tres)
    elif type == 'shut':
        t, ipdf, epdf, apdf = shut_time_pdf(mec, tres)
    else:
        print ('Wrong type.')

    sipdf = scaled_pdf(t, ipdf, math.log10(dx), len(ints))
    sepdf = scaled_pdf(t, epdf, math.log10(dx), len(ints))
    figure(figsize=(6, 4))
    semilogx(x*1000, y, 'k-', t, sipdf, 'r--', t, sepdf, 'b-')
    savefig(outfile, bbox_inches=0)