import math
import sys

import numpy as np

def expPDF(t, tau, area):
    """
    Calculate exponential probabolity density function.

    Parameters
    ----------
    t : float or ndarray.
        Time.
    tau : ndarray, shape(k, 1)
        Time constants.
    area : ndarray, shape(k, 1)
        Component relative area.

    Returns
    -------
    f : float or ndarray.
    """

    if tau.shape[0] == 1:
        f = np.sum((area / tau) * np.exp(-t / tau))
    else:
        f = np.zeros(t.shape)
        for i in range(tau.shape[0]):
            f += (area[i] / tau[i]) * np.exp(-t / tau[i])
    return f

def expPDF_mean_sd(tau, area):
    """
    Calculate mean and standard deviation for exponential PDF.

    Parameters
    ----------
    tau : ndarray, shape(k, 1)
        Time constants.
    area : ndarray, shape(k, 1)
        Component relative area.

    Returns
    -------
    m : float
        Mean.
    sd : float
        Standard deviation.
    """

    m = np.sum(area * tau)
    var = np.sum(area * tau * tau)
    sd = math.sqrt(2 * var - m * m)

    return m, sd

def expPDF_printout(eigs, ampl, output=sys.stdout):
    """
    """

    output.write('\nterm\tw\trate (1/sec)\ttau (ms)\tarea (%)')
    for i in range(eigs.shape[0]):
        output.write('\n{0:d}'.format(i+1) +
            '\t{0:.3f}'.format(ampl[i]) +
            '\t{0:.1f}'.format(eigs[i]) +
            '\t{0:.3f}'.format(1000 / eigs[i]) +
            '\t{0:.3f}'.format(100 * ampl[i] / eigs[i]))

    mean, sd = expPDF_mean_sd(1 / eigs, ampl / eigs)
    output.write('\nMean (ms) =\t {0:.3f}'.format(mean * 1000) +
        '\tSD =\t {0:.3f}'.format(sd * 1000) +
        '\tSD/mean =\t {0:.3f}'.format(sd / mean))

def expPDF_misclassified(tcrit, tau, area, comp):
    """
    Calculate number and fraction of misclassified events after division into
    bursts by critical time, tcrit.
    """

    tfast = tau[:comp]
    tslow = tau[comp:]
    afast = area[:comp]
    aslow = area[comp:]

    # Number of misclassified.
    enf = np.sum(afast * np.exp(-tcrit / tfast))
    ens = np.sum(aslow * (1 - np.exp(-tcrit / tslow)))

    # Fraction misclassified.
    pf = enf / np.sum(afast)
    ps = ens / np.sum(aslow)

    return enf, ens, pf, ps

def expPDF_misclassified_printout(tcrit, enf, ens, pf, ps, output=sys.stdout):
    """
    """

    output.write('\ntcrit = {0:.6f} ms'.format(tcrit * 1000))
    output.write('\n% misclassified: short = {0:.3f};'.format(pf * 100) +
        ' long = {0:.3f}'.format(ps * 100) +
        '\n# misclassified (out of 100): short = {0:.3f};'.format(enf * 100) +
        ' long = {0:.3f}'.format(ens * 100) +
        '\nTotal # misclassified (out of 100) = {0:.3f}\n'
        .format((enf + ens) * 100))

def expPDF_tcrit_DC(tcrit, tau, area, comp):
    """
    """

    enf, ens, pf, ps = expPDF_misclassified(tcrit, tau, area, comp)
    return ps - pf

def expPDF_tcrit_CN(tcrit, tau, area, comp):
    """
    """

    enf, ens, pf, ps = expPDF_misclassified(tcrit, tau, area, comp)
    return ens - enf

def expPDF_tcrit_Jackson(tcrit, tau, area, comp):
    """
    """

    tfast = tau[:comp]
    tslow = tau[comp:]
    afast = area[:comp]
    aslow = area[comp:]

    # Number of misclassified.
    enf = np.sum((afast / tfast) * np.exp(-tcrit / tfast))
    ens = np.sum((aslow / tslow) * np.exp(-tcrit / tslow))

    return enf - ens

def geometricPDF_mean_sd(rho, w):
    """
    Calculate mean and standard deviation for geometric PDF.

    Parameters
    ----------
    rho : ndarray, shape(k, 1)
        Probabilities.
    w : ndarray, shape(k, 1)
        Component amplitudes.

    Returns
    -------
    m : float
        Mean.
    sd : float
        Standard deviation.
    """

    k = rho.shape[0]
    m = np.sum(w / np.power(np.ones((k)) - rho, 2))
    var = np.sum(w * (np.ones((k)) + rho) / np.power(np.ones((k)) - rho, 3))
    sd = math.sqrt(var - m * m)

    return m, sd

def geometricPDF_printout(rho, w, output=sys.stdout):
    """
    """

    norm = 1 / (np.ones((rho.shape[0])) - rho)
    output.write('\nterm\tw\trho\tarea(%)\tNorm mean')
    for i in range(rho.shape[0]):
        output.write('\n{0:d}'.format(i+1) +
            '\t{0:.6f}'.format(w[i]) +
            '\t{0:.6f}'.format(rho[i]) +
            '\t{0:.3f}'.format(w[i] * norm[i] * 100) +
            '\t{0:.3f}'.format(norm[i]))

    mean, sd = geometricPDF_mean_sd(rho, w)
    output.write('\nMean number of openings per burst =\t {0:.3f}'.format(mean) +
        '\n\tSD =\t {0:.3f}'.format(sd) +
        '\tSD/mean =\t {0:.3f}'.format(sd / mean))

