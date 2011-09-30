import math, sys

import numpy as np

def expPDF(t, tau, area):
    """
    Calculate exponential probabolity density function.

    Parameters
    ----------
    t : float
        Time.
    tau : ndarray, shape(k, 1)
        Time constants.
    area : ndarray, shape(k, 1)
        Component relative area.

    Returns
    -------
    f : float
    """

    f = 0
    if t >= 0:
        f = np.sum((area / tau) * np.exp(-t / tau))
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

