import math

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

