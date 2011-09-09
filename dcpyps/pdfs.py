import math

import numpy as np

def expPDF_mean_sd(eigs, w):
    """
    Calculate mean and standard deviation for exponential PDF.

    Parameters
    ----------
    eigs : ndarray, shape(k, 1)
        Time constants.
    w : ndarray, shape(k, 1)
        Component amplitudes.

    Returns
    -------
    m : float
        Mean.
    sd : float
        Standard deviation.
    """

    m = np.sum(w / (eigs * eigs))  # (area * tau)
    var = np.sum(w / (eigs * eigs * eigs)) # (area * tau^2)
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

