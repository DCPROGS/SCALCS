"""
Gaussian filter, as in C & Sigworth (1994) Fig A3.1
"""

from math import *
import numpy as np
from scipy.special import erf

def gfilter(input, coeffs):
    """
    Gaussian filter, as in C & Sigworth (1994) Fig A3.1

    Parameters
    ----------
    input : array_like, shape (n), short int
    coeffs : array_like, shape (nc), floats
        Gaussian filter coefficients.

    Returns
    -------
    out : array_like, shape (n), short int
    """

    n = input.shape[0]
    out = np.zeros(n, 'h')
    nc = len(coeffs) - 1 # The central component is the zeroth, therefore reduce
                         # Actual number of comps in sum is now 2nc +1
    for i in range(n):
        jl = i - nc
        if jl < 0: jl = 0
        ju = i + nc +1
        if ju > n: ju = n

        arr1 = np.abs(np.arange(jl, ju) - i)
        arr2 = coeffs[arr1] * input[jl:ju]
        out[i] = np.sum(arr2)
        # DO NOT DELETE next 5 commented lines for future reference.
        #sum = 0
        #for j in range (jl,ju):
        #    k = abs(j-i)	#which component
        #    sum += input[j] * coeffs[k]
        #out[i] = sum
    return out

def gaussian_coeff(fc, verbose=0):
    """
    Calculate coefficients of gaussian filter.

    Parameters
    ----------
    fc : float
        Frequency in Hz.

    Returns
    -------
    a : array_like, shape (nc), floats
        Gaussian filter coefficients.
    """

    sigma = 0.13205 / fc
    # nc -number of coefficients, not counting the central one, a(0).
    nc = int(4 * sigma) + 1 #python rounds down
    B = -0.5 / (sigma * sigma)
    arr = np.arange(nc)
    a = np.exp(B * arr * arr)
    norm = 2 * np.sum(a) - 1
    a = a / norm

    if verbose:
        print("Normalized coefficients of Gaussian filter:")
        for x in range(nc):
            print("{0:d} ".format(x) + "{0:.6f}".format(a[x]))

    return a

def filter_trace(datain, fc, ffilter, srate, verbose=1):
    """
    Go through sections of entire trace and filter to have a final cutoff
    frequency fc.

    Parameters
    ----------
    datain : array_like, shape (n), short int
    fc : float
        Final cutoff frequency in Hz.
    ffilter : float
        Filter already applied to input data in Hz.
    srate : float
        Sampling rate in Hz.

    Returns
    -------
    dataout : array_like, shape (n), short int
    """

    ndatain = datain.shape[0]
    nbuf = 1000000 # size of data arrays
    novlap = 100 # size of overlap
    # Calculate number of sections needed for data.
    nsec = ndatain / nbuf
    # Do not filter the last section if it has less points than overlap.
    if ndatain - (nsec - 1) * nbuf < novlap:
        nsec = nsec - 1
    if verbose:
        print("File will be filtered in {0:d} sections.".format(nsec))

    # Calculate how many points to read in one block for a section to filter.
    nread = nbuf + 2 * novlap
    nleft = 0

    dataout = np.zeros(ndatain, 'h')
    # Get a frequency for a filter to apply to get requested final fc.
    ffc = fci(ffilter, fc)
    # Get Gaussian filter coefficients.
    a = gaussian_coeff(ffc / srate, verbose)

    for isec in range(nsec):
        n = nbuf * isec
        n1 = n - novlap
        n2 = n + nbuf + novlap
        if verbose: print(" Filtering section {0:d} ...".format(isec+1))
        
        if isec == 0:
            temp = np.zeros(nread, 'h')
            temp[0:novlap] = datain[0:novlap]
            temp[novlap:nread] = datain[0:nbuf+novlap]
            ftemp = gfilter(temp, a)
            dataout[n:n+nbuf] = ftemp[novlap:nbuf+novlap]

        elif isec == (nsec-1):
            nleft = ndatain - n
            nread = nleft + 2 * novlap
            temp = np.zeros(nread, 'h')
            temp[0:nread-novlap] = datain[n1:ndatain]
            temp[nread-novlap:nread] = datain[ndatain-novlap:ndatain]
            ftemp = gfilter(temp, a)
            dataout[n:n+nleft] = ftemp[novlap:novlap+nleft]

        else:
            temp = np.zeros(nread, 'h')
            temp[0:nread] = datain[n1:n2]
            ftemp = gfilter(temp, a)
            dataout[n:n+nbuf] = ftemp[novlap:novlap+nbuf]

    if verbose: print("Filtering finished.")
    srate1, idelt = resample(fc, srate, verbose)
    if idelt > 1:
        if (ndatain % idelt) != 0:
            dataout = np.append(dataout, np.zeros(idelt - (ndatain % idelt)))
        dataout = np.reshape(dataout, (dataout.shape[0] / idelt, -1))[:,0]

    return dataout, srate1

def fci(ffilter, fc):
    """
    Calculate frequency by which to filter a trace which already filtered with
    ffilter to get final fc. 

    Parameters
    ----------
    ffilter : float
        Filter alreade applied to datain. In Hz.
    fc : float
        Final cutoff frequency in Hz.

    Returns
    -------
    fci : float
        Intermediate filter frequency in Hz.

    """
    
    fci = 1. / sqrt((1.0 / (fc * fc)) - (1.0 / (ffilter * ffilter)))
    return fci

def resample(fc, srate, verbose=0):
    """
    Calculate how much sampling rate can be reduced for output given the
    filter fc. Reduce 'sample rate' for output. If srate > 10*fc then
    recommend largest reduction that keeps srate at least 10fc. First
    reduction possible (without interpolation) is by factor of 2, so try
    only if srate > 20fc.

    Parameters
    ----------
    datain : array_like, shape (n), short int
    fc : float
        Final cutoff frequency in Hz.
    srate : float
        Sampling rate in Hz.

    Returns
    -------
    srate1 : float
        New sampling ratein Hz.
    idelt : int
        Resample each idelt point.
    """

    srate1 = srate # value for output
    r = srate / fc
    idelt = int(r / 10)
    if idelt >= 2:
        srate1 = srate / idelt
    r1 = srate1 / fc

    if verbose:
        print(" Sample rate is {0:.1f} times final fc.".format(r))
        print(" Recommend reduction the sample rate for output file by using")
        print(" only every {0:d}th point to give sample rate of ".format(idelt) +
            "{0:.3f} Hz".format(srate1))
        print(" which is {0:.1f} times fc.".format(r1))

    return srate1, idelt

def false_events(tres, fc, rms, amp):
    """
    Version for EKDIST/new SCAN (avamp, rms already in pA). \
    To calc false event rate (per sec) in EKDIST (from RESINT.) \
    First calc threshold as amp attained by pulse of length=tres (in ms).
    """
    u = erf(2.668 * fc * tres)
    phi = u * amp    #'threshold' (pA)
    var = (rms) ** 2    # noise variance (pA)**2
    # Calc rate from C & Sigworth eq. 9, with k=1
    frate = fc * np.exp(-(phi * phi) / (2. * var))
    return frate    # false event rate
