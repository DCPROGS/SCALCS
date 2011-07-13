"""
Gaussian filter, as in C & Sigworth (1994) Fig A3.1
"""

from math import *
import numpy as np

def gfilter(datin, fc, srate):
    """
    Gaussian filter, as in C & Sigworth (1994) Fig A3.1

    NB maximum of 54/2=27 data points at each end are affected by edge
    effects, so read in arrays that overlap by at least this number
    of points, and discard overlaps.

    Parameters
    ----------
    datin : array_like, shape (n, ), short int
    srate : float
        Sampling rate in Hz.
    fc : float
        -3 dB frequency in Hz.

    Returns
    -------
    datin : array_like, shape (n, 1), short int
    """

    # finter=microsec between sample points
    # fc1=fc/sample freq = fc/srate = fc*finter*1.e-6
    # eg srate=10 kHz, finter=100 microsec:
    # fc=1150 Hz  fc1=1150*100*1.e-6 =0.115
    # sample freq (Hz) = srate = 1/(finter*1.e-6) =1.e6/finter

    fc1 = fc / srate
    if fc1 < 0.01:
        print ' Error in GFILTER: fc1 is too small !'

    # Calculate the coefficients in a()
    # nc=number of coefficients, not counting the central one, a(0)
    sigma = 0.132505 / fc1
    a = []
    if sigma < 0.62:
        # then !narrow impulse -three terms only
       a.append(sigma * sigma / 2.0)
       a.append(1.0 - 2.0 * a[0])
       nc = 1
    else:
       nc = int(4.0 * sigma)
       b = -0.5 / (sigma * sigma)
       a.append(1.0)
       sum = 1.0
       for i in range(1, nc+1):
           a.append(np.exp(i * i * b))
           sum = sum + 2 * a[i]
       # Normalise the coefficients.
       a = a / sum

    # Now do the filtering.
    n = datin.shape[0]
    datout = np.zeros(n, 'h')
    for i in range(0, nc):
        jl = i - nc
        if jl < 0: jl = 0
        ju = i + nc
        if ju > n: ju = n
        sum = 0.0
        for j in range(jl, ju):
            k = np.fabs(j - i)
            sum = sum + float(datin[j]) * a[k]
        datout[i] = (sum)   # !!!! short int

    return datout

def fci(ffilter, fc):
    """
    Calculate frequency filter by which to filter a previously filtered (with 
    ffilter) data to get final fc. 
    """
    
    fci = 1. / sqrt((1.0 / (fc * fc)) - (1.0 / (ffilter * ffilter)))
    return fci

def resample(fc, srate):
    """
    Reduce 'sample rate' for output. If srate > 10*fc then recommend largest
    reduction that keeps srate at least 10fc. First reduction possible
    (without interpolation) is by factor of 2, so try only if srate>20fc
    """

    idelt = 1	#defoult
    r = srate / fc
    srate1 = srate		#value for output
    r1 = r
    i = 2
    while r1 >= 14:
        srate1 = srate / i
        r1 = srate1 / fc
        i += 1
    idelt = i
    if r1 < 10:
        idelt = i - 1
        srate1 = srate / idelt
        r1 = srate1 / fc
    
    #print ' Sample rate is ', r, ' times final fc:'
    #print ' Recommend reduction the sample rate for output file by using'
    #print ' only every',idelt,'th point to give sample rate of',srate1,'Hz'
    #print ' which is ', r1,' times fc.', 
    #print '  Keep every idelt''th point: idelt [', idelt,'] = '

    srate1 = srate / idelt
    
    return srate, idelt



