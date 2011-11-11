"""Library of routines for calculating responses to concentration jumps."""

__author__="Andrew Plested"
__date__ ="$Dec 20, 2010 12:07:36 PM$"

import sys
from math import*

import numpy as np
from scipy.special import erf
import scipy.integrate as scpi

import qmatlib as qml

def erf_pulse(pulse_width, pulse_conc, rise_t, pulse_centre, t_step):
    """
    Construct a "top-hat" pulse with rise and fall from error function.

    Parameters
    ----------
    pulse_width : int
        Approximate FWHM in microseconds.
    pulse_conc : float
        Ligand concentration during jump in Molar.
    rise_t : float
        Desired 10 - 90% rise time (can use 0. for square pulse).
    centre : int
        Position of pulse in simulation trace.
    time_step : int
        Sampling interval in microseconds.

    Returns
    -------
    z : dictionary (key- float, value- float)
        Dictionary keys- time points in microseconds.
        Dictionary values- concentration in moles.
    """

    erf_rise = 1.8  #10-90% of erf(x)
    step_factor = float(rise_t) / ( erf_rise * t_step )

    z = {}

    # lock to sampling intervals
    left_fwhm  = int((pulse_centre - (float(pulse_width) / 2)) / t_step)*t_step
    right_fwhm = int((pulse_centre + (float(pulse_width) / 2)) / t_step)*t_step

    # fill in with true top hat at pulse_conc - will overwrite flanks with erf
    for r in range(left_fwhm, right_fwhm, t_step):

        z [r] = pulse_conc

    # Return a perfect top hat if rise time is 0
    # or shorter than sampling interval
    if rise_t < t_step :
        return z

    #reduce number of points but scale with rise_t
    pts = int(float(rise_t) / 3) 
    for s in range(-pts, pts + 1): #symmetrical

        # scale x into correct intervals
        x = float(s) / step_factor
        # y-shift erf to 0 to 1; scale by concentration
        y = (erf(x) + 1.) / 2. * pulse_conc          

        # flatten differences under 10 pM during constant-concentration sections
        if y < 10e-12:
            y = 0.

        if y > pulse_conc - 10e-12:
            y = pulse_conc

        #inflexion points centred on left_fwhm, right_fwhm
        t_rise = left_fwhm + x * step_factor * t_step
        t_fall = right_fwhm + x * step_factor * t_step

        # truncate rise and fall if pulse is too thin to reach max.
        if t_rise <= pulse_centre:
            z[t_rise] = y

        if t_fall >= pulse_centre:
            z[t_fall] = pulse_conc-y

    return z

def rcj_single(mec, parameters):
    """
    Calculate a single concentration jump and a single relaxation.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    parameters : dictionary
        Dictioanry contains parameters describing the concentration jump.

    Returns
    -------
    cjump : dictionary
        Contains time sample point (microsec) - concentration (M) pairs.
    relax : dictionary
    """

    cjump = make_jump(parameters)
    relax = rcj_calc (cjump, mec, parameters)

    return cjump, relax

def make_family():
    """
    testing - make a family of pulses to examine properties
    alternate main()
    no arguments
    returns nothing
    """

    center = 10000
    width = 2500. #desired 10-90% in microseconds
    family = []
    peak_conc = 30e-3
    step_size = 8
    max_length =0
    for rise_time in range(50,500,50):

        profile = erf_pulse(width,peak_conc,rise_time,center,step_size)
        family.append(profile)

        if len(profile) > max_length:
            max_length = len (profile)

    rcj_fileops.family_write(family,max_length)

def make_jump(pa):
    """
    Generate a dictionary containing concentration jump profile.

    Parameters
    ----------
    pa : dictionary
        Parameters describing concentration pulse.

    Returns
    -------
    cjump : dictionary
        Time sample point (microsec)- concentration (Molar) pairs.
    """

    cjump = {}

    pw = pa['pulse_width']
    pc = pa['peak_conc']
    rt = pa['rise_time']
    pce= pa['pulse_centre']
    ss = pa['step_size']
    rl = pa['record_length']

    for t in range(0,rl,ss):
        cjump [float(t)] = 0.     # fill dictionary with blanks

    profile = erf_pulse(pw, pc, rt, pce, ss )
    cjump.update(profile)        # overwrite blank dictionary keys with pulse values
    return cjump

def rcj_calc (cjump, mec, paras, eff='c'):
    """
    Parameters
    ----------
    cjump : dictionary
        Time sample point (microsec)- concentration (Molar) pairs.
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    paras : dictionary
        Parameters describing concentration pulse.

    Returns
    -------
    relax : dictionary
        Time sample point (microsec)- P pairs. P is numpy array (k, 1) which
        elements are state occupancies.
    """

    relax = {}
    dt = paras['step_size'] * 1.e-6  # Convert from microseconds to seconds.
    firststep = True

    # Sort the dictionary to get time elements in order.
    for t_point in sorted(cjump):
        # Extract concentration.
        c = cjump [t_point]
        mec.set_eff(eff, c)

        if not firststep:
            # Not the first step, so use P from last step to calculate
            # new occupancy. Note that P is not used again in the calculation,
            # so can be updated safely.
            w = coefficient_calc(mec.k, mec.A, P)

            #loop over states to get occupancy of each
            for s in range(mec.k):
                # r is a running total over contributions of all components
                r = 0
                for ju, k in zip(w[:, s], mec.eigenvals):
                    r += ju * np.exp(k * dt)
                P[s] = r
        else:
            # First iteration, just calculate Pinf at initial concentration.
            P = qml.pinf(mec.Q)
            firststep = False

        # Must copy P, otherwise every value in dictionary = P (entire
        # output has last value!)
        relax [t_point] = P.copy()
    return relax

def coefficient_calc(k, A, p_occup):
    """
    Calculate weighted components for relaxation for each state p * An.

    Parameters
    ----------
    k : int
        Number of states in mechanism.
    A : array-like, shape (k, k, k)
        Spectral matrices of Q matrix.
    p_occup : array-like, shape (k, 1)
        Occupancies of mechanism states.

    Returns
    -------
    w : ndarray, shape (k, k)
    """

    w = np.zeros((k, k))
    for n in range (k):
        w[n, :] = np.dot(p_occup, A[n, :, :])
    return w

def compose_rcj_out (cjump, relax, kA, offset=1.2, output_option=2):
    """
    Write output to text file as lines of t,j,p0,p1,p2....pn,pOpen with
    option to omit arguments.

    Parameters
    ----------
    cjump : dictionary
        Time sample point (microsec)- concentration (Molar) pairs.
    relax : dictionary
        Time sample point (microsec)- P pairs. P is numpy array (k, 1) which
        elements are state occupancies.
    kA : int
        Number of open states in mechanism.
    offset : float
        Offset the jump from the response.
    output_option : int
        2 : give occupancies and Popen;
        1 : Popen only;
        0 : occupancies only.

    Returns
    -------
        lines : a list of strings
    """

    lines = []
    for time_key in sorted(relax):
        line = str(time_key)+'\t'+str(cjump[time_key]+offset)
        isochrone = relax[time_key]

        if output_option != 1:
            #Full occupancy
            for elem in isochrone:
                line += '\t'+str(elem)
        if output_option != 0:
            #Open probability
            Popen = 0.
            for o in range(kA):
                Popen = Popen + isochrone[o]
            line += '\t'+str(Popen)

        lines.append(line)
    return lines

def convert_to_arrays_Popen(cjump, relax, kA):
    """
    Convert concentration profile and relaxation dictionaries into arrays.

    Parameters
    ----------
    cjump : dictionary
        Time sample point (microsec)- concentration (Molar) pairs.
    relax : dictionary
        Time sample point (microsec)- P pairs. P is numpy array (k, 1) which
        elements are state occupancies.
    kA : int
        Number of open states in mechanism.

    Returns
    -------
    t : ndarray, shape(dict length, 1)
        Time points in microsec.
    cj : ndarray, shape(dict length, 1)
        Concentration profile in Molar.
    rl : ndarray, shape(dict length, 1)
        Open probaility relaxation.
    """

    t = np.zeros(len(cjump))
    cj = np.zeros(len(cjump))
    rl = np.zeros(len(cjump))
    index = 0

    for time_key in sorted(relax):
        t[index] = time_key
        cj[index] = cjump[time_key] * 1.01

        #Open Probability
        for o in range(kA):
            rl[index] += relax[time_key][o]
        index +=1

    return t, cj, rl

def convert_to_arrays_all(cjump, relax, k):
    """
    Convert concentration profile and relaxation dictionaries into arrays.

    Parameters
    ----------
    cjump : dictionary
        Time sample point (microsec)- concentration (Molar) pairs.
    relax : dictionary
        Time sample point (microsec)- P pairs. P is numpy array (k, 1) which
        elements are state occupancies.
    k : int
        Number of states in mechanism.

    Returns
    -------
    t : ndarray, shape(dict length, 1)
        Time points in microsec.
    cj : ndarray, shape(dict length, 1)
        Concentration profile in Molar.
    rl : ndarray, shape(dict length, 1)
        Open probaility relaxation.
    """

    t = np.zeros(len(cjump))
    cj = np.zeros(len(cjump))
    rl = np.zeros((k, len(cjump)))
    index = 0

    for time_key in sorted(relax):
        t[index] = time_key
        cj[index] = cjump[time_key] * 1.01

        for o in range(k):
            rl[o, index] = relax[time_key][o]
        index +=1

    return t, cj, rl
