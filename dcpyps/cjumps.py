"""Library of routines for calculating responses to concentration jumps."""

__author__="remis"
__date__ ="$08-Nov-2011 21:43:14$"

import sys
from math import*

import numpy as np
from scipy.special import erf
import scipy.integrate as scpi

import qmatlib as qml

def dPdt(P, t, mec, cfunc, cargs):
    """
    Calculate derivativ of occupancies.
    dP/dt = P * Q

    Parameters
    ----------
    P : ndarray
        Occupancies.
    t : float
        Time.
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    cfunc : function
        Concentration profile.
    cargs : tuple
        Arguments for cfunc(t, cargs).

    Returns
    -------
    dpdt : ndarray
        Derivative of each state occupancy.
    """
    
    conc = cfunc(t, cargs)
    mec.set_eff('c', conc)
    dpdt = np.dot(P, mec.Q)
    return dpdt

def pulse_instexp(t, (cmax, cb, prepulse, tdec)):
    """
    Generate concentration pulse with instantaneous rise to maximal current
    and exponential decay.
    
    Parameters
    ----------
    t : ndarray or float
        Time samples.
    cmax : float
        Peak concentration.
    cb : float
        background concentration.
    prepulse : float
        Time before pulse starts.
    tdec : float
        Decay time constant.

    Returns
    -------
    c : ndarray
        Concentration profile.
    """

    if np.isscalar(t):
        if t <= prepulse:
            conc = 0.0
        else:
            conc = cmax * exp(-(t - prepulse) / tdec)
    else:
        t1 = np.extract(t[:] < prepulse, t)
        t2 = np.extract(t[:] >= prepulse, t)
        conc2 = cmax * np.exp(-(t2 - prepulse) / tdec)
        conc = np.append(t1 * 0.0, conc2)

    conc = conc + cb
    return conc

def pulse_erf(t, (cmax, cb, centre, width, rise, decay)):
    """
    Generate realistic concentration pulse with rise and fall from error function.

    Parameters
    ----------
    t : ndarray or float
        Time samples.
    cmax : float
        Peak concentration.
    cb : float
        background concentration.
    prepulse : float
        Time before pulse starts.
    width : float
        Pulse half width.
    rise : float
        Rise time constant for error function.
    decay : float
        Decay time constant for error function.

    Returns
    -------
    c : ndarray
        Concentration profile.
    """

    conc = (cmax * 0.5 *
        (erf((t - centre + width / 2.) / rise) -
        erf((t - centre - width / 2.) / decay)))
    conc = conc + cb
    return conc

def pulse_square(t, (cmax, cb, prepulse, pulse)):
    """
    Generate square pulse.

    Parameters
    ----------
    t : ndarray or float
        Time samples.
    cmax : float
        Peak concentration.
    cb : float
        background concentration.
    prepulse : float
        Time before pulse starts. 
    pulse : float
        Pulse half width.

    Returns
    -------
    c : ndarray
        Concentration profile.
    """

    if np.isscalar(t):
        if (t > prepulse) and (t <= (prepulse + pulse)):
            conc = cmax
        else:
            conc = 0.0
    else:
        t1 = t[np.where(t < prepulse)]
        t2 = t[np.where((t >= prepulse) & (t <= (prepulse + pulse)))]
        t3 = t[np.where(t > (prepulse + pulse))]
        c1 = cmax * np.ones(t2.shape)
        c2 = np.append(t1 * 0.0, c1)
        conc = np.append(c2, t3 * 0.0)

    conc = conc + cb
    return conc

def pulse_square_paired(t, (cmax, cb, prepulse, pulse, inter)):
    """
    Generate paired square pulses.

    Parameters
    ----------
    t : ndarray or float
        Time samples.
    cmax : float
        Peak concentration.
    cb : float
        background concentration.
    prepulse : float
        Time before first pulse starts.
    pulse : float
        Square pulse width.
    interpulse : float
        Time between two square pulses.

    Returns
    -------
    c : ndarray
        Concentration profile.
    """

    if np.isscalar(t):
        if (t >= prepulse) and (t <= (prepulse + pulse)):
            conc = cmax
        elif (t >= (prepulse + pulse + inter)) and (t <= (prepulse + 2 * pulse + inter)):
            conc = cmax
        else:
            conc = 0.0
    else:
        c1 = t[np.where(t < prepulse)] * 0.0
        t2 = t[np.where((t >= prepulse) & (t <= (prepulse + pulse)))]
        c2 = np.append(c1, cmax * np.ones(t2.shape))
        t3 = t[np.where((t > (prepulse + pulse)) & (t < (prepulse + pulse + inter)))]
        c3 = np.append(c2, t3 * 0.0)
        t4 = t[np.where((t >= (prepulse + pulse + inter)) & (t <= (prepulse + 2 * pulse + inter)))]
        c4 = np.append(c3, cmax * np.ones(t4.shape))
        t5 = t[np.where(t > (prepulse + 2 * pulse + inter))]
        conc = np.append(c4, t5 * 0.0)

    conc = conc + cb
    return conc

def solve_jump(mec, reclen, step, cfunc, cargs):
    """
    Calculate response to a concentration pulse by integration.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    reclen : float
        Trace length.
    step : float
        Sampling time interval.
    cfunc : function
        Concentration profile.
    cargs : tuple
        Arguments for cfunc(t, cargs).

    Returns
    -------
    t : ndarray
        Time samples.
    c : ndarray
        Concentration profile.
    P : ndarray
        All state occupancies.
    Popen : ndarray
        Open probability.
    """

    t = np.arange(0, reclen, step)
    mec.set_eff('c', cargs[1])
    P0 = qml.pinf(mec.Q)

    abserr = 1.0e-8
    relerr = 1.0e-6

    Pt = scpi.odeint(dPdt, P0, t, args=(mec, cfunc, cargs),
        atol=abserr,rtol=relerr)
    P = Pt.transpose()

    Popen = np.zeros(t.shape)
    for i in range(mec.kA):
        Popen += P[i]

    c =  cfunc(t, cargs)
    return t, c, Popen, P

def calc_jump (mec, reclen, step, cfunc, cargs):
    """
    Calculate response to a concentration pulse directly from Q matrix.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    reclen : float
        Trace length.
    step : float
        Sampling time interval.
    cfunc : function
        Concentration profile.
    cargs : tuple
        Arguments for cfunc(t, cargs).

    Returns
    -------
    t : ndarray
        Time samples.
    c : ndarray
        Concentration profile.
    P : ndarray
        All state occupancies.
    Popen : ndarray
        Open probability.
    """

    t = np.arange(0, reclen, step)
    c =  cfunc(t, cargs)
    
    mec.set_eff('c', cargs[1])
    pi = qml.pinf(mec.Q)
    Pt = np.array([pi.copy()])

    for i in range(1, t.shape[0]):

        mec.set_eff('c', c[i])
        w = coefficient_calc(mec.k, mec.A, pi)
        #loop over states to get occupancy of each
        for s in range(mec.k):
            # r is a running total over contributions of all components
            r = 0
            for ju, k in zip(w[:, s], mec.eigenvals):
                r += ju * np.exp(k * step)
            pi[s] = r
        Pt = np.append(Pt, [pi.copy()], axis=0)

    P = Pt.transpose()
    Popen = np.zeros(t.shape)
    for i in range(mec.kA):
        Popen += P[i]
        
    return t, c, Popen, P

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

def weighted_taus(mec, cmax, width, eff='c'):
    """
    Calculate weighted on and off time constants for a square concentration 
    pulse.
    
    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    cmax : float
        Pulse concentration.
    width : float
        Pulse width.

    Returns
    -------
    tau_on_weighted, tau_off_weighted : floats
        Weighted time constants.
    """
    
    mec.set_eff(eff, 0)
    P0 = qml.pinf(mec.Q)
    eigs0 = mec.eigenvals
    A0 = mec.A

    mec.set_eff(eff, cmax)
    eigsInf = mec.eigenvals
    Ainf = mec.A
    w_on = coefficient_calc(mec.k, Ainf, P0)

    Pt = np.zeros((mec.k))
    for i in range(mec.k):
        for ju, eg in zip(w_on[:, i], eigsInf):
            Pt[i] += np.dot(ju, np.exp(eg * width))

    ampl_on = np.zeros((mec.k))
    for i in range(mec.k):
        for j in range(mec.kA):
            ampl_on[i] += w_on[i,j]
    max_ampl_on = np.max(np.abs(ampl_on))
    rel_ampl_on = ampl_on / max_ampl_on

    tau_on_weighted = 0
    for i in range(mec.k-1):
        tau_on_weighted += -rel_ampl_on[i] * (-1 / eigsInf[i])
            
    w_off = coefficient_calc(mec.k, A0, Pt)
    ampl_off = np.zeros((mec.k))
    for i in range(mec.k):
        for j in range(mec.kA):
            ampl_off[i] += w_off[i,j]
    max_ampl_off = np.max(np.abs(ampl_off))
    rel_ampl_off = ampl_off / max_ampl_off

    tau_off_weighted = 0
    for i in range(mec.k-1):
        tau_off_weighted += rel_ampl_off[i] * (-1 / eigs0[i])
        
    return tau_on_weighted, tau_off_weighted

def printout(mec, cmax, width, output=sys.stdout, eff='c'):
    """
    """

    #TODO: on/off binding
    #TODO: move some of calculations from here to separate functions

    gamma = 30 # Conductance in pS
    Vm = -80e-3 # Transmembrane potential in V.

    mec.set_eff(eff, 0)
    P0 = qml.pinf(mec.Q)
    eigs0 = mec.eigenvals
    A0 = mec.A

    output.write('\n\nEquilibrium occupancies before t=0, at concentration = 0.0:')
    for i in range(mec.k):
        output.write('\np00({0:d}) = '.format(i+1) +
            '{0:.5g}'.format(P0[i]))

    mec.set_eff(eff, cmax)
    Pinf = qml.pinf(mec.Q)
    eigsInf = mec.eigenvals
    Ainf = mec.A
    w_on = coefficient_calc(mec.k, Ainf, P0)

    output.write('\n\nEquilibrium occupancies at maximum concentration = {0:.5g} mM:'
        .format(cmax * 1000))
    for i in range(mec.k):
        output.write('\npinf({0:d}) = '.format(i+1) +
            '{0:.5g}'.format(Pinf[i]))

    Pt = np.zeros((mec.k))
    for i in range(mec.k):
        for ju, eg in zip(w_on[:, i], eigsInf):
            Pt[i] += np.dot(ju, np.exp(eg * width))

    output.write('\n\nOccupancies at the end of {0:.5g} ms pulse:'.
        format(width * 1000))
    for i in range(mec.k):
        output.write('\npt({0:d}) = '.format(i+1) +
            '{0:.5g}'.format(Pt[i]))

    output.write('\n\nON-RELAXATION for ideal step:')
    output.write('\nTime course for current')
    output.write('\n\nComp\tEigen\t\tTau (ms)')
    for i in range(mec.k-1):
        output.write('\n{0:d}\t'.format(i+1) +
            '{0:.5g}\t\t'.format(eigsInf[i]) +
            '{0:.5g}\t'.format(-1000 / eigsInf[i])) # convert to ms

    ampl_on = np.zeros((mec.k))
    for i in range(mec.k):
        for j in range(mec.kA):
            ampl_on[i] += w_on[i,j]
    cur_on = ampl_on * gamma * Vm
    max_ampl_on = np.max(np.abs(ampl_on))
    rel_ampl_on = ampl_on / max_ampl_on
    area_on = np.zeros((mec.k-1))

    tau_on_weighted = 0
    output.write('\n\nAmpl.(t=0,pA)\tRel.ampl.\t\tArea(pC)')
    for i in range(mec.k-1):
        area_on[i] = -1000 * cur_on[i] / eigsInf[i]
        output.write('\n{0:.5g}\t\t'.format(cur_on[i]) +
            '{0:.5g}\t\t'.format(rel_ampl_on[i]) +
            '{0:.5g}\t'.format(area_on[i]))
        tau_on_weighted += -rel_ampl_on[i] * (-1000 / eigsInf[i])
            
    output.write('\n\nWeighted On Tau (ms) = {0:.5g}'.format(tau_on_weighted))

    output.write('\n\nTotal current at t=0 (pA) = {0:.5g}'.
        format(np.sum(cur_on)))
    output.write('\nTotal current at equilibrium (pA) = {0:.5g}'.
        format(cur_on[-1]))
    output.write('\nTotal area (pC) = {0:.5g}'.
        format(np.sum(area_on)))

    #TODO: Current at the end of pulse
    ct = cur_on[:-1] * np.exp(width * eigsInf[:-1])

    output.write('\nCurrent at the end of {0:.5g}'.format(width
        * 1000) + ' ms pulse = {0:.5g}'.format(np.sum(ct) + cur_on[-1]))

    # Calculate off- relaxation.
    output.write('\n\nOFF-RELAXATION for ideal step:')
    output.write('\nTime course for current')
    output.write('\n\nComp\tEigen\t\tTau (ms)')
    for i in range(mec.k-1):
        output.write('\n{0:d}\t'.format(i+1) +
            '{0:.5g}\t\t'.format(eigs0[i]) +
            '{0:.5g}\t'.format(-1000 / eigs0[i]))

    w_off = coefficient_calc(mec.k, A0, Pt)
    ampl_off = np.zeros((mec.k))
    for i in range(mec.k):
        for j in range(mec.kA):
            ampl_off[i] += w_off[i,j]
    cur_off = ampl_off * gamma * Vm
    max_ampl_off = np.max(np.abs(ampl_off))
    rel_ampl_off = ampl_off / max_ampl_off
    area_off = np.zeros((mec.k-1))

    tau_off_weighted = 0
    output.write('\n\nAmpl.(t=0,pA)\tRel.ampl.\t\tArea(pC)')
    for i in range(mec.k-1):
        area_off[i] = -1000 * cur_off[i] / eigs0[i]
        output.write('\n{0:.5g}\t\t'.format(cur_off[i]) +
            '{0:.5g}\t\t'.format(rel_ampl_off[i]) +
            '{0:.5g}\t'.format(area_off[i]))
        tau_off_weighted += rel_ampl_off[i] * (-1000 / eigs0[i])
            
    output.write('\n\nWeighted Off Tau (ms) = {0:.5g}'.format(tau_off_weighted))

    output.write('\n\nTotal current at t=0 (pA) = {0:.5g}'.
        format(np.sum(cur_off)))
    output.write('\nTotal current at equilibrium (pA) = {0:.5g}'.
        format(cur_off[-1]))
    output.write('\nTotal area (pC) = {0:.5g}'.
        format(np.sum(area_off)))
