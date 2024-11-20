"""Library of routines for calculating responses to concentration jumps."""

__author__="remis"
__date__ ="$08-Nov-2011 21:43:14$"

import numpy as np
import scipy.integrate as scpi
from scipy.special import erf

from scalcs import qmatlib as qml

##### Calculate occupancies and rate of change of occupancies #####
##### using Q-matrix formalism #####

def dPdt(P, t, mec, cfunc, cargs):
    """
    Calculate the rate of change (derivative) of state occupancies:

    dP/dt = P * Q

    Parameters
    ----------
    P : ndarray
        Vector containing occupancies of each of states.
    t : float
        Time from the start.
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    cfunc : function
        Concentration profile.
    cargs : tuple
        Arguments for cfunc(t, cargs).

    Returns
    -------
    dp/dt : ndarray
        Derivative of each state occupancy.
    """
    
    mec.set_eff('c', cfunc(t, cargs))
    return np.dot(P, mec.Q)

def P_t(t, eigs, w):
    """
    Calculate occupancies at given time.

    Parameters
    ----------
    t : float
        Time from the start.
    eigs : ndarray
        Vector containing eigenvalues.
    w : ndarray
        The amplitudes of each k-1 component.

    Returns
    -------
    Pt : ndarray
        Vector containing occupancies of each of k states.
    """
    Pt = np.zeros((eigs.shape))
    for i in range(eigs.size):
        Pt[i] = np.sum(w[:, i] * np.exp(eigs * t))
    #Pt = np.sum(w * np.exp(eigs * t).reshape(w.shape[0],1,1), axis=1)
    return Pt

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

##### Calculate macroscopic current response #####

def solve_jump(mec, reclen, step, cfunc, cargs, abserr=1.0e-8, relerr=1.0e-6):
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
    rtol, atol : float, optional
        Tolerance limits for the error control performed by the scipy.odeint solver.

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
    Pt = scpi.odeint(dPdt, P0, t, args=(mec, cfunc, cargs),
        atol=abserr,rtol=relerr)
    P = Pt.transpose()
    Popen = np.sum(P[: mec.kA], axis=0)
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
        eigenvals, A = qml.eigenvalues_and_spectral_matrices(mec.Q)
        w = coefficient_calc(mec.k, A, pi)
        pi = P_t(step, eigenvals, w)
        Pt = np.append(Pt, [pi.copy()], axis=0)

    P = Pt.transpose()
    Popen = np.sum(P[: mec.kA], axis=0)
    return t, c, Popen, P



##### Concentration pulse profiles #####

def pulse_instexp(t, pars):
#def pulse_instexp(t, (cmax, cb, prepulse, tdec)):
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
    
    cmax, cb, prepulse, tdec = pars

    if np.isscalar(t):
        if t <= prepulse:
            conc = 0.0
        else:
            conc = cmax * np.exp(-(t - prepulse) / tdec)
    else:
        t1 = np.extract(t[:] < prepulse, t)
        t2 = np.extract(t[:] >= prepulse, t)
        conc2 = cmax * np.exp(-(t2 - prepulse) / tdec)
        conc = np.append(t1 * 0.0, conc2)

    return conc + cb

def pulse_erf(t, pars):
#def pulse_erf(t, (cmax, cb, centre, width, rise, decay)):
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

    cmax, cb, centre, width, rise, decay = pars
    conc = (cmax * 0.5 *
        (erf((t - centre + width / 2.) / rise) -
        erf((t - centre - width / 2.) / decay)))
    return conc + cb

def pulse_square(t, pars):
#def pulse_square(t, (cmax, cb, prepulse, pulse)):
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
    
    cmax, cb, prepulse, pulse = pars
    if np.isscalar(t):
        conc = cmax if ((t > prepulse) and (t <= (prepulse + pulse))) else 0.0
    else:
        t1 = t[np.where(t < prepulse)]
        t2 = t[np.where((t >= prepulse) & (t <= (prepulse + pulse)))]
        t3 = t[np.where(t > (prepulse + pulse))]
        c1 = cmax * np.ones(t2.shape)
        c2 = np.append(t1 * 0.0, c1)
        conc = np.append(c2, t3 * 0.0)

    return conc + cb

def pulse_square_paired(t, pars):
#def pulse_square_paired(t, (cmax, cb, prepulse, pulse, inter)):
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

    cmax, cb, prepulse, pulse, inter = pars
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

    return conc + cb

##### Printout and related utility functions. #####
# TODO: need drastic refactoring

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
    eigs0, A0 = qml.eigenvalues_and_spectral_matrices(mec.Q)
    P0 = qml.pinf(mec.Q)
    mec.set_eff(eff, cmax)
    eigsInf, Ainf = qml.eigenvalues_and_spectral_matrices(mec.Q)
    w_on = coefficient_calc(mec.k, Ainf, P0)
    Pt = P_t(width, eigsInf, w_on)
    w_off = coefficient_calc(mec.k, A0, Pt)

    ampl_on = np.sum(w_on[:, :mec.kA], axis=1)
    max_ampl_on = np.max(np.abs(ampl_on))
    rel_ampl_on = ampl_on / max_ampl_on
    tau_on_weighted = np.sum(-rel_ampl_on[:-1] * (-1 / eigsInf[:-1]))
#    tau_on = -1 / eigsInf[:-1]

    ampl_off = np.sum(w_off[:, :mec.kA], axis=1)
    max_ampl_off = np.max(np.abs(ampl_off))
    rel_ampl_off = ampl_off / max_ampl_off
    tau_off_weighted = np.sum(rel_ampl_off[: -1] * (-1 / eigs0[:-1]))
#    tau_off = -1 / eigs0[:-1]

    #return tau_on_weighted, tau_on, tau_off_weighted, tau_off
    return tau_on_weighted, tau_off_weighted

def printout(mec, cmax, width, eff='c'):
    """
    """
    #TODO: on/off binding
    #TODO: move some of calculations from here to separate functions
    
    str_out = ('\n*******************************************\n' +
        'CONCENTRATION JUMPS\n')

    gamma = 30 # Conductance in pS
    Vm = -80e-3 # Transmembrane potential in V.

    mec.set_eff(eff, 0)
    P0 = qml.pinf(mec.Q)
    eigs0, A0 = qml.eigenvalues_and_spectral_matrices(mec.Q)
    str_out += ('\nEquilibrium occupancies before t=0, at concentration = 0.0:\n')
    for i in range(mec.k):
        str_out += ('p00({0:d}) = {1:.5g}\n'.format(i+1, P0[i]))

    mec.set_eff(eff, cmax)
    Pinf = qml.pinf(mec.Q)
    eigsInf, Ainf = qml.eigenvalues_and_spectral_matrices(mec.Q)
    w_on = coefficient_calc(mec.k, Ainf, P0)
    str_out += ('\nEquilibrium occupancies at maximum concentration = {0:.5g} mM:\n'
        .format(cmax * 1000))
    for i in range(mec.k):
        str_out += ('pinf({0:d}) = '.format(i+1) + '{0:.5g}\n'.format(Pinf[i]))

    Pt = P_t(width, eigsInf, w_on)
    str_out += ('\nOccupancies at the end of {0:.5g} ms pulse:\n'.
        format(width * 1000))
    for i in range(mec.k):
        str_out += ('pt({0:d}) = '.format(i+1) + '{0:.5g}\n'.format(Pt[i]))

    tau_on_weighted, tau_off_weighted = weighted_taus(mec, cmax, width, eff='c')

    str_out += ('\nON-RELAXATION for ideal step:\n' +
        'Time course for current\n' +
        '\nComp\tEigen\t\tTau (ms)\n')
    for i in range(mec.k-1):
        str_out += ('{0:d}\t'.format(i+1) +
            '{0:.5g}\t\t'.format(eigsInf[i]) +
            '{0:.5g}\t\n'.format(-1000 / eigsInf[i])) # convert to ms

    ampl_on = np.sum(w_on[:, :mec.kA], axis=1)
    cur_on = ampl_on * gamma * Vm
    max_ampl_on = np.max(np.abs(ampl_on))
    rel_ampl_on = ampl_on / max_ampl_on
    area_on = -cur_on[:-1] / eigsInf[:-1]
    str_out += ('\nAmpl.(t=0,pA)\tRel.ampl.\t\tArea(pC)\n')
    for i in range(mec.k-1):
        str_out += ('{0:.5g}\t\t'.format(cur_on[i]) +
            '{0:.5g}\t\t'.format(rel_ampl_on[i]) +
            '{0:.5g}\t\n'.format(area_on[i] * 1000))

    str_out += ('\nWeighted On Tau (ms) = {0:.5g}\n'.format(tau_on_weighted * 1000))
    str_out += ('\nTotal current at t=0 (pA) = {0:.5g}\n'.
        format(np.sum(cur_on)))
    str_out += ('Total current at equilibrium (pA) = {0:.5g}\n'.
        format(cur_on[-1]))
    str_out += ('Total area (pC) = {0:.5g}\n'.
        format(np.sum(area_on)))
    #TODO: Current at the end of pulse
    ct = cur_on[:-1] * np.exp(width * eigsInf[:-1])
    str_out += ('Current at the end of {0:.5g}'.format(width
        * 1000) + ' ms pulse = {0:.5g}\n'.format(np.sum(ct) + cur_on[-1]))

    # Calculate off- relaxation.
    str_out += ('\nOFF-RELAXATION for ideal step:\n' +
        'Time course for current\n' +
        '\nComp\tEigen\t\tTau (ms)\n')
    for i in range(mec.k-1):
        str_out += ('{0:d}\t'.format(i+1) +
            '{0:.5g}\t\t'.format(eigs0[i]) +
            '{0:.5g}\t\n'.format(-1000 / eigs0[i]))

    w_off = coefficient_calc(mec.k, A0, Pt)
    ampl_off = np.sum(w_off[:, :mec.kA], axis=1)
    cur_off = ampl_off * gamma * Vm
    max_ampl_off = np.max(np.abs(ampl_off))
    rel_ampl_off = ampl_off / max_ampl_off
    area_off = np.zeros((mec.k-1))
    str_out += ('\nAmpl.(t=0,pA)\tRel.ampl.\t\tArea(pC)\n')
    for i in range(mec.k-1):
        area_off[i] = -1000 * cur_off[i] / eigs0[i]
        str_out += ('{0:.5g}\t\t'.format(cur_off[i]) +
            '{0:.5g}\t\t'.format(rel_ampl_off[i]) +
            '{0:.5g}\t\n'.format(area_off[i]))
            
    str_out += ('\nWeighted Off Tau (ms) = {0:.5g}\n'.format(tau_off_weighted * 1000))
    str_out += ('\nTotal current at t=0 (pA) = {0:.5g}\n'.
        format(np.sum(cur_off)))
    str_out += ('Total current at equilibrium (pA) = {0:.5g}\n'.
        format(cur_off[-1]))
    str_out += ('Total area (pC) = {0:.5g}\n'.format(np.sum(area_off)))
 
    return str_out

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
        wton[i], ton[i], wtoff[i], toff[i] = weighted_taus(mec, c[i], width)

    ton = ton.transpose()
    toff = toff.transpose()

    return c * 1000, wton * 1000, ton * 1000, wtoff * 1000, toff * 1000
