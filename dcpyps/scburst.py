"""A collection of functions for single channel burst calculations.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 20:29:14$"

import sys
import math

import numpy as np
from numpy import linalg as nplin

import qmatlib as qml

def phiBurst(mec):
    """
    Calculate the start probabilities of a burst (Eq. 3.2, CH82).
    PhiB = (pCinf * (QCB * GBA + QCA)) / (pCinf * (QCB * GBA + QCA) * uA)

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    phiB : array_like, shape (1, kA)
    """

    uA = np.ones((mec.kA, 1))
    pC = qml.pinf(mec.Q)[mec.kE:]
    nom = np.dot(pC, (np.dot(mec.QCB, mec.GBA) + mec.QCA))
    denom = np.dot(nom, uA)
    phiB = nom / denom
    return phiB

def endBurst(mec):
    r"""
    Calculate the end vector for a burst (Eq. 3.4, CH82).

    .. math::

       \bs{e}_\text{b} = (\bs{I}-\bs{G}_\cl{AB} \bs{G}_\cl{BA}) \bs{u}_\cl{A}

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    eB : array_like, shape (kA, 1)
    """

    uA = np.ones((mec.kA, 1))
    I = np.eye(mec.kA)
    eB = np.dot((I - np.dot(mec.GAB, mec.GBA)), uA)
    return eB

def length_mean(mec):
    """
    Calculate the mean burst length (Eq. 3.19, CH82).
    m = PhiB * (I - GAB * GBA)^(-1) * (-QAA^(-1)) * \
        (I - QAB * (QBB^(-1)) * GBA) * uA

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    m : float
        The mean burst length.
    """

    uA = np.ones((mec.kA, 1))
    I = np.eye(mec.kA)
    invQAA = -1 * nplin.inv(mec.QAA)
    invQBB = nplin.inv(mec.QBB)
    interm1 = nplin.inv(I - np.dot(mec.GAB, mec.GBA))
    interm2 = I - np.dot(np.dot(mec.QAB, invQBB), mec.GBA)
    m = (np.dot(np.dot(np.dot(np.dot(phiBurst(mec), interm1), invQAA),
        interm2), uA)[0])
    return m

def open_time_mean(mec):
    """
    Calculate the mean total open time per burst (Eq. 3.26, CH82).

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    m : float
        The mean total open time per burst.
    """

    uA = np.ones((mec.kA, 1))
    VAA = mec.QAA + np.dot(mec.QAB, mec.GBA)
    m = np.dot(np.dot(phiBurst(mec), -nplin.inv(VAA)), uA)[0]
    return m

def openings_mean(mec):
    """
    Calculate the mean number of openings per burst (Eq. 3.7, CH82).
    mu = phiB * (I - GAB * GBA)^(-1) * uA

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    mu : float
        The mean number ofopenings per burst.
    """

    uA = np.ones((mec.kA,1))
    I = np.eye(mec.kA)
    interm = nplin.inv(I - np.dot(mec.GAB, mec.GBA))
    mu = np.dot(np.dot(phiBurst(mec), interm), uA)[0]
    return mu

def openings_distr(mec, r):
    """
    The distribution of openings per burst (Eq. 3.5, CH82).
    P(r) = phiB * (GAB * GBA)^(r-1) * eB

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    r : int
        Number of openings per burst.

    Returns
    -------
    Pr : float
        Probability of seeing r openings per burst.
    """

    GG = np.dot(mec.GAB, mec.GBA)
    if r == 1:
        interm = np.eye(mec.kA)
    elif r == 2:
        interm = GG
    else:
        interm = GG
        for i in range(2, r):
            interm = np.dot(interm, GG)
    Pr = np.dot(np.dot(phiBurst(mec), interm), endBurst(mec))
    return Pr

def openings_cond_distr_depend_on_start_state(mec, r):
    """
    The distribution of openings per burst coditional on starting state.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    r : int
        Number of openings per burst.

    Returns
    -------
    vecPr : array_like, shape (kA, 1)
        Probability of seeing r openings per burst depending on starting state.
    """

    GG = np.dot(mec.GAB, mec.GBA)
    if r == 1:
        interm = np.eye(mec.kA)
    elif r == 2:
        interm = GG
    else:
        interm = GG
        for i in range(2, r):
            interm = np.dot(interm, GG)
    vecPr = np.dot(interm, endBurst(mec))
    vecPr = vecPr.transpose()
    return vecPr

def length_pdf(mec, t):
    """
    Probability density function of the burst length (Eq. 3.17, CH82).
    f(t) = phiB * [PEE(t)]AA * (-QAA) * eB, where PEE(t) = exp(QEE * t)

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    f : float
    """

    expQEEA = qml.expQt(mec.QEE, t)[:mec.kA, :mec.kA]
    f = np.dot(np.dot(np.dot(phiBurst(mec), expQEEA), -mec.QAA),
        endBurst(mec))
    return f

def length_cond_pdf(mec, t):
    """
    The distribution of burst length coditional on starting state.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    t : float
        Length.

    Returns
    -------
    vec : array_like, shape (kA, 1)
        Probability of seeing burst length t depending on starting state.
    """

    expQEEA = qml.expQt(mec.QEE, t)[:mec.kA, :mec.kA]
    vec = np.dot(np.dot(expQEEA, -mec.QAA), endBurst(mec))
    vec = vec.transpose()
    return vec

def length_pdf_components(mec):
    """
    Calculate time constants and areas for an ideal (no missed events)
    exponential burst length probability density function.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    taus : ndarray, shape(k, 1)
        Time constants.
    areas : ndarray, shape(k, 1)
    """

    areas = np.zeros(mec.kE)
    eigs, A = qml.eigs(-mec.QEE)
    for i in range(mec.kE):
        areas[i] = (np.dot(np.dot(np.dot(phiBurst(mec),
            A[i][:mec.kA, :mec.kA]), (-mec.QAA)), endBurst(mec)) / eigs[i])

    taus = 1 / eigs
    return taus, areas

def printout(mec, output=sys.stdout, eff='c'):
    """
    Output burst calculations into selected device (sys.stdout, printer, file,
    text field.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    output : output device
        Default device: sys.stdout
    eff : string
        Effector; e.g. 'c'- concentration.
    """

    phiB = phiBurst(mec)
    output.write('\n\nInitial vector for burst (phiB) = \n')
    for i in range(mec.kA):
        output.write('{0:.6f}\t'.format(phiB[i]))
    
    tau, area = length_pdf_components(mec)
    output.write('\n\nBURST LENGTH DISTRIBUTION')
    output.write('\nterm\ttau (ms)\tarea (%)')
    for i in range(mec.kE):
        output.write('\n{0:d}'.format(i+1) +
            '\t{0:.3f}'.format(tau[i] * 1000) +
            '\t{0:.3f}'.format(area[i] * 100))

    m = length_mean(mec)
    output.write('\n\nMean burst length = {0:.3f} millisec'.
        format(m * 1000))
        
    mu = openings_mean(mec)
    output.write('\n\nMean number of openings per burst = {0:.3f}'.
        format(mu))
        
    mop = open_time_mean(mec)
    output.write('\n\nThe mean total open time per burst = {0:.3f} '.
        format(mop * 1000) + 'millisec')
        
    bpop = mop / m
    output.write('\nPopen WITHIN BURST = (open time/bst)/(bst length)\
        = {0:.3f} \n'.format(bpop))

