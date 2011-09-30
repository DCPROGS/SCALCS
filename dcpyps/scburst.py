"""A collection of functions for single channel burst calculations.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 20:29:14$"

import sys
import math

import numpy as np
from numpy import linalg as nplin

import qmatlib as qml
import pdfs

def phiBurst(mec):
    """
    Calculate the start probabilities of a burst (Eq. 3.2, CH82).
    phiB = (pCinf * (QCB * GBA + QCA)) / (pCinf * (QCB * GBA + QCA) * uA)

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
    eigs : ndarray, shape(k, 1)
        Time constants.
    w : ndarray, shape(k, 1)
        Component amplitudes.
    """

    w = np.zeros(mec.kE)
    eigs, A = qml.eigs(-mec.QEE)
    for i in range(mec.kE):
        w[i] = np.dot(np.dot(np.dot(phiBurst(mec),
            A[i][:mec.kA, :mec.kA]), (-mec.QAA)), endBurst(mec))
    return eigs, w

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

def openings_distr_components(mec):
    """
    Calculate coeficients for geometric ditribution P(r)- probability of
    seeing r openings (Eq. 3.9 CH82):
    P(r) = sum(W * rho^(r-1))
    where w
    wm = phiB * Am * endB (Eq. 3.10 CH82)
    and rho- eigenvalues of GAB * GBA.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    r : int
        Number of openings per burst.

    Returns
    -------
    rho : ndarray, shape (kA,)
    w : ndarray, shape (kA,)
    """

    GG = np.dot(mec.GAB, mec.GBA)
    rho, A = qml.eigs(GG)
    w = np.dot(np.dot(phiBurst(mec), A), endBurst(mec)).transpose()[0]
    return rho, w

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

def open_time_total_pdf_components(mec):
    """
    Eq. 3.23, CH82
    """

    VAA = mec.QAA + np.dot(mec.QAB, mec.GBA)
    eigs, A = qml.eigs(-VAA)
    uA = np.ones((mec.kA, 1))

    w = np.zeros(mec.kA)
    for i in range(mec.kA):
        w[i] = np.dot(np.dot(np.dot(phiBurst(mec), A[i]), (-VAA)), uA)

    return eigs, w

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

def shut_times_between_burst_pdf_components(mec):
    """
    """

    uA = np.ones((mec.kA, 1))
    uC = np.ones((mec.kC, 1))
    eigsB, AmatB = qml.eigs(-mec.QBB)
    eigsF, AmatF = qml.eigs(-mec.QFF)
    pA = qml.pinf(mec.Q)[:mec.kA]
    GBC = -np.dot(nplin.inv(mec.QBB), mec.QBC)
    end = np.dot((np.dot(mec.QAB, GBC) + mec.QAC), uC)
    start = pA / np.dot(pA, end)

    rowB = np.dot(start, mec.QAB)
    rowF = np.dot(start, mec.QAF)
    colB = np.dot(mec.QBA, uA)
    colF = np.dot(mec.QFA, uA)
    wB = -np.dot(np.dot(rowB, AmatB), colB)
    wF = np.dot(np.dot(rowF, AmatF), colF)

    w = np.append(wB, wF)
    eigs = np.append(eigsB, eigsF)
    return eigs, w

def shut_time_total_pdf_components(mec):
    """
    Eq. 3.40, CH82
    """

    WBB = mec.QBB + np.dot(mec.QBA, mec.GAB)
    eigs, A = qml.eigs(-WBB)
    norm = 1 - np.dot(phiBurst(mec), endBurst(mec))[0]

    w = np.zeros(mec.kB)
    for i in range(mec.kB):
        w[i] = np.dot(np.dot(np.dot(np.dot(phiBurst(mec), mec.GAB),
            A[i]), (mec.QBA)), endBurst(mec)) / norm

    return eigs, w

def first_opening_length_pdf_components(mec):
    """
    Calculate time constants and amplitudes for an ideal (no missed events)
    pdf of first opening in a burst with 2 or more openings.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    eigs : ndarray, shape(k, 1)
        Time constants.
    w : ndarray, shape(k, 1)
        Component amplitudes.
    """

    uA = np.ones((mec.kA, 1))
    eigs, A = qml.eigs(-mec.QAA)
    GG = np.dot(mec.GAB, mec.GBA)
    norm = np.dot(np.dot(phiBurst(mec), GG), uA)[0]

    w = np.zeros(mec.kA)
    for i in range(mec.kA):
        w[i] = np.dot(np.dot(np.dot(np.dot(phiBurst(mec),
            A[i]), (-mec.QAA)), GG), uA) / norm

    return eigs, w

def printout(mec, output=sys.stdout):
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

    output.write('\n\n*******************************************\n')
    output.write('CALCULATED SINGLE CHANNEL BURST PDFS ETC....\n')

    # # #
    phiB = phiBurst(mec)
    output.write('\nInitial vector for burst (phiB) = \n')
    for i in range(mec.kA):
        output.write('{0:.6f}\t'.format(phiB[i]))
    endB = endBurst(mec)
    output.write('\nEnd vector for burst (endB) = \n')
    for i in range(mec.kA):
        output.write('{0:.6f}\t'.format(endB[i, 0]))

    # # #
    eigs, w = length_pdf_components(mec)
    output.write('\n\nTotal burst length, unconditional pdf')
    output.write('\nFbst(t) =')
    pdfs.expPDF_printout(eigs, w, output)
    m = length_mean(mec)
    output.write('\nMean from direct matrix calc = {0:.3f} millisec'.
        format(m * 1000))

    # # #
    rho, w = openings_distr_components(mec)
    output.write('\n\nNumber (r) of openings / burst (unconditional)')
    output.write('\nP(r) =')
    pdfs.geometricPDF_printout(rho, w, output)
    mu = openings_mean(mec)
    output.write('\nMean from direct matrix calc = {0:.3f}'. format(mu))

    # # #
    output.write('\n\nPDF of first opening in a burst with 2 or more openings')
    output.write('\nf(open; r>1) =')
    eigs, w = first_opening_length_pdf_components(mec)
    pdfs.expPDF_printout(eigs, w, output)

    # # #
    output.write('\n\nPDF of total open time per bursts')
    output.write('\nf(open tot) =')
    eigs, w = open_time_total_pdf_components(mec)
    pdfs.expPDF_printout(eigs, w, output)
    mop = open_time_mean(mec)
    output.write('\nMean from direct matrix calc = {0:.3f} '.
        format(mop * 1000) + 'millisec')

    # # #
    output.write('\n\nPDF of total shut time per bursts')
    output.write('\nf(gap tot) =')
    eigs, w = shut_time_total_pdf_components(mec)
    pdfs.expPDF_printout(eigs, w, output)

    # # #
    output.write('\n\nPDF of gaps between bursts')
    output.write('\nf(gap) =')
    eigs, w = shut_times_between_burst_pdf_components(mec)
    pdfs.expPDF_printout(eigs, w, output)

    # # #
    bpop = mop / m
    output.write('\n\nPopen WITHIN BURST = (open time/bst)/(bst length)\
        = {0:.3f} \n'.format(bpop))
    output.write('*******************************************\n')

    
