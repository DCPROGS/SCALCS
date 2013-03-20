"""A collection of functions for single channel burst calculations.
"""

__author__="R.Lape, University College London"
__date__ ="$07-Dec-2010 20:29:14$"

import sys

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
    pC = qml.pinf(mec.Q)[mec.kE:mec.kG]
    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    nom = np.dot(pC, (np.dot(mec.QCB, GBA) + mec.QCA))
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
    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    eB = np.dot((I - np.dot(GAB, GBA)), uA)
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
    Calculate time constants and amplitudes for an ideal (no missed events)
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
    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    interm1 = nplin.inv(I - np.dot(GAB, GBA))
    interm2 = I - np.dot(np.dot(mec.QAB, invQBB), GBA)
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

def length_no_single_openings_pdf_components(mec):
    """
    Calculate time constants and amplitudes for an ideal (no missed events)
    exponential burst length probability density function for bursts with
    two or more openings.

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

    eigsA, AmatA = qml.eigs(-mec.QAA)
    eigsE, AmatE = qml.eigs(-mec.QEE)
    eigs = np.append(eigsE, eigsA)
    A = np.append(AmatE[:,:mec.kA, :mec.kA], -AmatA, axis=0)
    w = np.zeros(mec.kA + mec.kE)

    endB = endBurst(mec)
    start = phiBurst(mec)
    norm = 1 - np.dot(start, endB)[0]

    for i in range(mec.kA + mec.kE):
        w[i] = np.dot(np.dot(np.dot(start,
            A[i]), (-mec.QAA)), endB) / norm
     
    return eigs, w

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

    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    GG = np.dot(GAB, GBA)
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

    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    GG = np.dot(GAB, GBA)
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
    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    interm = nplin.inv(I - np.dot(GAB, GBA))
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

    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    GG = np.dot(GAB, GBA)
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

    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    VAA = mec.QAA + np.dot(mec.QAB, GBA)
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
    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    VAA = mec.QAA + np.dot(mec.QAB, GBA)
    m = np.dot(np.dot(phiBurst(mec), -nplin.inv(VAA)), uA)[0]
    return m

def shut_times_inside_burst_pdf_components(mec):
    """
    Calculate time constants and amplitudes for a PDF of all gaps within
    bursts (Eq. 3.75, CH82).

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
    eigs, A = qml.eigs(-mec.QBB)
    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    interm = nplin.inv(np.eye(mec.kA) - np.dot(GAB, GBA))
    norm = openings_mean(mec) - 1

    w = np.zeros(mec.kB)
    for i in range(mec.kB):
        w[i] = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(phiBurst(mec), interm),
            GAB), A[i]), (-mec.QBB)), GBA), uA) / norm

    return eigs, w

def shut_times_between_burst_pdf_components(mec):
    """
    Calculate time constants and amplitudes for a PDF of gaps between bursts.

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
    eigsB, AmatB = qml.eigs(-mec.QBB)
    eigsF, AmatF = qml.eigs(-mec.QFF)
    pA = qml.pinf(mec.Q)[:mec.kA]
    end = np.dot(-mec.QAA, endBurst(mec))
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

def shut_times_between_burst_mean(mec):
    """
    Calculate the mean length of the gap between bursts (Eq. 3.86, CH82).

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    m : float
        The mean shut time between bursts.
    """

    pA = qml.pinf(mec.Q)[:mec.kA]
    end = np.dot(-mec.QAA, endBurst(mec))
    start = pA / np.dot(pA, end)
    uA = np.ones((mec.kA, 1))

    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    invQFF = -nplin.inv(mec.QFF)
    invQBB = -nplin.inv(mec.QBB)
    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)

    m1 = np.dot(np.dot(mec.QAF, invQFF), GFA)
    m2 = np.dot(np.dot(mec.QAB, invQBB), GBA)
    m = np.dot(np.dot(start, m1 - m2), uA)[0]

    return m

def shut_time_total_pdf_components_2more_openings(mec):
    """
    Calculate time constants and amplitudes for a PDF of total shut time 
    per burst (Eq. 3.40, CH82) for bursts with at least 2 openings.

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

    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    WBB = mec.QBB + np.dot(mec.QBA, GAB)
    eigs, A = qml.eigs(-WBB)
    norm = 1 - np.dot(phiBurst(mec), endBurst(mec))[0]

    w = np.zeros(mec.kB)
    for i in range(mec.kB):
        w[i] = np.dot(np.dot(np.dot(np.dot(phiBurst(mec), GAB),
            A[i]), (mec.QBA)), endBurst(mec)) / norm

    return eigs, w

def shut_time_total_mean(mec):
    """
    Calculate the mean total shut time per burst (Eq. 3.41, CH82).

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    m : float
        The mean total shut time per burst.
    """

    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    WBB = mec.QBB + np.dot(mec.QBA, GAB)
    invW = - nplin.inv(WBB)
    uA = np.ones((mec.kA, 1))
    m = np.dot(np.dot(np.dot(np.dot(phiBurst(mec), GAB),
            invW), (GBA)), uA)[0]
    return m

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
    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    GG = np.dot(GAB, GBA)
    norm = np.dot(np.dot(phiBurst(mec), GG), uA)[0]

    w = np.zeros(mec.kA)
    for i in range(mec.kA):
        w[i] = np.dot(np.dot(np.dot(np.dot(phiBurst(mec),
            A[i]), (-mec.QAA)), GG), uA) / norm

    return eigs, w

def printout_pdfs(mec, output=sys.stdout):
    """
    Output burst calculations into selected device (sys.stdout, printer, file,
    text field.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    output : output device
        Default device: sys.stdout
    """

    output.write('\n*******************************************\n')
    output.write('CALCULATED SINGLE CHANNEL BURST PDFS ETC....\n')

    # # #
    phiB = phiBurst(mec)
    output.write('Initial vector for burst (phiB) = \n')
    str = ''
    for i in range(mec.kA):
        str += '{0:.5g}\t'.format(phiB[i])
    output.write(str + '\n')
    endB = endBurst(mec)
    output.write('End vector for burst (endB) = \n')
    str = ''
    for i in range(mec.kA):
        str += '{0:.5g}\t'.format(endB[i, 0])
    output.write(str + '\n')

    # # #
    eigs, w = length_pdf_components(mec)
    output.write('\nTotal burst length, unconditional pdf\n')
    output.write('Fbst(t) =\n')
    pdfs.expPDF_printout(eigs, w, output)
    mbl = length_mean(mec)
    output.write('Mean from direct matrix calc = {0:.5g} millisec\n'.
        format(mbl * 1000))
        
    # # #
    eigs, w = length_no_single_openings_pdf_components(mec)
    output.write('\nBurst length pdf for bursts with 2 or more openings.\n')
    output.write('Fbst(bst>1) =\n')
    pdfs.expPDF_printout(eigs, w, output)

    # # #
    rho, w = openings_distr_components(mec)
    output.write('\nNumber (r) of openings / burst (unconditional)\n')
    output.write('P(r) =\n')
    pdfs.geometricPDF_printout(rho, w, output)
    mu = openings_mean(mec)
    output.write('Mean from direct matrix calc = {0:.5g}\n'. format(mu))

    # # #
    output.write('\nPDF of first opening in a burst with 2 or more openings\n')
    output.write('f(open; r>1) =\n')
    eigs, w = first_opening_length_pdf_components(mec)
    pdfs.expPDF_printout(eigs, w, output)

    # # #
    output.write('\nPDF of total open time per bursts\n')
    output.write('f(open tot) =\n')
    eigs, w = open_time_total_pdf_components(mec)
    pdfs.expPDF_printout(eigs, w, output)
    mop = open_time_mean(mec)
    output.write('Mean from direct matrix calc = {0:.5g} '.
        format(mop * 1000) + 'millisec\n')

    # # #
    output.write('\nPDF of total shut time per bursts for bursts with at least 2 openings\n')
    output.write('f(gap tot) =\n')
    eigs, w = shut_time_total_pdf_components_2more_openings(mec)
    pdfs.expPDF_printout(eigs, w, output)
    msh = shut_time_total_mean(mec)
    output.write('Mean of total shut time for all bursts = {0:.5g} '.
        format(msh * 1000) + 'millisec\n')

    output.write('\nNo of gaps within burst per unit open time = {0:.5g} \n'.
        format((mu - 1) / mop))

    # # #
    output.write('\nPDF of gaps inside bursts\n')
    output.write('f(gap) =\n')
    eigs, w = shut_times_inside_burst_pdf_components(mec)
    pdfs.expPDF_printout(eigs, w, output)

    # # #
    output.write('\nPDF of gaps between bursts\n')
    output.write('f(gap) =\n')
    eigs, w = shut_times_between_burst_pdf_components(mec)
    pdfs.expPDF_printout(eigs, w, output)
    msh = shut_times_between_burst_mean(mec)
    output.write('Mean from direct matrix calc = {0:.5g} '.
        format(msh * 1000) + 'millisec\n')

    # # #
    bpop = mop / mbl
    output.write('\nPopen WITHIN BURST = (open time/bst)/(bst length)\
        = {0:.5g} \n'.format(bpop))
    tpop = mop / (mbl + msh)
    output.write('Total Popen = (open time/bst)/(bst_length + ' +
        'mean gap between burst) = {0:.5g} \n'.format(tpop))
