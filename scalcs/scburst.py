import sys

import numpy as np
from numpy import linalg as nplin
from deprecated import deprecated

from scalcs import qmatlib as qml
from scalcs import pdfs

from scalcs.qmatlib import QMatrix

class SCBurst(QMatrix):
    """
    SCBurst calculates various burst-related quantities from a Q-matrix based on methods in Colquhoun & Hawkes (1982).
    """

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0):
        super().__init__(Q, kA=kA, kB=kB, kC=kC, kD=kD)
        #self.GAB = self._GXY(self.QAA, self.QAB)
        #self.GBA = self._GXY(self.QBB, self.QBA)
        #self.GABAG = np.dot(self.GAB, self.GBA)
        self.GABAG = self.GAB @ self.GBA
        self.probability_of_ending = self.IA - self.GABAG
        #self.GAF = self._GXY(self.QAA, self.QAF)
        #self.GFA = self._GXY(self.QFF, self.QFA)
        self.invQAA = nplin.inv(self.QAA)
        self.invQBB = nplin.inv(self.QBB)
        self.invQFF = nplin.inv(self.QFF)
        self.pinfC = self.pinf[self.kE:self.kG]
        self.pinfA = self.pinf[:self.kA]

        self.WBB = self.QBB + self.QBA @ self.GAB

    @property
    def start_burst(self):
        """
        Calculate the start probabilities of a burst (Eq. 3.2, CH82).
        phiB = (pCinf * (QCB * GBA + QCA)) / (pCinf * (QCB * GBA + QCA) * uA)

        Returns
        -------
        phiB : array_like, shape (1, kA)
        """

        nom = self.pinfC @ (self.QCB @ self.GBA + self.QCA)
        return nom / (nom @ self.uA)
    
    @property
    def end_burst(self):
        r"""
        Calculate the end vector for a burst (Eq. 3.4, CH82).

        .. math::

        \bs{e}_\text{b} = (\bs{I}-\bs{G}_\cl{AB} \bs{G}_\cl{BA}) \bs{u}_\cl{A}

        Returns
        -------
        eB : array_like, shape (kA, 1)
        """

        return self.probability_of_ending @ self.uA

    @property
    def mean_number_of_openings(self):
        """
        Calculate the mean number of openings per burst (Eq. 3.7, CH82).
        mu = phiB * (I - GAB * GBA)^(-1) * uA

        Returns
        -------
        mu : float
            The mean number ofopenings per burst.
        """

        #return np.dot(np.dot(self.start_burst, nplin.inv(self.probability_of_ending)), self.uA)[0]
        return (self.start_burst @ nplin.inv(self.probability_of_ending) @ self.uA)[0]


    def openings_distr_components(self):
        """
        Calculate coeficients for geometric ditribution P(r)- probability of
        seeing r openings (Eq. 3.9 CH82):
        P(r) = sum(W * rho^(r-1))
        where w
        wm = phiB * Am * endB (Eq. 3.10 CH82)
        and rho- eigenvalues of GAB * GBA.

        Returns
        -------
        rho : ndarray, shape (kA,)
        w : ndarray, shape (kA,)
        """

        rho, A = qml.eigenvalues_and_spectral_matrices(self.GABAG)
        w = np.dot(np.dot(self.start_burst, A), self.end_burst).transpose()[0]
        return rho, w

    def openings_distr(self, r):
        """
        The distribution of openings per burst (Eq. 3.5, CH82).
        P(r) = phiB * (GAB * GBA)^(r-1) * eB

        Parameters
        ----------
        r : int
            Number of openings per burst.

        Returns
        -------
        Pr : float
            Probability of seeing r openings per burst.
        """

        if r == 1:
            interm = self.IA
        else:
            interm = np.linalg.matrix_power(self.GABAG, r - 1)
        return self.start_burst @ interm @ self.end_burst

    def openings_cond_distr_depend_on_start_state(self, r):
        """
        The distribution of openings per burst coditional on starting state.

        Parameters
        ----------
        r : int
            Number of openings per burst.

        Returns
        -------
        vecPr : array_like, shape (kA, 1)
            Probability of seeing r openings per burst depending on starting state.
        """
        if r == 1:
            interm = self.IA
        else:
            interm = np.linalg.matrix_power(self.GABAG, r - 1)
        return (interm @ self.end_burst).transpose()

    @property
    def mean_length(self):
        """
        Calculate the mean burst length (Eq. 3.19, CH82).
        m = PhiB * (I - GAB * GBA)^(-1) * (-QAA^(-1)) * \
            (I - QAB * (QBB^(-1)) * GBA) * uA

        Returns
        -------
        m : float
            The mean burst length.
        """

        inv_ending = nplin.inv(self.probability_of_ending)
        interm2 = self.IA - self.QAB @ self.invQBB @self.GBA
        return (self.start_burst @ inv_ending @ -self.invQAA @ interm2 @ self.uA)[0]

    def length_pdf_components(self):
        """
        Calculate eigenvalues and amplitudes for an ideal (no missed events)
        exponential burst length probability density function.

        Returns
        -------
        eigs : ndarray, shape(k, 1)
            Eigenvalues.
        w : ndarray, shape(k, 1)
            Component amplitudes.
        """

        w = np.zeros(self.kE)
        eigs, A = qml.eigenvalues_and_spectral_matrices(-self.QEE)
        for i in range(self.kE):
            w[i] = (self.start_burst @ A[i][:self.kA, :self.kA] @ (-self.QAA) @ self.end_burst)[0]
        return eigs, w

    def length_pdf_no_single_openings_components(self):
        """
        Calculate eigenvalues and amplitudes for an ideal (no missed events)
        exponential burst length probability density function for bursts with
        two or more openings.

        Returns
        -------
        e : ndarray, shape(k, 1)
            Eigenvalues.
        w : ndarray, shape(k, 1)
            Component amplitudes.
        """

        eA, AA = qml.eigenvalues_and_spectral_matrices(-self.QAA)
        eE, AE = qml.eigenvalues_and_spectral_matrices(-self.QEE)
        e = np.concatenate((eE, eA))
        A = np.concatenate((AE[:, :self.kA, :self.kA], -AA), axis=0)
        w = np.array([(self.start_burst @ A[i] @ (-self.QAA) @ self.end_burst)[0] / self.probability_more_than_one_opening
            for i in range(self.kA + self.kE)])
       
        return e, w

    def length_pdf_direct(self, t):
        """
        Probability density function of the burst length (Eq. 3.17, CH82).
        f(t) = phiB * [PEE(t)]AA * (-QAA) * eB, where PEE(t) = exp(QEE * t)

        Parameters
        ----------
        t : float
            Burst length.

        Returns
        -------
        f : float
            PDF value for given burst length- t.
        """

        expQEEA = qml.expQt(self.QEE, t)[ : self.kA, : self.kA]
        return self.start_burst @ expQEEA @ -self.QAA @ self.end_burst

    def length_cond_pdf(self, t):
        """
        The distribution of burst length coditional on starting state.

        Parameters
        ----------
        t : float
            Burst length.

        Returns
        -------
        vec : array_like, shape (kA, 1)
            Probability of seeing burst length t depending on starting state.
        """

        expQEEA = qml.expQt(self.QEE, t)[ : self.kA, : self.kA]
        return (expQEEA @ -self.QAA @ self.end_burst).transpose()

    @property
    def mean_open_time(self):
        """
        Calculate the mean total open time per burst (Eq. 3.26, CH82).

        Returns
        -------
        m : float
            The mean total open time per burst.
        """

        VAA = self.QAA + self.QAB @ self.GBA
        return (self.start_burst @ -nplin.inv(VAA) @ self.uA)[0]
    
    def total_open_time_pdf_components(self):
        """
        Eq. 3.23, CH82

        Returns
        -------
        e : ndarray, shape(k, 1)
            Eigenvalues.
        w : ndarray, shape(k, 1)
            Component amplitudes.
        """

        VAA = self.QAA + self.QAB @ self.GBA
        e, A = qml.eigenvalues_and_spectral_matrices(-VAA)

        w = np.zeros(self.kA)
        for i in range(self.kA):
            w[i] = np.dot(np.dot(np.dot(self.start_burst, A[i]), (-VAA)), self.uA)[0]

        return e, w

    @property
    def probability_more_than_one_opening(self):
        """Calculate probability of a burst having more than one opening. 
           Probability of a burst having just one opening is 
           P(1) = start_burst * end_burst"""
        
        return 1 - (self.start_burst @ self.end_burst)[0]

    def first_opening_length_pdf_components(self):
        """
        Calculate time constants and amplitudes for an ideal (no missed events)
        pdf of first opening in a burst with 2 or more openings.

        Returns
        -------
        e : ndarray, shape(k, 1)
            Eigenvalues.
        w : ndarray, shape(k, 1)
            Component amplitudes.
        """

        e, A = qml.eigenvalues_and_spectral_matrices(-self.QAA)
        w = np.array([
            (self.start_burst @ A[i] @ (-self.QAA) @ self.GABAG @ self.uA)[0] / self.probability_more_than_one_opening
            for i in range(self.kA)])
        return e, w

    @property
    def mean_shut_time(self):
        """
        Calculate the mean total shut time per burst (Eq. 3.41, CH82) for all bursts (including one opening bursts).

        Returns
        -------
        m : float
            The mean total shut time per burst.
        """

        return (self.start_burst @ self.GAB @ -nplin.inv(self.WBB) @ self.GBA @ self.uA)[0]
    
    @property
    def mean_shut_times_between_bursts(self):
        """
        Calculate the mean length of the gap between bursts (Eq. 3.86, CH82).

        Returns
        -------
        m : float
            The mean shut time between bursts.
        """

        start = self.pinfA / (self.pinfA @ -self.QAA @ self.end_burst)
        m1 = self.QAF @ -self.invQFF @ self.GFA
        m2 = self.QAB @ -self.invQBB @ self.GBA
        return (start @ (m1 - m2) @ self.uA)[0]

    def shut_times_inside_burst_pdf_components(self):
        """
        Calculate time constants and amplitudes for a PDF of all gaps within
        bursts (Eq. 3.75, CH82).

        Returns
        -------
        e : ndarray, shape(k, 1)
            Eigenvalues.
        w : ndarray, shape(k, 1)
            Component amplitudes.
        """

        e, A = qml.eigenvalues_and_spectral_matrices(-self.QBB)
        w = -np.array([ (self.start_burst @ nplin.inv(self.probability_of_ending) @ self.GAB @ A[i] @ self.QBB @ self.GBA @ self.uA)[0] 
            / (self.mean_number_of_openings - 1) for i in range(self.kB) ])
        return e, w

    def shut_time_total_pdf_components_2more_openings(self):
        """
        Calculate time constants and amplitudes for a PDF of total shut time 
        per burst (Eq. 3.40, CH82) for bursts with at least 2 openings.

        Returns
        -------
        e : ndarray, shape(k, 1)
            Eigenvalues.
        w : ndarray, shape(k, 1)
            Component amplitudes.
        """

        e, A = qml.eigenvalues_and_spectral_matrices(-self.WBB)
        norm = 1 - (self.start_burst @ self.end_burst)[0]

        w = np.zeros(self.kB)
        for i in range(self.kB):
            w[i] = (self.start_burst @ self.GAB @ A[i] @ self.QBA @ self.end_burst)[0] / norm

        return e, w

    def shut_times_between_burst_pdf_components(self):
        """
        Calculate time constants and amplitudes for a PDF of gaps between bursts.

        Returns
        -------
        e : ndarray, shape(k, 1)
            Eigenvalues.
        w : ndarray, shape(k, 1)
            Component amplitudes.
        """

        eigsB, AmatB = qml.eigenvalues_and_spectral_matrices(-self.QBB)
        eigsF, AmatF = qml.eigenvalues_and_spectral_matrices(-self.QFF)
        start = self.pinfA / (self.pinfA @ -self.QAA @ self.end_burst)
        wB = -start @ self.QAB @ AmatB @ self.QBA @ self.uA
        wF = start @ self.QAF @ AmatF @ self.QFA @ self.uA
        return np.append(eigsB, eigsF), np.append(wB, wF)
    
    @property
    def burst_popen(self):
        """
        Calculate the burst open probability.
        """
        return self.mean_open_time / self.mean_length

    @property
    def total_popen(self):
        """
        Calculate the total open probability.
        """
        return self.mean_open_time / (self.mean_length + self.mean_shut_times_between_bursts)
   


#######################   DEPRECATED   ##############################
@deprecated("Use 'SCBurst.start_burst'")
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

@deprecated("Use 'SCBurst.end_burst'")
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

@deprecated("Use 'SCBurst.mean_number_of_opening'")
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

@deprecated("Use 'SCBurst.openings_distr'")
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

@deprecated("Use 'SCBurst.openings_cond_distr_depend_on_start_stat'")
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

@deprecated("Use 'SCBurst.mean_length'")
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

@deprecated("Use 'SCBurst.length_pdf_direct'")
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

@deprecated("Use 'SCBurst.length_cond_pdf'")
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

@deprecated("Use 'SCBurst.mean_open_time'")
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

@deprecated("Use 'SCBurst.first_opening_length_pdf_components'")
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
    eigs, A = qml.eigenvalues_and_spectral_matrices(-mec.QAA)
    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    GG = np.dot(GAB, GBA)
    norm = np.dot(np.dot(phiBurst(mec), GG), uA)[0]

    w = np.zeros(mec.kA)
    for i in range(mec.kA):
        w[i] = np.dot(np.dot(np.dot(np.dot(phiBurst(mec),
            A[i]), (-mec.QAA)), GG), uA) / norm

    return eigs, w

@deprecated("Use 'SCBurst.mean_shut_time'")
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

@deprecated("Use 'SCBurst.shut_time_total_pdf_components_2more_openings'")
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
    eigs, A = qml.eigenvalues_and_spectral_matrices(-WBB)
    norm = 1 - np.dot(phiBurst(mec), endBurst(mec))[0]

    w = np.zeros(mec.kB)
    for i in range(mec.kB):
        w[i] = np.dot(np.dot(np.dot(np.dot(phiBurst(mec), GAB),
            A[i]), (mec.QBA)), endBurst(mec)) / norm

    return eigs, w

@deprecated("Use 'SCBurst.mean_shut_time_between_bursts'")
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

@deprecated("Use 'SCBurst.length_pdf_components'")
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
    eigs, A = qml.eigenvalues_and_spectral_matrices(-mec.QEE)
    for i in range(mec.kE):
        w[i] = np.dot(np.dot(np.dot(phiBurst(mec),
            A[i][:mec.kA, :mec.kA]), (-mec.QAA)), endBurst(mec))[0]
        # Due to deprecation warning needs [0] at the end
        #DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
        
    return eigs, w

@deprecated("Use 'SCBurst.length_pdf_components'")
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

    eigsA, AmatA = qml.eigenvalues_and_spectral_matrices(-mec.QAA)
    eigsE, AmatE = qml.eigenvalues_and_spectral_matrices(-mec.QEE)
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

@deprecated("Use 'SCBurst.total_open_time_pdf_components'")
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
    eigs, A = qml.eigenvalues_and_spectral_matrices(-VAA)
    uA = np.ones((mec.kA, 1))

    w = np.zeros(mec.kA)
    for i in range(mec.kA):
        w[i] = np.dot(np.dot(np.dot(phiBurst(mec), A[i]), (-VAA)), uA)[0]  # [0] at the end needed due to NumPy 1.25 deprecation

    return eigs, w

@deprecated("Use 'SCBurst.shut_times_inside_burst_pdf_components'")
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
    eigs, A = qml.eigenvalues_and_spectral_matrices(-mec.QBB)
    GAB, GBA = qml.iGs(mec.Q, mec.kA, mec.kB)
    interm = nplin.inv(np.eye(mec.kA) - np.dot(GAB, GBA))
    norm = openings_mean(mec) - 1

    w = np.zeros(mec.kB)
    for i in range(mec.kB):
        w[i] = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(phiBurst(mec), interm),
            GAB), A[i]), (-mec.QBB)), GBA), uA) / norm

    return eigs, w

@deprecated("Use 'SCBurst.sgut_time_between_bursts_pdf_components'")
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
    eigsB, AmatB = qml.eigenvalues_and_spectral_matrices(-mec.QBB)
    eigsF, AmatF = qml.eigenvalues_and_spectral_matrices(-mec.QFF)
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

@deprecated("Use 'SCBurst.openings_distribution_components'")
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
    rho, A = qml.eigenvalues_and_spectral_matrices(GG)
    w = np.dot(np.dot(phiBurst(mec), A), endBurst(mec)).transpose()[0]
    return rho, w

@deprecated("Use 'qmatprint.SCBurstPrints'")
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

    str = ('\n*******************************************\n' +
        'CALCULATED SINGLE CHANNEL BURST PDFS ETC....\n')

    # # #
    phiB = phiBurst(mec)
    str += ('Initial vector for burst (phiB) = \n')
    str1 = ''
    for i in range(mec.kA):
        str1 += '{0:.5g}\t'.format(phiB[i])
    str += str1 + '\n'
    endB = endBurst(mec)
    str += 'End vector for burst (endB) = \n'
    str1 = ''
    for i in range(mec.kA):
        str1 += '{0:.5g}\t'.format(endB[i, 0])
    str += str1 + '\n'

    # # #
    eigs, w = length_pdf_components(mec)
    str += ('\nTotal burst length, unconditional pdf\n')
    str += ('Fbst(t) =\n')
    str += pdfs.expPDF_printout(eigs, w)
    mbl = length_mean(mec)
    str += ('Mean from direct matrix calc = {0:.5g} millisec\n'.
        format(mbl * 1000))
        
    # # #
    eigs, w = length_no_single_openings_pdf_components(mec)
    str += ('\nBurst length pdf for bursts with 2 or more openings.\n')
    str += ('Fbst(bst>1) =\n')
    str += pdfs.expPDF_printout(eigs, w)

    # # #
    rho, w = openings_distr_components(mec)
    str += ('\nNumber (r) of openings / burst (unconditional)\n')
    str += ('P(r) =\n')
    str += pdfs.geometricPDF_printout(rho, w)
    mu = openings_mean(mec)
    str += ('Mean from direct matrix calc = {0:.5g}\n'. format(mu))

    # # #
    str += ('\nPDF of first opening in a burst with 2 or more openings\n')
    str += ('f(open; r>1) =\n')
    eigs, w = first_opening_length_pdf_components(mec)
    str += pdfs.expPDF_printout(eigs, w)

    # # #
    str += ('\nPDF of total open time per bursts\n')
    str += ('f(open tot) =\n')
    eigs, w = open_time_total_pdf_components(mec)
    str += pdfs.expPDF_printout(eigs, w)
    mop = open_time_mean(mec)
    str += ('Mean from direct matrix calc = {0:.5g} '.
        format(mop * 1000) + 'millisec\n')

    # # #
    str += ('\nPDF of total shut time per bursts for bursts with at least 2 openings\n')
    str += ('f(gap tot) =\n')
    eigs, w = shut_time_total_pdf_components_2more_openings(mec)
    str += pdfs.expPDF_printout(eigs, w)
    msh = shut_time_total_mean(mec)
    str += ('Mean of total shut time for all bursts = {0:.5g} '.
        format(msh * 1000) + 'millisec\n')

    str += ('\nNo of gaps within burst per unit open time = {0:.5g} \n'.
        format((mu - 1) / mop))

    # # #
    str += ('\nPDF of gaps inside bursts\n')
    str += ('f(gap) =\n')
    eigs, w = shut_times_inside_burst_pdf_components(mec)
    str += pdfs.expPDF_printout(eigs, w)

    # # #
    str += ('\nPDF of gaps between bursts\n')
    str += ('f(gap) =\n')
    eigs, w = shut_times_between_burst_pdf_components(mec)
    str += pdfs.expPDF_printout(eigs, w)
    msh = shut_times_between_burst_mean(mec)
    str += ('Mean from direct matrix calc = {0:.5g} '.
        format(msh * 1000) + 'millisec\n')

    # # #
    bpop = mop / mbl
    str += ('\nPopen WITHIN BURST = (open time/bst)/(bst length)\
        = {0:.5g} \n'.format(bpop))
    tpop = mop / (mbl + msh)
    str += ('Total Popen = (open time/bst)/(bst_length + ' +
        'mean gap between burst) = {0:.5g} \n'.format(tpop))

    return str
