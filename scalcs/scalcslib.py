import sys
from math import*
from decimal import*
from deprecated import deprecated

import scipy.optimize as so
import numpy as np
from numpy import linalg as nplin

from scalcs import qmatlib as qml
from scalcs import pdfs
from scalcs.qmatlib import HJCMatrix, AsymptoticPDFCalculator


class AsymptoticPDF(AsymptoticPDFCalculator):
    '''
    Class to calculate dwell-time distributions (open and shut times) using
    HJC models from the Q matrix.
    '''

    def __init__(self, mec, tres=0.0): #Q, kA=1, kB=1, kC=0, kD=0, tres=0.0):
        super().__init__(mec.Q, kA=mec.kA, kB=mec.kB, kC=mec.kC, kD=mec.kD, tres=tres)
        
    @property
    def apparentPopen(self):
        """Apparent open probability (Eq. from the mean open and shut times)."""
        return self.apparent_mean_open_time / (self.apparent_mean_open_time + self.apparent_mean_shut_time)

    @property
    def apparent_mean_open_time(self):
        """Calculate apparent mean open time using HJC probability density function."""
        QexpQF = np.dot(self.QAF, self.expQFF)
        return self.tres + np.dot(self.HJCphiA, np.dot(np.dot(self.dARSdS, QexpQF), self.uF))[0]

    @property
    def apparent_mean_shut_time(self):
        """Calculate apparent mean shut time using HJC probability density function."""
        QexpQA = np.dot(self.QFA, self.expQAA)
        return self.tres + np.dot(self.HJCphiF, np.dot(np.dot(self.dFRSdS, QexpQA), self.uA))[0]

    def HJC_asymptotic_open_time_pdf_components(self):
        """Get the roots and areas for open times."""
        return self._asymptotic_pdf_components(open=True)

    def HJC_asymptotic_shut_time_pdf_components(self):
        """Get the roots and areas for shut times."""
        return self._asymptotic_pdf_components(open=False)

    def _asymptotic_pdf_components(self, open):
        """Helper to calculate roots and areas for open or shut times."""
        roots = self.asymptotic_roots(open=open)
        areas = self.asymptotic_areas(roots, open=open)
        return roots, areas


class ExactPDFCalculator(HJCMatrix):
    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0, tres=0.0):
        """
        Initialize the ExactPDFCalculator.

        Parameters
        ----------
        Q : ndarray
            The Q matrix representing the transition rates.
        kA, kB, kC, kD : int, optional
            Dimensions of different state subspaces. Defaults are 1 for kA and kB, 0 for kC and kD.
        """
        super().__init__(Q, kA=kA, kB=kB, kC=kC, kD=kD, tres=tres)
   
    def exact_GAMAxx(self, open=True):
        """
        Calculate gama coeficients for the exact dwell time pdf (Eq. 3.22, HJC90).

        Parameters
        ----------
        open : bool, optional
            True for open time pdf and False for shut time pdf.

        Returns
        -------
        eigen : ndarray, shape (k,)
            Eigenvalues of -Q matrix.
        gama00, gama10, gama11 : ndarrays
            Constants for the exact dwell time pdf.
        """

        u = self.uF if open else self.uA
        phi = self.HJCphiA if open else self.HJCphiF
        eigs, Z00, Z10, Z11 = self.Zxx(open=open)
        gama00 = (phi @ Z00 @ u).T[0]
        gama10 = (phi @ Z10 @ u).T[0]
        gama11 = (phi @ Z11 @ u).T[0]
        return eigs, gama00, gama10, gama11

    def Zxx(self, open=True):
        """
        Calculate Z constants for the exact open time PDF (Eq. 3.22, HJC90).
        This function handles both open and shut times by toggling `open`.

        Parameters
        ----------
        open : bool, optional
            True for open time PDF, False for shut time PDF (default is True).

        Returns
        -------
        eigs : ndarray, shape (k,)
            Eigenvalues of the -Q matrix.
        Z00, Z10, Z11 : ndarrays, shape (k, kA, kF)
            Z constants for the exact open/shut time PDF.
        """

        eigs, A = qml.eigenvalues_and_spectral_matrices(-self.Q)
        kx = self.kA if open else self.kF
        expQyy = self.expQFF if open else self.expQAA
        Qyx = self.QFA if open else self.QAF
        Qxy = self.QAF if open else self.QFA

        # Compute  Dj (Eq. 3.16, HJC90) and Cimr (Eq. 3.18, HJC90).
        D = np.empty((self.k))
        if open:
            C00 = A[ : ,    : self.kA,    : self.kA]
            A1  = A[ : ,    : self.kA, self.kA :   ]
        else:
            C00 = A[ : , self.kA :   , self.kA :   ]
            A1  = A[ : , self.kA :   ,    : self.kA]
        D = A1 @ expQyy @ Qyx

        #C11 = np.empty((self.k, kx, kx))
        #for i in range(self.k):
        #    C11[i] = D[i] @ C00[i]
        # Vectorized computation of C11 (Eq. 3.18, HJC90)
        C11 = np.einsum('ijk,ikl->ijl', D, C00)

        C10 = np.empty((self.k, kx, kx))
        for i in range(self.k):
            S = sum(
                ((D[i] @ C00[j]) + (D[j] @ C00[i])) / (eigs[j] - eigs[i])
                for j in range(self.k) if j != i
            )
            C10[i] = S

        # Matrix M and Zxx calculation
        M = np.dot(Qxy, expQyy)
        Z00 = np.einsum('ijk,kl->ijl', C00, M)
        Z10 = np.einsum('ijk,kl->ijl', C10, M)
        Z11 = np.einsum('ijk,kl->ijl', C11, M)
# Old code; keep for refernce
#        Z00 = np.array([np.dot(C, M) for C in C00])
#        Z10 = np.array([np.dot(C, M) for C in C10])
#        Z11 = np.array([np.dot(C, M) for C in C11])

        return eigs, Z00, Z10, Z11


############################   ADJACENT PDF   #######################################

class AdjacentPDF(HJCMatrix):
    """ Calculates adjecent pdf components. """

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0, tres=0.0):
        super().__init__(Q, kA=kA, kB=kB, kC=kC, kD=kD, tres=tres)
        self.uA = np.ones((self.kA))[:,np.newaxis]
        self.uF = np.ones((self.kF))[:,np.newaxis]
        self.phiAr = self.phiA.reshape(1, self.kA)
        self.invQAA, self.invQFF = -nplin.inv(self.QAA), nplin.inv(self.QFF)

    def adjacent_open_to_shut_range_mean(self, u1, u2):
        """
        Calculate mean (ideal- no missed events) open times adjacent to a specified shut time range.

        Parameters
        ----------
        u1, u2 : floats
            Shut time range.
        """
        
        expQFFr = qml.expQ(self.QFF, u2) - qml.expQ(self.QFF, u1)
        col = self.QAF @ self.invQFF @ expQFFr @ self.QFA @ self.uA
        return (self.phiAr @ qml.Qpow(self.invQAA, 2) @ col)[0, 0] / (self.phiAr @ self.invQAA @ col)[0, 0]

    def adjacent_open_to_shut_range_pdf_components(self, u1, u2): #, QAA, QAF, QFF, QFA, phiA):
        """
        Calculate time constants and areas for an ideal (no missed events)
        exponential probability density function of open times adjacent to a 
        specified shut time range.

        Parameters
        ----------
        u1, u2 : floats
            Shut time range.

        Returns
        -------
        eigs : ndarray, shape(k, 1)
            Eigenvalues.
        areas : ndarray, shape(k, 1)
            Component relative areas.
        """

        expQFFr = qml.expQ(self.QFF, u2) - qml.expQ(self.QFF, u1)
        col = self.QAF @ self.invQFF @ expQFFr @ self.QFA @ self.uA
        w = np.zeros(self.kA)
        eigs, A = qml.eigenvalues_and_spectral_matrices(-self.QAA)
        den = (self.phiAr @ self.invQAA @ col)[0, 0]
        #TODO: remove 'for'
        #w = np.array([(self.phiA @ A[i] @ col / den) for i in range(self.kA)])
        for i in range(self.kA):
            w[i] = (self.phiA @ A[i] @ col) / den
        return eigs, w

    #TODO: recview next two functions
    def HJC_adjacent_mean_open_to_shut_time_pdf(self, shut_times):
        """
        Calculate theoretical HJC (with missed events correction) mean open time
        given previous/next gap length (continuous function; CHS96 Eq.3.5). 

        Parameters
        ----------
        shut_times : array of floats
            Shut time interval.

        Returns
        -------
        mp : ndarray of floats
            Mean open time given previous gap length.
        mn : ndarray of floats
            Mean open time given next gap length.
        """
        
        DARS = self.dARSdS()
        eigs, A = qml.eigenvalues_and_spectral_matrices(-self.Q)
        FZ00, FZ10, FZ11 = self.Zxx(open=False)
        Froots = asymptotic_roots(self.tres, open=False)
        FR = self.R(Froots, open=False)
        Q1 = DARS @ self.QAF @ self.expQFF
        col1 = Q1 @ self.uF
        row1 = self.HJCphiA @ Q1
        
        mp, mn = [], []
        for t in shut_times:
            eGFAt = qml.eGAF(t, self.tres, eigs, FZ00, FZ10, FZ11, Froots, FR, self.QFA, self.expQAA)
            denom = (self.HJCphiF @ eGFAt @ self.uA)[0]
            mp.append((self.HJCphiF @ eGFAt @ col1)[0] / denom)
            mn.append((row1 @ eGFAt @ self.uA)[0] / denom)
        return np.array(mp), np.array(mn)

    def HJC_dependency(self, top, tsh):
        """
        Calculate normalised joint distribution (CHS96, Eq. 3.22) of an open time
        and the following shut time as proposed by Magleby & Song 1992. 
        
        Parameters
        ----------
        top, tsh : array_like of floats
            Open and shut tims.

        Returns
        -------
        dependency : ndarray
        """
        
        eigs, A = qml.eigenvalues_and_spectral_matrices(-self.Q)
        FZ00, FZ10, FZ11 = self.Zxx(open=False)
        Froots = self.asymptotic_roots(open=False)
        FR = self.R(Froots, open=False) 
        AZ00, AZ10, AZ11 = self.Zxx(open=True)
        Aroots = self.asymptotic_roots(open=True)
        AR = self.R(Aroots, open=True)

        dependency = np.zeros((top.shape[0], tsh.shape[0]))
        for i in range(top.shape[0]):
            eGAFt = qml.eGAF(top[i], self.tres, eigs, AZ00, AZ10, AZ11, Aroots, AR, self.QAF, self.expQFF)
            fo = (self.HJCphiA @ eGAFt @ self.uF)[0]
            for j in range(tsh.shape[0]):
                eGFAt = qml.eGAF(tsh[j], self.tres, eigs, FZ00, FZ10, FZ11, Froots, FR, self.QFA, self.expQAA)
                fs = (self.HJCphiF @ eGFAt @ self.uA)[0]
                fos = (self.HJCphiA @ eGAFt @ eGFAt @ self.uA)[0]
                dependency[i, j] = (fos - (fo * fs)) / (fo * fs)
        return dependency
    

############################   FUNCTIONS TO REVIEW   ########################################





def asymptotic_pdf(t, tres, tau, area):
    """
    Calculate asymptotic probabolity density function.

    Parameters
    ----------
    t : ndarray.
        Time.
    tres : float
        Time resolution.
    tau : ndarray, shape(k, 1)
        Time constants.
    area : ndarray, shape(k, 1)
        Component relative area.

    Returns
    -------
    apdf : ndarray.
    """
    t1 = np.extract(t[:] < tres, t)
    t2 = np.extract(t[:] >= tres, t)
    apdf2 = t2 * pdfs.ExpPDF(tau, area).calculate(t2 - tres)   #pdfs.expPDF(t2 - tres, tau, area)
    apdf = np.append(t1 * 0.0, apdf2)

    return apdf

def exact_pdf(t, tres, roots, areas, eigvals, gamma00, gamma10, gamma11):
    r"""
    Calculate exponential probabolity density function with exact solution for
    missed events correction (Eq. 21, HJC92).

    .. math::
       :nowrap:

       \begin{align*}
       f(t) =
       \begin{cases}
       f_0(t)                          & \text{for}\; 0 \leq t \leq t_\text{res} \\
       f_0(t) - f_1(t - t_\text{res})  & \text{for}\; t_\text{res} \leq t \leq 2 t_\text{res}
       \end{cases}
       \end{align*}

    Parameters
    ----------
    t : float
        Time.
    tres : float
        Time resolution (dead time).
    roots : array_like, shape (k,)
    areas : array_like, shape (k,)
    eigvals : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    gama00, gama10, gama11 : lists of floats
        Coeficients for the exact open/shut time pdf.

    Returns
    -------
    f : float
    """

    if t < tres:
        f = 0
    elif ((tres < t) and (t < (2 * tres))):
        f = qml.f0((t - tres), eigvals, gamma00)
    elif ((tres * 2) < t) and (t < (3 * tres)):
        f = (qml.f0((t - tres), eigvals, gamma00) -
            qml.f1((t - 2 * tres), eigvals, gamma10, gamma11))
    else:
        f = pdfs.ExpPDF(-1 / roots, areas).calculate(t-tres) #pdfs.expPDF(t - tres, -1 / roots, areas)
    return f

def likelihood(theta, opts):
    """
    Calculate likelihood for a series of open and shut times using ideal
    probability density functions.
    """

    mec = opts['mec']
    conc = opts['conc']
    bursts = opts['data']

    #mec.set_rateconstants(np.exp(theta))
    mec.theta_unsqueeze(np.exp(theta))
    mec.set_eff('c', conc)

    startB = qml.phiA(mec)
    endB = np.ones((mec.kF, 1))

    loglik = 0
    for ind in bursts:
        burst = bursts[ind]
        grouplik = startB
        for i in range(len(burst)):
            t = burst[i]
            if i % 2 == 0: # open time
                GAFt = qml.iGt(t, mec.QAA, mec.QAF)
            else: # shut
                GAFt = qml.iGt(t, mec.QFF, mec.QFA)
            grouplik = np.dot(grouplik, GAFt)
            if grouplik.max() > 1e50:
                grouplik = grouplik * 1e-100
                print ('grouplik was scaled down')
        grouplik = np.dot(grouplik, endB)
        loglik += log(grouplik[0])

    newrates = np.log(mec.theta())
    return -loglik, newrates

def HJClik(theta, opts):
    """
    Calculate likelihood for a series of open and shut times using HJC missed
    events probability density functions (first two dead time intervals- exact
    solution, then- asymptotic).

    Lik = phi * eGAF(t1) * eGFA(t2) * eGAF(t3) * ... * eGAF(tn) * uF
    where t1, t3,..., tn are open times; t2, t4,..., t(n-1) are shut times.

    Gaps > tcrit are treated as unusable (e.g. contain double or bad bit of
    record, or desens gaps that are not in the model, or gaps so long that
    next opening may not be from the same channel). However this calculation
    DOES assume that all the shut times predicted by the model are present
    within each group. The series of multiplied likelihoods is terminated at
    the end of the opening before an unusable gap. A new series is then
    started, using appropriate initial vector to give Lik(2), ... At end
    these are multiplied to give final likelihood.

    Parameters
    ----------
    theta : array_like
        Guesses.
    bursts : dictionary
        A dictionary containing lists of open and shut intervals.
    opts : dictionary
        opts['mec'] : instance of type Mechanism
        opts['tres'] : float
            Time resolution (dead time).
        opts['tcrit'] : float
            Ctritical time interval.
        opts['isCHS'] : bool
            True if CHS vectors should be used (Eq. 5.7, CHS96).

    Returns
    -------
    loglik : float
        Log-likelihood.
    newrates : array_like
        Updated rates/guesses.
    """
    # TODO: Errors.

    mec = opts['mec']
    conc = opts['conc']
    tres = opts['tres']
    tcrit = opts['tcrit']
    is_chsvec = opts['isCHS']
    bursts = opts['data']

    mec.theta_unsqueeze(np.exp(theta))
    mec.set_eff('c', conc)

    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    expQFF = qml.expQ(mec.QFF, tres)
    expQAA = qml.expQ(mec.QAA, tres)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)
    phiF = qml.phiHJC(eGFA, eGAF, mec.kF)
    startB = qml.phiHJC(eGAF, eGFA, mec.kA)
    endB = np.ones((mec.kF, 1))

    eigen, A = qml.eigenvalues_and_spectral_matrices(-mec.Q)
    AZ00, AZ10, AZ11 = qml.Zxx(mec.Q, eigen, A, mec.kA, mec.QFF,
        mec.QAF, mec.QFA, expQFF, True)
    Aroots = asymptotic_roots(tres,
        mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)
    AR = qml.AR(Aroots, tres, mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)
    FZ00, FZ10, FZ11 = qml.Zxx(mec.Q, eigen, A, mec.kA, mec.QAA,
        mec.QFA, mec.QAF, expQAA, False)
    Froots = asymptotic_roots(tres,
        mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA)
    FR = qml.AR(Froots, tres, mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA)

    if is_chsvec:
        startB, endB = qml.CHSvec(Froots, tres, tcrit,
            mec.QFA, mec.kA, expQAA, phiF, FR)

    loglik = 0
    for ind in range(len(bursts)):
        burst = bursts[ind]
        grouplik = startB
        for i in range(len(burst)):
            t = burst[i]
            if i % 2 == 0: # open time
                eGAFt = qml.eGAF(t, tres, eigen, AZ00, AZ10, AZ11, Aroots,
                AR, mec.QAF, expQFF)
            else: # shut
                eGAFt = qml.eGAF(t, tres, eigen, FZ00, FZ10, FZ11, Froots,
                FR, mec.QFA, expQAA)
            grouplik = np.dot(grouplik, eGAFt)
            if grouplik.max() > 1e50:
                grouplik = grouplik * 1e-100
                #print 'grouplik was scaled down'
        grouplik = np.dot(grouplik, endB)
        try:
            loglik += log(grouplik[0])
        except:
            print ('HJClik: Warning: likelihood has been set to 0')
            print ('likelihood=', grouplik[0])
            print ('rates=', mec.unit_rates())
            loglik = 0
            break

    newrates = np.log(mec.theta())
    return -loglik, newrates

#####################   DEPRECATED FUNCTIONS   #################################


@deprecated("Use '...'")
def ideal_dwell_time_pdf(t, QAA, phiA):
    """
    Probability density function of the open time.
    f(t) = phiOp * exp(-QAA * t) * (-QAA) * uA
    For shut time pdf A by F in function call.

    Parameters
    ----------
    t : float
        Time (sec).
    QAA : array_like, shape (kA, kA)
        Submatrix of Q.
    phiA : array_like, shape (1, kA)
        Initial vector for openings

    Returns
    -------
    f : float
    """

    kA = QAA.shape[0]
    uA = np.ones((kA, 1))
    expQAA = qml.expQ(QAA, t)
    f = np.dot(np.dot(np.dot(phiA, expQAA), -QAA), uA)
    return f

@deprecated("Use '...'")
def ideal_subset_time_pdf(Q, k1, k2, t):
    """
    
    """
    
    u = np.ones((k2 - k1 + 1, 1))
    phi, QSub = qml.phiSub(Q, k1, k2)
    expQSub = qml.expQ(QSub, t)
    f = np.dot(np.dot(np.dot(phi, expQSub), -QSub), u)
    return f

@deprecated("Use '...'")
def adjacent_open_to_shut_range_mean(u1, u2, QAA, QAF, QFF, QFA, phiA):
    """
    Calculate mean (ideal- no missed events) open times adjacent to a 
    specified shut time range.

    Parameters
    ----------
    u1, u2 : floats
        Shut time range.
    QAA, QAF, QFF, QFA : array_like
        Submatrices of Q.
    phiA : array_like, shape (1, kA)
        Initial vector for openings

    Returns
    -------
    m : float
        Mean open time.
    """
    
    kA = QAA.shape[0]
    uA = np.ones((kA))[:,np.newaxis]
    invQAA, invQFF = -nplin.inv(QAA), nplin.inv(QFF)
    expQFFr = qml.expQ(QFF, u2) - qml.expQ(QFF, u1)
    col = np.dot(np.dot(np.dot(np.dot(QAF, invQFF), expQFFr), QFA), uA)
    row1 = np.dot(phiA, qml.Qpow(invQAA, 2))
    row2 = np.dot(phiA, invQAA)
    m = np.dot(row1, col)[0, 0] / np.dot(row2, col)[0, 0]
    return m

@deprecated("Use '...'")
def HJC_dependency(top, tsh, tres, Q, QAA, QAF, QFF, QFA):
    """
    Calculate normalised joint distribution (CHS96, Eq. 3.22) of an open time
    and the following shut time as proposed by Magleby & Song 1992. 
    
    Parameters
    ----------
    top, tsh : array_like of floats
        Open and shut tims.
    tres : float
        Time resolution.
    Q : array, shape (k,k)
        Q matrix. 
    QAA, QAF, QFF, QFA : array_like
        Submatrices of Q.

    Returns
    -------
    dependency : ndarray
    """
    
    kA, kF = QAA.shape[0], QFF.shape[0]
    uA = np.ones((kA))[:,np.newaxis]
    uF = np.ones((kF))[:,np.newaxis]
    expQFF = qml.expQ(QFF, tres)
    expQAA = qml.expQ(QAA, tres)
    GAF, GFA = qml.iGs(Q, kA, kF)
    eGAF = qml.eGs(GAF, GFA, kA, kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, kF, kA, expQAA)
    phiA = qml.phiHJC(eGAF, eGFA, kA)
    phiF = qml.phiHJC(eGFA, eGAF, kF)
    eigs, A = qml.eigenvalues_and_spectral_matrices(-Q)
    FZ00, FZ10, FZ11 = qml.Zxx(Q, eigs, A, kA, QAA, QFA, QAF, expQAA, False)
    Froots = asymptotic_roots(tres, QFF, QAA, QFA, QAF, kF, kA)
    FR = qml.AR(Froots, tres, QFF, QAA, QFA, QAF, kF, kA)
    AZ00, AZ10, AZ11 = qml.Zxx(Q, eigs, A, kA, QFF, QAF, QFA, expQFF, True)
    Aroots = asymptotic_roots(tres, QAA, QFF, QAF, QFA, kA, kF)
    AR = qml.AR(Aroots, tres, QAA, QFF, QAF, QFA, kA, kF)

    dependency = np.zeros((top.shape[0], tsh.shape[0]))
    
    for i in range(top.shape[0]):
        eGAFt = qml.eGAF(top[i], tres, eigs, AZ00, AZ10, AZ11, Aroots,
                AR, QAF, expQFF)
        fo = np.dot(np.dot(phiA, eGAFt), uF)[0]
        
        for j in range(tsh.shape[0]):
            eGFAt = qml.eGAF(tsh[j], tres, eigs, FZ00, FZ10, FZ11, Froots,
                FR, QFA, expQAA)
            fs = np.dot(np.dot(phiF, eGFAt), uA)[0]
            fos = np.dot(np.dot(np.dot(phiA, eGAFt), eGFAt), uA)[0]
            dependency[i, j] = (fos - (fo * fs)) / (fo * fs)
    return dependency

@deprecated("Use '...'")
def HJC_adjacent_mean_open_to_shut_time_pdf(sht, tres, Q, QAA, QAF, QFF, QFA):
    """
    Calculate theoretical HJC (with missed events correction) mean open time
    given previous/next gap length (continuous function; CHS96 Eq.3.5). 

    Parameters
    ----------
    sht : array of floats
        Shut time interval.
    tres : float
        Time resolution.
    Q : array, shape (k,k)
        Q matrix.
    QAA, QAF, QFF, QFA : array_like
        Submatrices of Q.

    Returns
    -------
    mp : ndarray of floats
        Mean open time given previous gap length.
    mn : ndarray of floats
        Mean open time given next gap length.
    """
    
    kA, kF = QAA.shape[0], QFF.shape[0]
    uA = np.ones((kA))[:,np.newaxis]
    uF = np.ones((kF))[:,np.newaxis]
    expQFF = qml.expQ(QFF, tres)
    expQAA = qml.expQ(QAA, tres)
    GAF, GFA = qml.iGs(Q, kA, kF)
    eGAF = qml.eGs(GAF, GFA, kA, kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, kF, kA, expQAA)
    phiA = qml.phiHJC(eGAF, eGFA, kA)
    phiF = qml.phiHJC(eGFA, eGAF, kF)
    DARS = qml.dARSdS(tres, QAA, QFF, GAF, GFA, expQFF, kA, kF)
    eigs, A = qml.eigenvalues_and_spectral_matrices(-Q)
    FZ00, FZ10, FZ11 = qml.Zxx(Q, eigs, A, kA, QAA, QFA, QAF, expQAA, False)
    Froots = asymptotic_roots(tres, QFF, QAA, QFA, QAF, kF, kA)
    FR = qml.AR(Froots, tres, QFF, QAA, QFA, QAF, kF, kA)
    Q1 = np.dot(np.dot(DARS, QAF), expQFF)
    col1 = np.dot(Q1, uF)
    row1 = np.dot(phiA, Q1)
    
    mp = []
    mn = []
    for t in sht:
        eGFAt = qml.eGAF(t, tres, eigs, FZ00, FZ10, FZ11, Froots,
                    FR, QFA, expQAA)
        denom = np.dot(np.dot(phiF, eGFAt), uA)[0]
        nom1 = np.dot(np.dot(phiF, eGFAt), col1)[0]
        nom2 = np.dot(np.dot(row1, eGFAt), uA)[0]
        mp.append(nom1 / denom)
        mn.append(nom2 / denom)
    
    return np.array(mp), np.array(mn)

@deprecated("Use '...'")
def adjacent_open_to_shut_range_pdf_components(u1, u2, QAA, QAF, QFF, QFA, phiA):
    """
    Calculate time constants and areas for an ideal (no missed events)
    exponential probability density function of open times adjacent to a 
    specified shut time range.

    Parameters
    ----------
    t : float
        Time (sec).
    QAA : array_like, shape (kA, kA)
        Submatrix of Q.
    phiA : array_like, shape (1, kA)
        Initial vector for openings

    Returns
    -------
    taus : ndarray, shape(k, 1)
        Time constants.
    areas : ndarray, shape(k, 1)
        Component relative areas.
    """

    kA = QAA.shape[0]
    uA = np.ones((kA))[:,np.newaxis]
    invQAA, invQFF = -nplin.inv(QAA), nplin.inv(QFF)
    expQFFr = qml.expQ(QFF, u2) - qml.expQ(QFF, u1)
    col = np.dot(np.dot(np.dot(np.dot(QAF, invQFF), expQFFr), QFA), uA)
    w = np.zeros(kA)
    eigs, A = qml.eigenvalues_and_spectral_matrices(-QAA)
    row = np.dot(phiA, invQAA)
    den = np.dot(row, col)[0, 0]
    #TODO: remove 'for'
    for i in range(kA):
        w[i] = np.dot(np.dot(phiA, A[i]), col) / den
    return eigs, w


@deprecated("Use '...'")
def ideal_popen(mec):
    """
    Calculate ideal equilibrium open probability, Popen.

    Parameters
    ----------
    mec : instance of type Mechanism

    Returns
    -------
    popen : float
        Open probability. 
    """
    p = qml.pinf(mec.QGG)
    return np.sum(p[ : mec.kA]) / np.sum(p)

@deprecated("Use '...'")
def ideal_subset_mean_life_time(Q, state1, state2):
    """
    Calculate mean life time in a specified subset. Add all rates out of subset
    to get total rate out. Skip rates within subset.

    Parameters
    ----------
    mec : instance of type Mechanism
    state1,state2 : int
        State numbers (counting origin 1)

    Returns
    -------
    mean : float
        Mean life time.
    """

    k = Q.shape[0]
    p = qml.pinf(Q)
    # Total occupancy for subset.
    pstot = np.sum(p[state1-1 : state2])

    # Total rate out
    if pstot == 0:
        mean = 0.0
    else:
        s = 0.0
        for i in range(state1-1, state2):
            for j in range(k):
                if (j < state1-1) or (j > state2 - 1):
                    s += Q[i, j] * p[i] / pstot

        mean = 1 / s
    return mean

@deprecated("Use '...'")
def ideal_mean_latency_given_start_state(mec, state):
    """
    Calculate mean latency to next opening (shutting), given starting in
    specified shut (open) state.

    mean latency given starting state = pF(0) * inv(-QFF) * uF

    F- all shut states (change to A for mean latency to next shutting
    calculation), p(0) = [0 0 0 ..1.. 0] - a row vector with 1 for state in
    question and 0 for all other states.

    Parameters
    ----------
    mec : instance of type Mechanism
    state : int
        State number (counting origin 1)

    Returns
    -------
    mean : float
        Mean latency.
    """

    if state <= mec.kA:
        # for calculating mean latency to next shutting
        p = np.zeros((mec.kA))
        p[state-1] = 1
        u = np.ones((mec.kA, 1))
        invQ = nplin.inv(-mec.QAA)
    else:
        # for calculating mean latency to next opening
        p = np.zeros((mec.kI))
        p[state-mec.kA-1] = 1
        u = np.ones((mec.kI, 1))
        invQ = nplin.inv(-mec.QII)

    mean = np.dot(np.dot(p, invQ), u)[0]
    return mean

@deprecated("Use '...'")
def exact_mean_time(tres, QAA, QFF, QAF, kA, kF, GAF, GFA):
    """
    Calculate exact mean open or shut time from HJC probability density
    function.

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    QAA : array_like, shape (kA, kA)
    QFF : array_like, shape (kF, kF)
    QAF : array_like, shape (kA, kF)
        QAA, QFF, QAF - submatrices of Q.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.
    GAF : array_like, shape (kA, kB)
    GFA : array_like, shape (kB, kA)
        GAF, GFA- transition probabilities

    Returns
    -------
    mean : float
        Apparent mean open/shut time.
    """
    
    expQFF = qml.expQ(QFF, tres)
    expQAA = qml.expQ(QAA, tres)
    eGAF = qml.eGs(GAF, GFA, kA, kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, kF, kA, expQAA)

    phiA = qml.phiHJC(eGAF, eGFA, kA)
    QexpQF = np.dot(QAF, expQFF)
    DARS = qml.dARSdS(tres, QAA, QFF,
        GAF, GFA, expQFF, kA, kF)
    uF = np.ones((kF, 1))
    # meanOpenTime = tres + phiA * DARS * QexpQF * uF
    mean = tres + np.dot(phiA, np.dot(np.dot(DARS, QexpQF), uF))[0]

    return mean

@deprecated("Use '...'")
def exact_mean_open_shut_time(mec, tres):
    """
    Calculate exact mean open or shut time from HJC probability density
    function.

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    QAA : array_like, shape (kA, kA)
    QFF : array_like, shape (kF, kF)
    QAF : array_like, shape (kA, kF)
        QAA, QFF, QAF - submatrices of Q.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.
    GAF : array_like, shape (kA, kB)
    GFA : array_like, shape (kB, kA)
        GAF, GFA- transition probabilities

    Returns
    -------
    mean : float
        Apparent mean open/shut time.
    """
    GAF = qml.GXY(mec.QAA, mec.QAF) 
    GFA = qml.GXY(mec.QFF, mec.QFA)
    #GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    expQFF = qml.expQ(mec.QFF, tres)
    expQAA = qml.expQ(mec.QAA, tres)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)

    phiA = qml.phiHJC(eGAF, eGFA, mec.kA)
    phiF = qml.phiHJC(eGFA, eGAF, mec.kF)
    QexpQF = np.dot(mec.QAF, expQFF)
    QexpQA = np.dot(mec.QFA, expQAA)
    DARS = qml.dARSdS(tres, mec.QAA, mec.QFF, GAF, GFA, expQFF, mec.kA, mec.kF)
    DFRS = qml.dARSdS(tres, mec.QFF, mec.QAA, GFA, GAF, expQAA, mec.kF, mec.kA)
    uF, uA = np.ones((mec.kF, 1)), np.ones((mec.kA, 1))
    # meanOpenTime = tres + phiA * DARS * QexpQF * uF
    meanA = tres + np.dot(phiA, np.dot(np.dot(DARS, QexpQF), uF))[0]
    meanF = tres + np.dot(phiF, np.dot(np.dot(DFRS, QexpQA), uA))[0]

    return meanA, meanF

@deprecated("Use '...'")
def asymptotic_areas(tres, roots, QAA, QFF, QAF, QFA, kA, kF, GAF, GFA):
    """
    Find the areas of the asymptotic pdf (Eq. 58, HJC92).

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    roots : array_like, shape (1,kA)
        Roots of the asymptotic pdf.
    QAA : array_like, shape (kA, kA)
    QFF : array_like, shape (kF, kF)
    QAF : array_like, shape (kA, kF)
    QFA : array_like, shape (kF, kA)
        QAA, QFF, QAF, QFA - submatrices of Q.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.
    GAF : array_like, shape (kA, kB)
    GFA : array_like, shape (kB, kA)
        GAF, GFA- transition probabilities

    Returns
    -------
    areas : ndarray, shape (1, kA)
    """

    expQFF = qml.expQ(QFF, tres)
    expQAA = qml.expQ(QAA, tres)
    eGAF = qml.eGs(GAF, GFA, kA, kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, kF, kA, expQAA)
    phiA = qml.phiHJC(eGAF, eGFA, kA)
    R = qml.AR(roots, tres, QAA, QFF, QAF, QFA, kA, kF)
    uF = np.ones((kF,1))
    areas = np.zeros(kA)
    for i in range(kA):
        areas[i] = ((-1 / roots[i]) *
            np.dot(phiA, np.dot(np.dot(R[i], np.dot(QAF, expQFF)), uF)))[0]  # [0] at the end needed due to NumPy 1.25 deprecation

#    rowA = np.zeros((kA,kA))
#    colA = np.zeros((kA,kA))
#    for i in range(kA):
#        WA = qml.W(roots[i], tres,
#            QAA, QFF, QAF, QFA, kA, kF)
#        rowA[i] = qml.pinf(WA)
#        AW = np.transpose(WA)
#        colA[i] = qml.pinf(AW)
#
#    for i in range(kA):
#        uF = np.ones((kF,1))
#        nom = np.dot(np.dot(np.dot(np.dot(np.dot(phiA, colA[i]), rowA[i]),
#            QAF), expQFF), uF)
#        W1A = qml.dW(roots[i], tres, QAF, QFF, QFA, kA, kF)
#        denom = -roots[i] * np.dot(np.dot(rowA[i], W1A), colA[i])
#        areas[i] = nom / denom

    return areas

@deprecated("Use '...'")
def asymptotic_roots(tres, QAA, QFF, QAF, QFA, kA, kF):
    """
    Find roots for the asymptotic probability density function (Eqs. 52-58,
    HJC92).

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    QAA : array_like, shape (kA, kA)
    QFF : array_like, shape (kF, kF)
    QAF : array_like, shape (kA, kF)
    QFA : array_like, shape (kF, kA)
        QAA, QFF, QAF, QFA - submatrices of Q.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.

    Returns
    -------
    roots : array_like, shape (1, kA)
    """

    sas = -1000000
    sbs = -0.0000001
    sro = bisect_intervals(sas, sbs, tres,
        QAA, QFF, QAF, QFA, kA, kF)

    roots = np.zeros(kA)
    for i in range(kA):
        roots[i] = so.brentq(qml.detW, sro[i, 0], sro[i, 1],
            args=(tres, QAA, QFF, QAF, QFA, kA, kF))

#        roots[i] = so.bisect(qml.detW, sro[i,0], sro[i,1],
#            args=(tres, QAA, QFF, QAF, QFA, kA, kF))

    return roots

@deprecated("Use '...'")
def bisect_gFB(s, tres, Q11, Q22, Q12, Q21, k1, k2):
    """
    Find number of eigenvalues of H(s) that are equal to or less than s.

    Parameters
    ----------
    s : float
        Laplace transform argument.
    tres : float
        Time resolution (dead time).
    Q11 : array_like, shape (k1, k1)
    Q22 : array_like, shape (k2, k2)
    Q21 : array_like, shape (k2, k1)
    Q12 : array_like, shape (k1, k2)
        Q11, Q12, Q22, Q21 - submatrices of Q.
    k1 : int
        A number of open/shut states in kinetic scheme.
    k2 : int
        A number of shut/open states in kinetic scheme.

    Returns
    -------
    ng : int
    """

    h = qml.H(s, tres, Q11, Q22, Q12, Q21, k2)
    eigval = nplin.eigvals(h)
    ng = (eigval <= s).sum()
    return ng

@deprecated("Use '...'")
def bisect_intervals(sa, sb, tres, Q11, Q22, Q12, Q21, k1, k2):
    """
    Find, according to Frank Ball's method, suitable starting guesses for
    each HJC root- the upper and lower limits for bisection. Exactly one root
    should be between those limits.

    Parameters
    ----------
    sa, sb : float
        Laplace transform arguments.
    tres : float
        Time resolution (dead time).
    Q11 : array_like, shape (k1, k1)
    Q22 : array_like, shape (k2, k2)
    Q21 : array_like, shape (k2, k1)
    Q12 : array_like, shape (k1, k2)
        Q11, Q12, Q22, Q21 - submatrices of Q.
    k1, k2 : int
        Numbers of open/shut states in kinetic scheme.

    Returns
    -------
    sr : array_like, shape (k2, 2)
        Limits of s value intervals containing exactly one root.
    """

    nga = bisect_gFB(sa, tres, Q11, Q22, Q12, Q21, k1, k2)
    if nga > 0: sa = sa * 4
    ngb = bisect_gFB(sb, tres, Q11, Q22, Q12, Q21, k1, k2)
    if ngb < k2: sb = sb / 4

    done = []
    todo = [[sa, sb, nga, ngb]]
#    nsplit = 0

#    while (len(done) < k1) and (nsplit < 1000):
    while todo:
        svv = todo.pop()
        sa1, sc, sb2, nga1, ngc, ngb2 = bisect_split(svv[0], svv[1], svv[2], svv[3],
            tres, Q11, Q22, Q12, Q21, k1, k2)
#        nsplit += 1

        # Check if either or both of the two subintervals output from
        # SPLIT contain only one root?
        if (ngc - nga1) == 1:
            done.append([sa1, sc])
#            if len(done) == k1:
#                break
        else:
            todo.append([sa1, sc, nga1, ngc])
        if (ngb2 - ngc) == 1:
            done.append([sc, sb2])
        else:
            todo.append([sc, sb2, ngc, ngb2])

    if len(done) < k1:
        sys.stderr.write(
            "bisectHJC: Warning: Only {0:d} roots out of {1:d} were located.".
            format(len(done), k1))
    return np.array(done)

@deprecated("Use '...'")
def bisect_split(sa, sb, nga, ngb, tres, Q11, Q22, Q12, Q21, k1, k2):
    """
    Split interval [sa, sb] into two subintervals, each of which contains
    at least one root.

    Parameters
    ----------
    sa, sb : float
        Limits of Laplace transform argument interval.
    nga, ngb : int
        Number of eigenvalues (roots) below sa or sb, respectively.
    tres : float
        Time resolution (dead time).
    Q11 : array_like, shape (k1, k1)
    Q22 : array_like, shape (k2, k2)
    Q21 : array_like, shape (k2, k1)
    Q12 : array_like, shape (k1, k2)
        Q11, Q12, Q22, Q21 - submatrices of Q.
    k1, k2 : int
        Numbers of open/shut states in kinetic scheme.

    Returns
    -------
    sa, sc, sb : floats
        Limits of s value intervals.
    nga, ngc, ngb : ints
        Number of eigenvalues below corresponding s values.
    """

    ntrymax = 1000
    ntry = 0
    #nerrs = False
    end = False

    while (not end) and (ntry < ntrymax):
        sc = (sa + sb) / 2.0
        ngc = bisect_gFB(sc, tres, Q11, Q22, Q12, Q21, k1, k2)
        if ngc == nga: sa = sc
        elif ngc == ngb: sb = sc
        else:
            end = True
        ntry += 1
    if not end:
        sys.stderr.write(
        "bisectHJC: Warning: unable to split intervals for bisection.")

    return sa, sc, sb, nga, ngc, ngb

@deprecated("Use '...'")
def exact_GAMAxx(mec, tres, open):
    """
    Calculate gama coeficients for the exact open time pdf (Eq. 3.22, HJC90).

    Parameters
    ----------
    tres : float
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    open : bool
        True for open time pdf and False for shut time pdf.

    Returns
    -------
    eigen : ndarray, shape (k,)
        Eigenvalues of -Q matrix.
    gama00, gama10, gama11 : ndarrays
        Constants for the exact open/shut time pdf.
    """

    expQFF = qml.expQ(mec.QII, tres)
    expQAA = qml.expQ(mec.QAA, tres)
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kI)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kI, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kI, mec.kA, expQAA)
    #TODO: replace 'eigs_sorted' by 'eigenvalues_and_spectral_matrices'
    eigs, A = qml.eigenvalues_and_spectral_matrices(-mec.Q)

    if open:
        phi = qml.phiHJC(eGAF, eGFA, mec.kA)
        Z00, Z10, Z11 = qml.Zxx(mec.Q, eigs, A, mec.kA,
            mec.QII, mec.QAI, mec.QIA, expQFF, open)
        u = np.ones((mec.kI,1))
    else:
        phi = qml.phiHJC(eGFA, eGAF, mec.kI)
        Z00, Z10, Z11 = qml.Zxx(mec.Q, eigs, A, mec.kA,
            mec.QAA, mec.QIA, mec.QAI, expQAA, open)
        u = np.ones((mec.kA, 1))

    gama00 = (np.dot(np.dot(phi, Z00), u)).T[0]
    gama10 = (np.dot(np.dot(phi, Z10), u)).T[0]
    gama11 = (np.dot(np.dot(phi, Z11), u)).T[0]

    return eigs, gama00, gama10, gama11

@deprecated("Use 'scalcs.popen'")
def exact_popen(mec, tres):
    """
    Calculate equilibrium open probability, Popen, corrected for missed events.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution.

    Returns
    -------
    popen : float
        Open probability. 
    """
    hmopen, hmshut = exact_mean_open_shut_time(mec, tres)
    return (hmopen / (hmopen + hmshut))

@deprecated("Use '...'")
def ideal_dwell_time_pdf_components(QAA, phiA):
    """
    Calculate time constants and areas for an ideal (no missed events)
    exponential open time probability density function.
    For shut time pdf A by F in function call.

    Parameters
    ----------
    t : float
        Time (sec).
    QAA : array_like, shape (kA, kA)
        Submatrix of Q.
    phiA : array_like, shape (1, kA)
        Initial vector for openings

    Returns
    -------
    taus : ndarray, shape(k, 1)
        Time constants.
    areas : ndarray, shape(k, 1)
        Component relative areas.
    """

    kA = QAA.shape[0]
    w = np.zeros(kA)
    #TODO: change 'eigs_sorted' into 'eigenvalues_and_spectral_matrices'
    eigs, A = qml.eigenvalues_and_spectral_matrices(-QAA)
    uA = np.ones((kA, 1))
    #TODO: remove 'for'
    for i in range(kA):
        w[i] = np.dot(np.dot(np.dot(phiA, A[i]), (-QAA)), uA)[0]  # [0] at the end needed due to NumPy 1.25 deprecation

    return eigs, w
