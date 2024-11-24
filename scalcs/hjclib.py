import sys
import numpy as np
import numpy.linalg as nplin
import scipy.optimize as so
from deprecated import deprecated

from scalcs import qmatlib as qml


class HJCMatrix(qml.QMatrix):
    """ Class to store HJC basic calculations. """
    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0, tres=0.0):
        super().__init__(Q, kA=kA, kB=kB, kC=kC, kD=kD)
        self.tres = tres

    @property
    def tres(self):
        return self._tres

    @tres.setter
    def tres(self, value):
        self._tres = value
        self._update_expQ()
        self._update_eG()

    def _update_expQ(self):
        self.expQFF = qml.expQ(self.QFF, self.tres)
        self.expQAA = qml.expQ(self.QAA, self.tres)

    def _update_eG(self):
        self.eGAF = self.eGs(self.GAF, self.GFA, self.kA, self.kF, self.expQFF)
        self.eGFA = self.eGs(self.GFA, self.GAF, self.kF, self.kA, self.expQAA)

    def eGs(self, GAF, GFA, kA, kF, expQFF):
        """
        Calculate eGAF, probabilities from transitions from apparently open to
        shut states regardles of when the transition occurs. Thease are Laplace
        transform of eGAF(t) when s=0. Used to calculat initial HJC vectors (HJC92).
        eGAF*(s=0) = (I - GAF * (I - expQFF) * GFA)^-1 * GAF * expQFF
        To caculate eGFA exhange A by F and F by A in function call.

        Parameters
        ----------
        GAF : array_like, shape (kA, kF)
        GFA : array_like, shape (kF, kA)
        kA : int
            A number of open states in kinetic scheme.
        kF : int
            A number of shut states in kinetic scheme.
        
        Returns
        -------
        eGAF : array_like, shape (kA, kF)
        """

        temp = np.eye(kA) - GAF @ (np.eye(kF) - expQFF) @ GFA
        return nplin.inv(temp) @ GAF @ expQFF

    @property
    def HJCphiA(self):
        """Calculate initial HJC vector for open states 
        phi*(I-eGAF*eGFA)=0 (Eq. 10, HJC92)"""
        return self._compute_HJCphi(self.eGAF, self.eGFA, self.kA, self.IA, self.uA)

    @property
    def HJCphiF(self):
        """Calculate initial HJC vector for shut states (Eq. 10, HJC92)."""
        return self._compute_HJCphi(self.eGFA, self.eGAF, self.kF, self.IF, self.uF)

    def _compute_HJCphi(self, eG_self, eG_other, k, I_self, u_self):
        """Helper to compute HJCphi vector for open/shut states."""
        if k == 1:
            return np.array([1])
        S = np.concatenate(((I_self - np.dot(eG_self, eG_other)), u_self), axis=1)
        return np.dot(u_self.T, nplin.inv(np.dot(S, S.T)))[0]
    
    def W(self, s, open=True):
        """
        Compute the W(s) matrix (Eq. 52, HJC92).
        """
        return s * (self.IA if open else self.IF) - self.H(s, open=open)

    def H(self, s, open=True):
        """
        Compute the H(s) matrix (Eq. 54, HJC92).
        """
        Q_self1 = self.QAA if open else self.QFF
        Q_other1 = self.QFF if open else self.QAA
        Q_self2 = self.QAF if open else self.QFA
        Q_other2 = self.QFA if open else self.QAF
        I_other = self.IF if open else self.IA
        invX = nplin.inv(s * I_other - Q_other1)
        expX = qml.expQ(-(s * I_other - Q_other1), self.tres)

        return Q_self1 + np.dot(np.dot(np.dot(Q_self2, invX), I_other - expX), Q_other2)

    def dW(self, s, open=True):
        """
        Compute the derivative of W(s) with respect to s (Eq. 56, HJC92).
        """
        Q_self1 = self.QFF if open else self.QAA
        Q_self2 = self.QAF if open else self.QFA
        Q_other2 = self.QFA if open else self.QAF
        I_self = self.IF if open else self.IA
        I_other = self.IA if open else self.IF

        X = s * I_self - Q_self1
        invX = nplin.inv(X)
        expX = qml.expQ(-X, self.tres)
        S = I_self - expX
        w1 = np.dot(S, invX) - self.tres * expX
        return I_other + np.dot(np.dot(Q_self2, w1), np.dot(invX, Q_other2))

    def detW(self, s, open=True):
        """
        Calculate determinant of WAA(s).

        Parameters
        ----------
        s : float
            Laplace transform argument.
        """
        return nplin.det(self.W(s, open))


class AsymptoticPDFCalculator(HJCMatrix):
    """Asymptotic PDF calculations"""

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0, tres=0.0):
        super().__init__(Q, kA=kA, kB=kB, kC=kC, kD=kD, tres=tres)
        self.derivative_calculator = DerivativeCalculator(Q, kA, kB, kC, kD, tres)

    def asymptotic_roots(self, open=True):
        """Find the roots for the asymptotic pdf (Eqs. 52-58, HJC92)."""
        return -self._calculate_roots(open=open)

    def _calculate_roots(self, open):
        sas, sbs = -1e6, -1e-7
        intervals = self._bisect_intervals(sas, sbs, open=open)
        root_count = self.kA if open else self.kF
        roots = np.array([so.brentq(self.detW, intervals[i, 0], intervals[i, 1], args=(open))
                          for i in range(root_count)])
        return roots
    
    def _bisect_intervals(self, sa, sb, open=True):
        """
        Use Frank Ball's method to find initial guesses for each HJC root.

        Parameters
        ----------
        sa, sb : float
            Laplace transform arguments to define the initial search interval.
        open : bool, optional
            Flag indicating whether to compute for open (True) or shut states (False).

        Returns
        -------
        array_like : shape (kA or kF, 2)
            Starting interval limits for bisection, containing exactly one root each.
        """
        root_count = self.kA if open else self.kF
        sa, sb = self.__adjust_intervals(sa, sb, root_count, open)
        
        intervals = []
        todo = [[sa, sb, self.__bisect_gFB(sa, open), self.__bisect_gFB(sb, open)]]
        
        while todo:
            sa1, sc, sb2, nga1, ngc, ngb2 = self.__bisect_split(todo.pop(), open)
            
            # Left interval: [sa1, sc]
            if (ngc - nga1) == 1:
                intervals.append([sa1, sc])
            else:
                todo.append([sa1, sc, nga1, ngc])
            
            # Right interval: [sc, sb2]
            if (ngb2 - ngc) == 1:
                intervals.append([sc, sb2])
            else:
                todo.append([sc, sb2, ngc, ngb2])
        
        # Check if all roots were located
        if len(intervals) < root_count:
            sys.stderr.write(f"bisectHJC: Warning: Only {len(intervals)} roots out of {root_count} were located.\n")
        
        return np.array(intervals)

    def __bisect_gFB(self, s, open=True):
        """
        Determine the number of eigenvalues of H(s) that are less than or equal to s.

        Parameters
        ----------
        s : float
            Laplace transform argument to evaluate.
        open : bool, optional
            Flag indicating whether to compute for open (True) or shut states (False).

        Returns
        -------
        int
            Number of eigenvalues less than or equal to the given s.
        """
        eigvals = nplin.eigvals(self.H(s, open=open))
        return (eigvals <= s).sum()

    def __adjust_intervals(self, sa, sb, root_count, open):
        """
        Adjust the initial search intervals for bisection based on the eigenvalue count.

        Parameters
        ----------
        sa, sb : float
            Initial interval limits to search for roots.
        root_count : int
            Number of roots expected to be found in the interval.
        open : bool, optional
            Flag indicating whether to compute for open (True) or shut states (False).

        Returns
        -------
        tuple : Adjusted interval limits (sa, sb).
        """
        nga = self.__bisect_gFB(sa, open=open)
        if nga > 0:
            sa *= 4
        
        ngb = self.__bisect_gFB(sb, open=open)
        if ngb < root_count:
            sb /= 4
        
        return sa, sb

    def __bisect_split(self, interval_data, open):
        """
        Split the interval [sa, sb] into two subintervals based on eigenvalue counts.

        Parameters
        ----------
        interval_data : list
            Contains the current interval limits and eigenvalue counts [sa, sb, nga, ngb].
        open : bool
            Flag indicating whether to compute for open (True) or shut states (False).

        Returns
        -------
        tuple : Split interval limits and corresponding eigenvalue counts 
                (sa1, sc, sb2, nga1, ngc, ngb2).
        """
        sa, sb, nga, ngb = interval_data
        max_attempts = 1000
        attempts = 0

        while attempts < max_attempts:
            sc = (sa + sb) / 2.0
            ngc = self.__bisect_gFB(sc, open=open)
            
            if ngc == nga:
                sa = sc
            elif ngc == ngb:
                sb = sc
            else:
                return sa, sc, sb, nga, ngc, ngb
            
            attempts += 1
        
        sys.stderr.write("bisectHJC: Warning: Unable to split intervals for bisection after 1000 attempts.\n")
        return sa, sc, sb, nga, ngc, ngb

    def asymptotic_areas(self, roots, open=True):
        """
        Calculate the areas of the asymptotic probability density function (Eq. 58, HJC92).

        Parameters
        ----------
        roots : array_like, shape (kA or kF)
            Roots of the asymptotic pdf.
        open : bool, optional
            Flag indicating the model type (open or closed), by default True.

        Returns
        -------
        areas : ndarray, shape (kA or kF)
            Areas corresponding to the asymptotic pdf roots.
        """

        R = self.R(-roots, open=open)
        k = self.kA if open else self.kF
        Q_other = self.QAF if open else self.QFA
        expQ_other = self.expQFF if open else self.expQAA
        phi = self.HJCphiA if open else self.HJCphiF
        u = self.uF if open else self.uA

        return np.array([(1 / roots[i]) * np.dot(phi, np.dot(R[i], np.dot(Q_other, expQ_other)).dot(u))[0] for i in range(k)])

    def R(self, roots, open=True): 
        """
        Compute the R matrix for the asymptotic areas.

        Parameters
        ----------
        roots : array_like, shape (kA or kF)
            Roots of the asymptotic pdf.
        open : bool, optional
            Flag indicating the model type (open or closed), by default True.

        Returns
        -------
        R : ndarray, shape (kA or kF, kA or kF, kA or kF)
            R matrix for asymptotic areas calculation.
        """
        k = self.kA if open else self.kF
        R = np.zeros((k, k, k))
        row = np.zeros((k, k))
        col = np.zeros((k, k))

        for i in range(k):
            W_matrix = self.W(roots[i], open=open)
            row[i] = qml.pinf(W_matrix)
            col[i] = qml.pinf(np.transpose(W_matrix))
        col = col.transpose()
        
        for i in range(k):
            nom = col[:,i].reshape((k, 1)) @ row[i,:].reshape((1, k))
            W1_matrix = self.dW(roots[i], open=open)
            denom = row[i,:].reshape((1, k)) @ W1_matrix @ col[:,i].reshape((k, 1))
            R[i] = nom / denom
        
        return R

    @property
    def dARSdS(self):
        """Public interface to compute dARSdS."""
        return self.derivative_calculator.dARSdS

    @property
    def dFRSdS(self):
        """Public interface to compute dFRSdS."""
        return self.derivative_calculator.dFRSdS


class DerivativeCalculator(HJCMatrix):
    """Handles derivative evaluations of the Laplace transform."""

    @property
    def dARSdS(self):
        r"""
        Evaluate the derivative with respect to s of the Laplace transform of the
        survival function (Eq. 3.6, CHS96) for open states:

        .. math::

        \left[ -\frac{\text{d}}{\text{d}s} {^\cl{A}\!\bs{R}^*(s)} \right]_{s=0}

        For same evaluation for shut states exhange A by F and F by A in function call.

        SFF = I - exp(QFF * tres)
        First evaluate [dVA(s) / ds] * s = 0.
        dVAds = -inv(QAA) * GAF * SFF * GFA - GAF * SFF * inv(QFF) * GFA +
        + tres * GAF * expQFF * GFA

        Then: DARS = inv(VA) * QAA^(-2) - inv(VA) * dVAds * inv(VA) * inv(QAA) =
        = inv(VA) * [inv(QAA) - dVAds * inv(VA)] * inv(QAA)
        where VA = I - GAF * SFF * GFA

        Returns
        -------
        DARS : array_like, shape (kA, kA)
        """

        invQAA = nplin.inv(self.QAA)
        invQFF = nplin.inv(self.QFF)

        SFF = self.IF - self.expQFF
        Q1 = self.tres * self.GAF @ self.expQFF @ self.GFA
        Q2 = self.GAF @ SFF @ invQFF @ self.GFA
        Q3 = -invQAA @ self.GAF @ SFF @ self.GFA

        VA = self.IA - self.GAF @ SFF @ self.GFA
        Q4 = invQAA - (Q1 - Q2 + Q3) @ nplin.inv(VA)
        return nplin.inv(VA) @ Q4 @ invQAA

    @property
    def dFRSdS(self):
        r"""
        Evaluate the derivative with respect to s of the Laplace transform of the
        survival function (Eq. 3.6, CHS96) for open states:

        .. math::

        \left[ -\frac{\text{d}}{\text{d}s} {^\cl{A}\!\bs{R}^*(s)} \right]_{s=0}

        For same evaluation for shut states exhange A by F and F by A in function call.

        SFF = I - exp(QFF * tres)
        First evaluate [dVA(s) / ds] * s = 0.
        dVAds = -inv(QAA) * GAF * SFF * GFA - GAF * SFF * inv(QFF) * GFA +
        + tres * GAF * expQFF * GFA

        Then: DARS = inv(VA) * QAA^(-2) - inv(VA) * dVAds * inv(VA) * inv(QAA) =
        = inv(VA) * [inv(QAA) - dVAds * inv(VA)] * inv(QAA)
        where VA = I - GAF * SFF * GFA

        Returns
        -------
        DARS : array_like, shape (kA, kA)
        """

        invQAA = nplin.inv(self.QAA)
        invQFF = nplin.inv(self.QFF)

        SAA = self.IA - self.expQAA
        Q1 = self.tres * self.GFA @ self.expQAA @ self.GAF
        Q2 = self.GFA @ SAA @ invQAA @ self.GAF
        Q3 = -invQFF @ self.GFA @ SAA @ self.GAF

        VF = self.IF - self.GFA @ SAA @ self.GAF
        Q4 = invQFF - (Q1 - Q2 + Q3) @ nplin.inv(VF)
        return nplin.inv(VF) @ Q4 @ invQFF


class ExactPDFCalculator(HJCMatrix):
    def __init__(self, mec, tres=0.0):
        """
        Initialize the ExactPDFCalculator.

        Parameters
        ----------
        Q : ndarray
            The Q matrix representing the transition rates.
        kA, kB, kC, kD : int, optional
            Dimensions of different state subspaces. Defaults are 1 for kA and kB, 0 for kC and kD.
        """
        super().__init__(mec.Q, kA=mec.kA, kB=mec.kB, kC=mec.kC, kD=mec.kD, tres=tres)
   
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


### Functions to review

def HAF(roots, tres, tcrit, QAF, expQFF, R):
    """
    Parameters
    ----------
    roots : array_like, shape (1, kA)
        Roots of the asymptotic pdf.
    tres : float
        Time resolution (dead time).
    tcrit : float
        Critical time.
    QAF : array_like, shape(kA, kF)
    expQFF : array_like, shape(kF, kF)
    R : array_like, shape(kA, kA, kA)

    Returns
    -------
    HAF : ndarray, shape(kA, kF)
    """

    coeff = -np.exp(roots * (tcrit - tres)) / roots
    temp = np.sum(R * coeff.reshape(R.shape[0],1,1), axis=0)
    return temp @ QAF @ expQFF

def CHSvec(roots, tres, tcrit, QFA, kA, expQAA, phiF, R):
    """
    Calculate initial and final CHS vectors for HJC likelihood function
    (Eqs. 5.5 or 5.7, CHS96).

    Parameters
    ----------
    roots : array_like, shape (1, kA)
        Roots of the asymptotic pdf.
    tres : float
        Time resolution (dead time).
    tcrit : float
        Critical time.
    QFA : array_like, shape(kF, kA)
    kA : int
    expQAA : array_like, shape(kA, kA)
    phiF : array_like, shape(1, kF)
    R : array_like, shape(kF, kF, kF)

    Returns
    -------
    start : ndarray, shape (1, kA)
        CHS start vector (Eq. 5.11, CHS96).
    end : ndarray, shape (kF, 1)
        CHS end vector (Eq. 5.8, CHS96).
    """

    H = HAF(roots, tres, tcrit, QFA, expQAA, R)
    u = np.ones((kA, 1))
    start = (phiF @ H) / (phiF @ H @ u)
    end = H @ u

    return start, end

def eGAF(t, tres, eigvals, Z00, Z10, Z11, roots, R, QAF, expQFF):
    #TODO: update documentation
    """
    Calculate transition density eGAF(t) for exact (Eq. 3.2, HJC90) and
    asymptotic (Eq. 3.24, HJC90) distribution.

    Parameters
    ----------
    t : float
        Time interval.
    tres : float
        Time resolution (dead time).
    eigvals : array_like, shape (1, k)
        Eigenvalues of -Q matrix.
    Z00, Z10, Z11 : array_like, shape (k, kA, kF)
        Z constants for the exact open time pdf.
    roots : array_like, shape (1, kA)
        Roots of the asymptotic pdf.
    R : array_like, shape(kA, kA, kA)
    QAF : array_like, shape(kA, kF)
    expQFF : array_like, shape(kF, kF)

    Returns
    -------
    eGAFt : array_like, shape(kA, kF)
    """

    if t < (tres * 2): # exact
        eGAFt = f0((t - tres), eigvals, Z00)
    elif t < (tres * 3):
        eGAFt = (f0((t - tres), eigvals, Z00) -
            f1((t - 2 * tres), eigvals, Z10, Z11))
    else: # asymptotic
        temp = np.sum(R * np.exp(roots *
            (t - tres)).reshape(R.shape[0],1,1), axis=0)
        eGAFt = np.dot(np.dot(temp, QAF), expQFF)

    return eGAFt

def f0(u, eigvals, Z00):
    """
    A component of exact time pdf (Eq. 22, HJC92).

    Parameters
    ----------
    u : float
        u = t - tres
    eigvals : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    Z00 : list of array_likes
        Constants for the exact open/shut time pdf.
        Z00 for likelihood calculation or gama00 for time distributions.

    Returns
    -------
    f : ndarray
    """

#    f = np.zeros(Z00[0].shape)
#    for i in range(len(eigvals)):
#        f += Z00[i] *  math.exp(-eigvals[i] * u)

    if Z00.ndim > 1:
        f = np.sum(Z00 *  np.exp(-eigvals * u).reshape(Z00.shape[0],1,1),
            axis=0)
    else:
        f = np.sum(Z00 *  np.exp(-eigvals * u))
    return f

def f1(u, eigvals, Z10, Z11):
    """
    A component of exact time pdf (Eq. 22, HJC92).

    Parameters
    ----------
    u : float
        u = t - tres
    eigvals : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    Z10, Z11 (or gama10, gama11) : list of array_likes
        Constants for the exact open/shut time pdf. Z10, Z11 for likelihood
        calculation or gama10, gama11 for time distributions.

    Returns
    -------
    f : ndarray
    """

#    f = np.zeros(Z10[0].shape)
#    for i in range(len(eigvals)):
#        f += (Z10[i] + Z11[i] * u) *  math.exp(-eigvals[i] * u)

    if Z10.ndim > 1:
        f = np.sum((Z10 + Z11 * u) *
            np.exp(-eigvals * u).reshape(Z10.shape[0],1,1), axis=0)
    else:
        f = np.sum((Z10 + Z11 * u) * np.exp(-eigvals * u))
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


### Deprecated functions ##############################################

@deprecated("Use 'HJCMatrix'")
def detW(s, tres, QAA, QFF, QAF, QFA, kA, kF):
    """
    Calculate determinant of WAA(s).
    To evaluate WFF(s) exhange A by F and F by A in function call.

    Parameters
    ----------
    s : float
        Laplace transform argument.
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
    detWAA : float
    """

    return nplin.det(W(s, tres, QAA, QFF, QAF, QFA, kA, kF))

@deprecated("Use 'ExactPDFCalculator'")
def Zxx(Q, eigen, A, kopen, QFF, QAF, QFA, expQFF, open):
    """
    Calculate Z constants for the exact open time pdf (Eq. 3.22, HJC90).
    Exchange A and F for shut time pdf.

    TODO: remove eigenvalues from return of this function

    Parameters
    ----------
    t : float
        Time.
    Q : array_like, shape (k, k)
    kopen : int
        Number of open states.
    QFF, QAF, QFA : array_like
        Submatrices of Q.
    open : bool
        True for open time pdf, False for shut time pdf.

    Returns
    -------
    eigen : array_like, shape (k,)
        Eigenvalues of -Q matrix.
    Z00, Z10, Z11 : array_like, shape (k, kA, kF)
        Z constants for the exact open time pdf.
    """

    k = Q.shape[0]
    kA = k - QFF.shape[0]
#    eigen, A = eigs(-Q)
    # Maybe needs check for equal eigenvalues.

    # Calculate Dj (Eq. 3.16, HJC90) and Cimr (Eq. 3.18, HJC90).
    D = np.empty((k))
    if open:
        C00 = A[:, :kopen, :kopen]
        A1 = A[:, :kopen, kopen:]
    else:
        C00 = A[:, kopen:, kopen:]
        A1 = A[:, kopen:, :kopen]
    D = np.dot(np.dot(A1, expQFF), QFA)

    C11 = np.empty((k, kA, kA))
    #TODO: try to remove 'for' cycles
    for i in range(k):
        C11[i] = np.dot(D[i], C00[i])

    C10 = np.empty((k, kA, kA))
    #TODO: try to remove 'for' cycles
    for i in range(k):
        S = np.zeros((kA, kA))
        for j in range(k):
            if j != i:
                S += ((np.dot(D[i], C00[j]) + np.dot(D[j], C00[i])) /
                    (eigen[j] - eigen[i]))
        C10[i] = S

    M = np.dot(QAF, expQFF)
    Z00 = np.array([np.dot(C, M) for C in C00])
    Z10 = np.array([np.dot(C, M) for C in C10])
    Z11 = np.array([np.dot(C, M) for C in C11])

    return Z00, Z10, Z11

@deprecated("Use 'scalcslib.CSDwells'")
def phiHJC(eGAF, eGFA, kA):
    """
    Calculate initial HJC vector for openings by solving
    phi*(I-eGAF*eGFA)=0 (Eq. 10, HJC92)
    For shuttings exhange A by F and F by A in function call.

    Parameters
    ----------
    eGAF : array_like, shape (kA, kF)
    eGFA : array_like, shape (kF, kA)
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.

    Returns
    -------
    phi : array_like, shape (kA)
    """

    if kA == 1:
        phi = np.array([1])

    else:
        Qsub = np.eye(kA) - np.dot(eGAF, eGFA)
        u = np.ones((kA, 1))
        S = np.concatenate((Qsub, u), 1)
        phi = np.dot(u.transpose(), nplin.inv(np.dot(S, S.transpose())))[0]

    return phi

@deprecated("Use 'scalcslib.DerivativeCalculator")
def dARSdS(tres, QAA, QFF, GAF, GFA, expQFF, kA, kF):
    r"""
    Evaluate the derivative with respect to s of the Laplace transform of the
    survival function (Eq. 3.6, CHS96) for open states:

    .. math::

       \left[ -\frac{\text{d}}{\text{d}s} {^\cl{A}\!\bs{R}^*(s)} \right]_{s=0}

    For same evaluation for shut states exhange A by F and F by A in function call.

    SFF = I - exp(QFF * tres)
    First evaluate [dVA(s) / ds] * s = 0.
    dVAds = -inv(QAA) * GAF * SFF * GFA - GAF * SFF * inv(QFF) * GFA +
    + tres * GAF * expQFF * GFA

    Then: DARS = inv(VA) * QAA^(-2) - inv(VA) * dVAds * inv(VA) * inv(QAA) =
    = inv(VA) * [inv(QAA) - dVAds * inv(VA)] * inv(QAA)
    where VA = I - GAF * SFF * GFA

    Parameters
    ----------
    tres : float
        Time resolution (dead time).
    QAA : array_like, shape (kA, kA)
    QAF : array_like, shape (kA, kF)
    QFF : array_like, shape (kF, kF)
    QFA : array_like, shape (kF, kA)
        Q11, Q12, Q22, Q21 - submatrices of Q.
    GAF : array_like, shape (kA, kF)
    GFA : array_like, shape (kF, kA)
        GAF, GFA - G matrices.
    expQFF : array_like, shape(kF, kF)
    expQAA : array_like, shape(kA, kA)
        expQFF, expQAA - exponentials of submatrices QFF and QAA.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.

    Returns
    -------
    DARS : array_like, shape (kA, kA)
    """

    invQAA = nplin.inv(QAA)
    invQFF = nplin.inv(QFF)

    #SFF = I - EXPQF
    I = np.eye(kF)
    SFF = I - expQFF

    #Q1 = tres * GAF * exp(QFF*tres) * GFA
    Q1 = tres * np.dot(GAF, np.dot(expQFF, GFA))
    #Q2 = GAF * SFF * inv(QFF) * GFA
    Q2 = np.dot(GAF, np.dot(SFF, np.dot(invQFF, GFA)))
    #Q3 = -inv(QAA) * GAF * SFF * GFA
    Q3 = np.dot(np.dot(np.dot(-invQAA, GAF), SFF), GFA)
    Q1 = Q1 - Q2 + Q3

    # VA = I - GAF * SFF * GFA
    I = np.eye(kA)
    VA = I - np.dot(np.dot(GAF, SFF), GFA)

    # DARS = inv(VA) * (QAA**-2) - inv(VA) * Q1 * inv(VA) * inv(QAA) =
    #      = inv(VA) * [inv(QAA) - Q1 * inv(VA)] * inv(QAA)
    Q3 = invQAA + - np.dot(Q1, nplin.inv(VA))
    DARS = np.dot(np.dot(nplin.inv(VA), Q3), invQAA)

    return DARS

@deprecated("Use 'HJCMatrix'")
def H(s, tres, QAA, QFF, QAF, QFA, kF):
    """
    Evaluate H(s) funtion (Eq. 54, HJC92).
    HAA(s) = QAA + QAF * (s*I - QFF) ^(-1) * (I - exp(-(s*I - QFF) * tau)) * QFA
    To evaluate HFF(s) exhange A by F and F by A in function call.

    Parameters
    ----------
    s : float
        Laplace transform argument.
    tres : float
        Time resolution (dead time).
    QAA : array_like, shape (kA, kA)
    QFF : array_like, shape (kF, kF)
    QAF : array_like, shape (kA, kF)
    QFA : array_like, shape (kF, kA)
        QAA, QFF, QAF, QFA - submatrices of Q.
    kF : int
        A number of shut states in kinetic scheme.

    Returns
    -------
    H : ndarray, shape (kA, kA)
    """

    IF = np.eye(kF)
    XFF = s * IF - QFF
    invXFF = nplin.inv(XFF)
    expXFF = expQ(-XFF, tres)
    H = QAA + np.dot(np.dot(np.dot(QAF, invXFF), IF - expXFF), QFA)
    return H

@deprecated("Use 'HJCMatrix'")
def W(s, tres, QAA, QFF, QAF, QFA, kA, kF):
    """
    Evaluate W(s) function (Eq. 52, HJC92).
    WAA(s) = s * IA - HAA(s)
    To evaluate WFF(s) exhange A by F and F by A in function call.

    Parameters
    ----------
    s : float
        Laplace transform argument.
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
    W : ndarray, shape (k2, k2)
    """

    IA = np.eye(kA)
    W = s * IA - H(s, tres, QAA, QFF, QAF, QFA, kF)
    return W

@deprecated("Use 'HJCMatrix'")
def dW(s, tres, QAF, QFF, QFA, kA, kF):
    """
    Evaluate the derivative with respect to s of the matrix W(s) at the root s
    (Eq. 56, HJC92) for open states. For same evaluation for shut states
    exhange A by F and F by A in function call.
    W'(s) = I + QAF * [SFF(s) * (s*I - QFF)^(-1) - tau * (I - SFF(s))] * eGFA(s)
    where SFF(s) = I - exp(-(s*I - QFF) * tau) (Eq. 17, HJC92)
    and eGFA(s) = (s*I - QFF)^(-1) * QFA (Eq. 4, HJC92).

    Parameters
    ----------
    s : float
        Laplace transform argument.
    tres : float
        Time resolution (dead time).
    QAF : array_like, shape (kA, kF)
    QFF : array_like, shape (kF, kF)
    QFA : array_like, shape (kF, kA)
        QAF, QFF, QFA - submatrices of Q.
    kA : int
        A number of open states in kinetic scheme.
    kF : int
        A number of shut states in kinetic scheme.

    Returns
    -------
    dW : ndarray, shape (kF, kF)
    """

    IF = np.eye(kF)
    IA = np.eye(kA)
    XFF = s * IF - QFF
    expXFF = expQ(-XFF, tres)
    SFF = IF - expXFF
    eGFAs = np.dot(nplin.inv(s * IF - QFF), QFA)
    w1 = np.dot(SFF, nplin.inv(s * IF - QFF)) - tres * (IF - SFF)
    dW = IA + np.dot(np.dot(QAF, w1), eGFAs)
    return dW

@deprecated("Use 'HJCMatrix'")
def AR(roots, tres, QAA, QFF, QAF, QFA, kA, kF):
    """
    
    Parameters
    ----------
    roots : array_like, shape (1, kA)
        Roots of the asymptotic pdf.
    tres : float
        Time resolution (dead time).
    QAA, QFF, QAF, QFA : array_like
        Submatrices of Q.
    kA, kF : ints
        Number of open and shut states.

    Returns
    -------
    R : ndarray, shape(kA, kA, kA)
    """

    R = np.zeros((kA, kA, kA))
    row = np.zeros((kA, kA))
    col1 = np.zeros((kA, kA))
    for i in range(kA):
        WA = W(roots[i], tres, QAA, QFF, QAF, QFA, kA, kF)
        AW = np.transpose(WA)

        row[i] = pinf(WA)
        col1[i] = pinf(AW)

#        try:
#            row[i] = pinf(WA)
#        except:
#            row[i] = pinf1(WA)
        
#        try:
#            col1[i] = pinf(AW)
#        except:
#            col1[i] = pinf1(AW)
    col = col1.transpose()

    for i in range(kA):
        nom = np.dot(col[:,i].reshape((kA, 1)), row[i,:].reshape((1, kA)))
        W1A = dW(roots[i], tres, QAF, QFF, QFA, kA, kF)
        denom = np.dot(np.dot(row[i,:].reshape((1, kA)), W1A),
            col[:,i].reshape((kA, 1)))
        R[i] = nom / denom

    return R


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



