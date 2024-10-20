import numpy as np
import numpy.linalg as nplin
from deprecated import deprecated
from functools import cached_property

def eigenvalues_and_spectral_matrices(Q, do_sorting=True):
    """
    Calculate eigenvalues and spectral matrices of a matrix Q.

    Parameters
    ----------
    Q : array_like, shape (k, k)
        Input matrix whose eigenvalues and spectral matrices are to be computed.
    do_sorting : bool, optional (default=True)
        If True, sorts the eigenvalues and spectral matrices based on the real part of the eigenvalues.

    Returns
    -------
    eigvals : ndarray, shape (k,)
        Eigenvalues of Q.
    A : ndarray, shape (k, k, k)
        Spectral matrices of Q.
    """
    eigvals, M = nplin.eig(Q)
    N = nplin.inv(M)
    #A = np.einsum('ij,kj->kij', M, N)  # Efficient matrix outer product
    k = N.shape[0]
    A = np.zeros((k, k, k))
    for i in range(k):
        A[i] = np.dot(M[:, i].reshape(k, 1), N[i].reshape(1, k))

    if do_sorting:
        sorted_indices = eigvals.real.argsort()
        eigvals = eigvals[sorted_indices]
        A = A[sorted_indices]

    return eigvals, A

def expQ(Q, t):
    """
    Calculate the matrix exponential of Q * t.

    Parameters
    ----------
    Q : array_like, shape (k, k)
        Input matrix.
    t : float
        Time scalar.

    Returns
    -------
    expQ : ndarray, shape (k, k)
        Exponential of matrix Q * t.
    """
    eigvals, A = eigenvalues_and_spectral_matrices(Q)
    return np.sum(A * np.exp(eigvals * t).reshape(-1, 1, 1), axis=0)

def powQ(Q, n):
    """
    Raise matrix Q to the power of n.

    Parameters
    ----------
    Q : array_like, shape (k, k)
        Input square matrix.
    n : int
        Power to which the matrix is to be raised.

    Returns
    -------
    Qn : ndarray, shape (k, k)
        Matrix Q raised to the power of n.
    """
    eigvals, A = eigenvalues_and_spectral_matrices(Q)
    return np.sum(A * (eigvals**n).reshape(-1, 1, 1), axis=0)

def pinf(Q):
    """Calculate equilibrium occupancies."""
    try:
        return pinf_extendQ(Q)
    except np.linalg.LinAlgError:
        return pinf_reduceQ(Q)

def pinf_extendQ(Q):
    """
    Calculate equilibrium occupancies by adding a column of ones to Q matrix.
    Pinf = uT * invert((S * transpos(S))).

    Parameters
    ----------
    Q : array_like, shape (k, k)

    Returns
    -------
    pinf : ndarray, shape (k1)
    """

    u = np.ones((Q.shape[0],1))
    extended_Q_matrix = np.hstack((Q, u))
    return (u.T @ nplin.inv(extended_Q_matrix @ extended_Q_matrix.T))[0]

def pinf_reduceQ(Q):
    """
    Calculate equilibrium occupancies with the reduced Q-matrix method.

    Parameters
    ----------
    Q : array_like, shape (k, k)

    Returns
    -------
    pinf : ndarray, shape (k1)
    """

    reduced_Q_matrix = (Q - Q[-1:, :])[:-1, :-1]
    temp = -Q[-1:, :-1] @ nplin.inv(reduced_Q_matrix)
    return np.append(temp, 1 - np.sum(temp))


class QMatrix:
    '''
    Transition rate matrix Q.
    '''

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0):
        """
        Initialize the QMatrix instance.

        Parameters:
        Q (np.ndarray): Transition rate matrix.
        kA, kB, kC, kD (int): State counts for different categories.
        """
        self.Q = Q
        self.kA, self.kB, self.kC, self.kD = kA, kB, kC, kD
        #TODO: sanity check- is self.k == self.num_states 
        self.k = self.kA + self.kB + self.kC + self.kD  # all states
        self.num_states = Q.shape[0]

        self._set_state_counts()
        self._set_submatrices()
        self._set_unity_vectors()

        #self.pinf = pinf(self.Q)
        self.pinf = self._calculate_pinf()

        self.GAF = self._GXY(self.QAA, self.QAF) 
        self.GFA = self._GXY(self.QFF, self.QFA)
        self.GAB = self._GXY(self.QAA, self.QAB)
        self.GBA = self._GXY(self.QBB, self.QBA)

    def _calculate_pinf(self):
        """ Placeholder for calculating steady-state probabilities (pinf). """
        #TODO: consider implementing the actual pinf computation here
        return pinf(self.Q)

    @cached_property
    def phiA(self):
        """ Calculate initial vector for openings. """
        nom = self.pinf[self.kA : ] @ self.QIA
        return nom / (nom @ self.uA)
    
    @cached_property
    def phiF(self):
        """ Calculate inital vector for shuttings. """
        return self.phiA @ self.GAF
    
    def _GXY(self, QXX, QXY):
        r"""
        Calculate G matrix (Eq. 1.25, CH82).
        Calculate GAB, GBA, GFA, GAF, etc by replacing X and Y with required subsets (e.g. A, B, F).

        .. math::

        \bs{G}_\cl{AB} &= -\bs{Q}_\cl{AA}^{-1} \bs{Q}_\cl{AB}

        """
        return nplin.inv(-QXX) @ QXY

    def state_lifetimes(self):
        """ Calculate state lifetimes based on diagonal elements of Q. """
        return -1 / np.diag(self.Q)

    def transition_probability(self):
        """ Calculate the transition probabilities. """
        transition_probability = self.Q.copy()
        np.fill_diagonal(transition_probability, 0)
        row_sums = -np.diag(self.Q)
        return transition_probability / row_sums[:, np.newaxis]

    def transition_frequency(self):
        """ Calculate the frequency of transitions. """
        transition_frequency = self.Q.T * self.pinf
        np.fill_diagonal(transition_frequency, 0)
        return transition_frequency

    def _set_state_counts(self):
        """Calculate state counts for various subsets."""
        self.kE = self.kA + self.kB  # burst states
        self.kF = self.kB + self.kC  # intra and inter burst shut states
        self.kG = self.kA + self.kB + self.kC  # cluster states
        self.kH = self.kC + self.kD  # gap between clusters states
        self.kI = self.kB + self.kC + self.kD  # all shut states
        
    def _set_submatrices(self):
        """Extract submatrices from the main Q matrix."""
        self.QFF = self.Q[self.kA:self.kG, self.kA:self.kG]
        self.QFA = self.Q[self.kA:self.kG, :self.kA]
        self.QAF = self.Q[:self.kA, self.kA:self.kG]
        self.QAA = self.Q[:self.kA, :self.kA]
        self.QEE = self.Q[:self.kE, :self.kE]
        self.QBB = self.Q[self.kA:self.kE, self.kA:self.kE]
        self.QAB = self.Q[:self.kA, self.kA:self.kE]
        self.QBA = self.Q[self.kA:self.kE, :self.kA]
        self.QBC = self.Q[self.kA:self.kE, self.kE:self.kG]
        self.QAC = self.Q[:self.kA, self.kE:self.kG]
        self.QCB = self.Q[self.kE:self.kG, self.kA:self.kE]
        self.QCA = self.Q[self.kE:self.kG, :self.kA]
        self.QII = self.Q[self.kA:self.k, self.kA:self.k]
        self.QIA = self.Q[self.kA:self.k, :self.kA]
        self.QAI = self.Q[:self.kA, self.kA:self.k]
        self.QGG = self.Q[:self.kG, :self.kG]

    def _set_unity_vectors(self):
        """Initialize unity vectors."""
        self.uk = np.ones((self.k, 1))
        self.uA = np.ones((self.kA, 1))
        self.uB = np.ones((self.kB, 1))
        self.uC = np.ones((self.kC, 1))
        self.uF = np.ones((self.kF, 1))
        self.IA = np.eye(self.kA)
        self.IF = np.eye(self.kF)

    def P(self, subset='inf'):
        """ Calculate the probability of being in specified subsets of states. """
        subsets = {
            'A': np.sum(self.pinf[        : self.kA]),
            'B': np.sum(self.pinf[self.kA : self.kE]),
            'C': np.sum(self.pinf[self.kE :        ]),
            'F': np.sum(self.pinf[self.kA :        ]),
            'B|F': np.sum(self.pinf[self.kA : self.kE]) / np.sum(self.pinf[self.kA : ]),
            'C|F': np.sum(self.pinf[self.kE :        ]) / np.sum(self.pinf[self.kA : ])
        }
        return subsets.get(subset, self.pinf)

    def phi(self, subset='A'):
        """ Calculate the conditional probability distribution over a subset of states. """
        phi_dict = {
            'A': self.pinf[        : self.kA] / np.sum(self.pinf[        : self.kA]),
            'B': self.pinf[self.kA : self.kE] / np.sum(self.pinf[self.kA : self.kE]),
            'F': self.pinf[self.kA : self.kG] / np.sum(self.pinf[self.kA : self.kG]),
        }
        return phi_dict.get(subset)
    
    def Popen(self):
        """ Calculate equilibrium open probability, Popen. """
        return np.sum(self.pinf[ : self.kA]) / np.sum(self.pinf)

    def subset_mean_lifetime(self, state1, state2):
        """
        Calculate the mean life time in a specified subset. Add all rates out of subset
        to get total rate out. Skip rates within subset.

        Parameters
        ----------
        state1,state2 : int
            State numbers (counting origin 1)
        """
        state1, state2 = state1 - 1, state2 - 1  # Convert to zero-indexed
        subset_pinf = self.pinf[state1 : state2+1]
        pstot = np.sum(subset_pinf) # Total occupancy for the subset
        if pstot == 0:
            return 0.0
        subset_Q_pinf = [subset_pinf] @ self.Q[state1 : state2+1, :]
        rate_out_of_subset = np.sum(subset_Q_pinf) - np.sum(subset_Q_pinf[ : , state1 : state2+1])
        return pstot / rate_out_of_subset

    def mean_latency_given_start_state(self, state):
        """
        Calculate mean latency to next opening (shutting), given starting in
        specified shut (open) state.

        mean latency given starting state = pF(0) * inv(-QFF) * uF

        F- all shut states (change to A for mean latency to next shutting
        calculation), p(0) = [0 0 0 ..1.. 0] - a row vector with 1 for state in
        question and 0 for all other states.

        Parameters
        ----------
        state : int
            State number (counting origin 1)
        """

        is_opening = state <= self.kA
        p_size = self.kA if is_opening else self.kI
        adjusted_state = state - 1 if is_opening else state - self.kA - 1
        Q_sub = self.QAA if is_opening else self.QII
        
        p = np.zeros(p_size)
        p[adjusted_state] = 1 
        invQ = nplin.inv(-Q_sub) 
        u = np.ones((p_size, 1)) 
        return (p @ invQ @ u)[0]  
    
    def ideal_open_time_pdf_components(self):
        """Calculate exponential open time PDF components."""
        return self._ideal_time_pdf_components(self.QAA, self.phiA, self.uA, self.kA)

    def ideal_shut_time_pdf_components(self):
        """Calculate exponential shut time PDF components."""
        return self._ideal_time_pdf_components(self.QFF, self.phiF, self.uF, self.kF)

    def _ideal_time_pdf_components(self, Q, phi, u, size):
        """Helper function for calculating PDF components."""
        eigs, A = eigenvalues_and_spectral_matrices(-Q)
        w = np.zeros(size)
        for i in range(size):
            w[i] = (phi @ A[i] @ -Q @ u)[0]
        #w = np.einsum('ij,ijk,kl->i', phi, A, (-Q) @ u)
        return eigs, w
    
    def ideal_dwell_time_pdf_direct(self, t, open=True):
        """
        Probability density function of the open time.
        f(t) = phiOp * exp(-QAA * t) * (-QAA) * uA
        """
        phiX = self.phiA if open else self.phiF
        QXX = self.QAA if open else self.QFF
        u = self.uA if open else self.uF
        return phiX @ expQ(QXX, t) @ -QXX @ u

    def ideal_subset_time_pdf(self, Q, k1, k2, t):
        """Calculate time PDF for a subset of states."""
        u = np.ones((k2 - k1 + 1, 1))
        phi, QSub = phiSub(Q, k1, k2)
        expQSub = expQ(QSub, t)
        return phi @ expQSub @ -QSub @ u


class HJCMatrix(QMatrix):
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
        self.expQFF = expQ(self.QFF, self.tres)
        self.expQAA = expQ(self.QAA, self.tres)

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
        expX = expQ(-(s * I_other - Q_other1), self.tres)

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
        expX = expQ(-X, self.tres)
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


### Functions to review

def iGt(t, QAA, QAB):
    """
    GAB(t) = PAA(t) * QAB      Eq. 1.20 in CH82
    PAA(t) = exp(QAA * t)      Eq. 1.16 in CH82
    """
    return expQ(QAA, t) @ QAB

def phiSub(Q, k1, k2):
    """
    Calculate initial vector for any subset.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    phi : ndarray, shape (kA)
    """

    u = np.ones((k2 - k1 + 1, 1))
    p1, p2, p3 = np.hsplit(pinf(Q),(k1, k2+1))
    p1c = np.hstack((p1, p3))

    #Q = Q.copy()
    Q1, Q2, Q3 = np.hsplit(Q,(k1, k2+1))
    Q21, Q22, Q23 = np.hsplit(Q2.transpose(),(k1, k2+1))
    Q22c = Q22.copy()
    Q12 = np.vstack((Q21.transpose(), Q23.transpose()))

    nom = np.dot(p1c, Q12)
    phi = nom / (nom @ u)
    return phi, Q22c

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


@deprecated("Use 'QMatrix'")
def GXY(QXX, QXY):
    r"""
    Calculate G matrix (Eq. 1.25, CH82).
    Calculate GAB, GBA, GFA, GAF, etc by replacing X and Y with required subsets (e.g. A, B, F).

    .. math::

       \bs{G}_\cl{AB} &= -\bs{Q}_\cl{AA}^{-1} \bs{Q}_\cl{AB}

    Parameters
    ----------
    QXX, QXY : array_like, shape (kX, kX), (kX, kY)
        Q-matrix subsets.

    Returns
    -------
    GXY : ndarray, shape (kX, kY)
    """

    return nplin.inv(-QXX) @ QXY

@deprecated("Use 'HJCMatrix'")
def eGs(GAF, GFA, kA, kF, expQFF):
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

@deprecated("Use 'eigenvalues_and_spectral_matrices'")
def eigs(Q):
    """
    Calculate eigenvalues and spectral matrices of a matrix Q.

    Parameters
    ----------
    Q : array_like, shape (k, k)

    Returns
    -------
    eigvals : ndarray, shape (k,)
        Eigenvalues of M.
    A : ndarray, shape (k, k, k)
        Spectral matrices of Q.
    """

    eigvals, M = nplin.eig(Q)
    N = nplin.inv(M)
    k = N.shape[0]
    A = np.zeros((k, k, k))
    # TODO: make this a one-liner avoiding loops
    # DO NOT DELETE commented explicit loops for future reference
    #
    # rev. 1
    # for i in range(k):
    #     X = np.empty((k, 1))
    #     Y = np.empty((1, k))
    #     X[:, 0] = M[:, i]
    #     Y[0] = N[i]
    #     A[i] = np.dot(X, Y)
    # END DO NOT DELETE
    #
    # rev. 2 - cumulative time fastest on my system
    for i in range(k):
        A[i] = np.dot(M[:, i].reshape(k, 1), N[i].reshape(1, k))

    # rev. 3 - cumulative time not faster
    # A = np.array([
    #         np.dot(M[:, i].reshape(k, 1), N[i].reshape(1, k)) \
    #             for i in range(k)
    #         ])

    return eigvals, A

@deprecated("Use 'eigenvalues_and_spectral_matrices'")
def eigs_sorted(Q):
    """
    Calculate eigenvalues and spectral matrices of a matrix Q. 
    Return eigenvalues in ascending order 

    Parameters
    ----------
    Q : array_like, shape (k, k)

    Returns
    -------
    eigvals : ndarray, shape (k,)
        Eigenvalues of M.
    A : ndarray, shape (k, k, k)
        Spectral matrices of Q.
    """

    eigvals, M = nplin.eig(Q)
    N = nplin.inv(M)
    k = N.shape[0]
    A = np.zeros((k, k, k))
    for i in range(k):
        A[i] = np.dot(M[:, i].reshape(k, 1), N[i].reshape(1, k))
    sorted_indices = eigvals.real.argsort()
    eigvals = eigvals[sorted_indices]
    A = A[sorted_indices, : , : ]
    return eigvals, A

@deprecated("Use 'expQ'")
def expQt(M, t):
    """
    Calculate exponential of a matrix M.
        expM = exp(M * t)

    Parameters
    ----------
    M : array_like, shape (k, k)
    t : float
        Time.

    Returns
    -------
    expM : ndarray, shape (k, k)
    """

    eigvals, A = eigs(M)

    # DO NOT DELETE commented explicit loops for future reference
    # k = M.shape[0]
    # expM = np.zeros((k, k))
    # rev. 1
    # TODO: avoid loops
    #    for i in range(k):
    #        for j in range(k):
    #            for m in range(k):
    #                expM[i, j] += A[m, i, j] * math.exp(eigvals[m] * t)
    #
    # rev.2:
    # for m in range(k):
    #     expM += A[m] * math.exp(eigvals[m] * t)
    # END DO NOT DELETE

    expM = np.sum(A * np.exp(eigvals * t).reshape(A.shape[0],1,1), axis=0)
    return expM


@deprecated("Use 'powQ'")
def Qpow(M, n):
    """
    Rise matrix M to power n.

    Parameters
    ----------
    M : array_like, shape (k, k)
    n : int
        Power.

    Returns
    -------
    Mn : ndarray, shape (k, k)
    """

    k = M.shape[0]
    eig, A = eigs(M)
    Mn = np.zeros((k, k))
    for i in range(k):
        Mn += A[i, :, :] * pow(eig[i], n)
    return Mn

@deprecated("Use 'pinf_extendQ'")
def pinf1(Q):
    """
    Calculate equilibrium occupancies by adding a column of ones
    to Q matrix.
    Pinf = uT * invert((S * transpos(S))).

    Parameters
    ----------
    Q : array_like, shape (k, k)

    Returns
    -------
    pinf : ndarray, shape (k1)
    """

    u = np.ones((Q.shape[0],1))
    S = np.concatenate((Q, u), 1)
    pinf = np.dot(u.transpose(), nplin.inv((np.dot(S,S.transpose()))))[0]
    return pinf
    
#@deprecated("Use 'pinf_reduceQ'")
#def pinf(Q):
#    """
#    Calculate equilibrium occupancies with the reduced Q-matrix method.
#
#    Parameters
#    ----------
#    Q : array_like, shape (k, k)
#
#    Returns
#    -------
#    pinf : ndarray, shape (k1)
#    """
#
#    R = (Q - Q[-1: , :])[:-1, :-1]
#    r = Q[-1: , :-1]
#    pinf = -np.dot(r, nplin.inv(R))
#    pinf = np.append(pinf, 1 - np.sum(pinf))
#    return pinf

@deprecated("Use 'GAB'")
def iGs(Q, kA, kB):
    r"""
    Calculate GBA and GAB matrices (Eq. 1.25, CH82).
    Calculate also GFA and GAF if kF is given instead of kB.

    .. math::

       \bs{G}_\cl{BA} &= -\bs{Q}_\cl{BB}^{-1} \bs{Q}_\cl{BA} \\
       \bs{G}_\cl{AB} &= -\bs{Q}_\cl{AA}^{-1} \bs{Q}_\cl{AB}

    Parameters
    ----------
    Q : array_like, shape (k, k)
    kA : int
        A number of open states in kinetic scheme.
    kB : int
        A number of short lived shut states in kinetic scheme.

    Returns
    -------
    GAB : ndarray, shape (kA, kB)
    GBA : ndarray, shape (kB, kA)
    """

    kE = kA + kB
    QBB = Q[kA:kE, kA:kE]
    QBA = Q[kA:kE, 0:kA]
    QAA = Q[0:kA, 0:kA]
    QAB = Q[0:kA, kA:kE]
    GAB = np.dot(nplin.inv(-1 * QAA), QAB)
    GBA = np.dot(nplin.inv(-1 * QBB), QBA)
    return GAB, GBA

@deprecated("Use 'scalcslib.CSDwells'")
def phiA(mec):
    """
    Calculate initial vector for openings.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    phi : ndarray, shape (kA)
    """

    uA = np.ones((mec.kA,1))
    pI = pinf(mec.Q)[mec.kA:]
    nom = np.dot(pI, mec.QIA)
    denom = np.dot(nom,uA)
    phi = nom / denom
    return phi

@deprecated("Use 'scalcslib.CSDwells'")
def phiF(mec):
    """
    Calculate inital vector for shuttings.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.

    Returns
    -------
    phi : ndarray, shape (kF)
    """

    GAF, GFA = iGs(mec.Q, mec.kA, mec.kI)
    phi = np.dot(phiA(mec), GAF)
    return phi

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
