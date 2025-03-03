import numpy as np
import numpy.linalg as nplin
from deprecated import deprecated
from functools import cached_property
import warnings

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

    def __init__(self, mec): #Q, kA=1, kB=1, kC=0, kD=0):
        """
        Initialize the QMatrix instance.

        Parameters:
        Q (np.ndarray): Transition rate matrix.
        kA, kB, kC, kD (int): State counts for different categories.
        """
        self.mec = mec
        self.Q = mec.Q
        self.kA, self.kB, self.kC, self.kD = mec.kA, mec.kB, mec.kC, mec.kD
        #TODO: sanity check- is self.k == self.num_states 
        self.k = self.kA + self.kB + self.kC + self.kD  # all states
        self.num_states = self.Q.shape[0]

        self._set_state_counts()
        self._set_submatrices()
        self._set_unity_vectors()

        #self.pinf = pinf(self.Q)
        self.pinf = self._calculate_pinf()

        self.GAF = self._GXY(self.QAA, self.QAF) 
        self.GFA = self._GXY(self.QFF, self.QFA)
        self.GAB = self._GXY(self.QAA, self.QAB)
        self.GBA = self._GXY(self.QBB, self.QBA)
        self.GBC = self._GXY(self.QBB, self.QBC)

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

        diagonal = self.Q.diagonal() # also np.diag(self.Q)
        if np.any(diagonal > 0):
            raise ValueError("Q matrix diagonal elements must be non-positive")
        
        #tmean = np.zeros_like(diagonal)
        #for i, d in enumerate(diagonal):
        #    if d < 0:
        #        tmean[i] = -1.0 / d
        #    else:
        #        tmean[i] = float('inf')  # Infinite lifetime for absorbing states

        return -1 / diagonal 

    def transition_probability(self):
        """ Calculate the transition probabilities. """
        transition_probability = self.Q.copy()
        np.fill_diagonal(transition_probability, 0)
        row_sums = -np.diag(self.Q)

        # Verify each row sums to approximately 1 (floating point tolerance)
        #row_sums = np.sum(transition_probability, axis=1)
        #for i, sum_i in enumerate(row_sums):
        #    if sum_i > 0 and not np.isclose(sum_i, 1.0, rtol=1e-5):
        #        warnings.warn(f"Row {i} in transition probability matrix sums to {sum_i}, not 1")

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


### TODO:Functions to review

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
