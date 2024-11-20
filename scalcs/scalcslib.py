import sys
import math
#from decimal import*
from deprecated import deprecated
from tabulate import tabulate
import scipy.optimize as so
import numpy as np
from numpy import linalg as nplin

from pylab import figure, semilogx, savefig

from samples import samples
from scalcs import qmatlib as qml
from scalcs import pdfs
from scalcs.qmatlib import HJCMatrix, AsymptoticPDFCalculator
from scalcs.qmatlib import QMatrix
from scalcs.pdfs import TCrits, ExpPDF


class AsymptoticPDF(AsymptoticPDFCalculator):
    '''
    Class to calculate dwell-time distributions (open and shut times) using
    HJC models from the Q matrix.
    '''

    def __init__(self, mec, tres=0.0): 
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


############################   PRINT GENERATORS PDF   #######################################

class QMatrixPrints(QMatrix):
    """
    Provides printable representations of Q-matrix properties and related calculations, including equilibrium occupancies, transition matrices,
    and PDF components for open and shut times.
    """

    def __init__(self, mec):
        # Initialize the QMatrix superclass.
        super().__init__(mec.Q, kA=mec.kA, kB=mec.kB, kC=mec.kC, kD=mec.kD)

    @property
    def print_Q(self):
        """Formatted Q-matrix with appropriate headers."""
        return '\nQ (units [1/s]) = \n' + tabulate(
            [['{:.3f}'.format(item) for item in row] for row in self.Q],
            headers=[i + 1 for i in range(self.k)],
            showindex=[i + 1 for i in range(self.k)],
            tablefmt='orgtbl'
        )    

    @property
    def print_pinf(self):
        """Formatted equilibrium state occupancies."""
        return '\nEquilibrium state occupancies:\n' + tabulate([self.pinf], 
            headers=[i + 1 for i in range(self.k)], tablefmt='orgtbl')

    @property
    def print_Popen(self):
        """Formatted equilibrium open probability."""
        return f'\nEquilibrium open probability Popen = {self.Popen():.6f}'

    @property
    def print_state_lifetimes(self):
        """Formatted mean lifetimes of each individual state in milliseconds."""
        return '\nMean lifetimes of each individual state (ms):\n' + tabulate(
            [[1000 / lifetime for lifetime in self.state_lifetimes()]],
            headers=[i + 1 for i in range(self.k)], 
            tablefmt='orgtbl')

    @property
    def print_transition_matrices(self):
        """Formatted transition probabilities and frequencies."""
        pi = self.transition_probability()
        fi = self.transition_frequency()

        prob_str = "\n\nProbability of transitions regardless of time:\n"
        prob_str += tabulate(
            ([f'{item:.4f}' for item in row] for row in pi),
            headers=[i+1 for i in range(self.k)],
            showindex=[i+1 for i in range(self.k)],
            tablefmt='orgtbl'
        )

        freq_str = "\n\nFrequency of transitions (per second):\n"
        freq_str += tabulate(
            ([f'{item:.4f}' for item in row] for row in fi),
            headers=[i+1 for i in range(self.k)],
            showindex=[i+1 for i in range(self.k)],
            tablefmt='orgtbl'
        )
        return prob_str + freq_str

    @property
    def print_subset_probabilities(self):
        """Formatted initial probabilities and conditional probabilities for state subsets."""
        return (
            f'\nInitial probabilities of finding channel in classes of states:\n'
            f'A states: P1(A) = {self.P("A"):.4g}\n'
            f'B states: P1(B) = {self.P("B"):.4g}\n'
            f'C states: P1(C) = {self.P("C"):.4g}\n'
            f'F states: P1(F) = {self.P("F"):.4g}\n\n'
            'Conditional probabilities:\n'
            f'P1(B|F) = {self.P("B|F"):.4g}\t\t'
            f'Probability of being in B if shut (in F) at t=0\n'
            f'P1(C|F) = {self.P("C|F"):.4g}\t\t'
            f'Probability of being in C if shut (in F) at t=0'
        )

    @property
    def print_initial_vectors(self):
        """Formatted initial vectors for state subsets."""
        return ('\nInitial vectors (conditional probability distribution over a subset of states):\n'
            f'phi(A) = {self.phi("A")}\n'
            f'phi(B) = {self.phi("B")}\n'
            f'phi(F) = {self.phi("F")}')

    @property
    def print_DC_table(self):
        """ Print DC table with open and shut state statistics, including mean lifetime and latency. """
        
        mean_latencies = [self.mean_latency_given_start_state(i) for i in range(1, self.k+1)]

        dc_str = '\n'
        header_open = ['Open \nstates', 'Equilibrium\n occupancy', 'Mean lifetime\n (ms)',
                       'Mean latency (ms)\nto next shutting\ngiven start \nin this state']
        DCtable_open = []
        mean_life_A = self.subset_mean_lifetime(1, self.kA)
        DCtable_open.append(['Subset A ', sum(self.pinf[:self.kA]), mean_life_A * 1000, ' '])
        for i in range(self.kA):
            DCtable_open.append([i+1, self.pinf[i], 1000*self.state_lifetimes()[i], 1000*mean_latencies[i]])

        dc_str += (tabulate(DCtable_open, headers=header_open, tablefmt='orgtbl', floatfmt=".6f") + '\n\n')

        header_shut = ['Shut\nstates', 'Equilibrium\n occupancy', 'Mean lifetime\n (ms)',
                       'Mean latency (ms)\nto next opening\ngiven start \nin this state']
        DCtable_shut = []
        for i in range(self.kA, self.k):
            if i == self.kA:
                mean_life_B = self.subset_mean_lifetime(self.kA+1, self.kE)
                DCtable_shut.append(['Subset B ', sum(self.pinf[self.kA: self.kE]), mean_life_B * 1000, '-'])
            if i == self.kE:
                mean_life_C = self.subset_mean_lifetime(self.kE+1, self.kG)
                DCtable_shut.append(['Subset C ', sum(self.pinf[self.kE: self.kG]), mean_life_C * 1000, '-'])
            if i == self.kG:
                mean_life_D = self.subset_mean_lifetime(self.kG+1, self.k)
                DCtable_shut.append(['Subset D ', sum(self.pinf[self.kG: self.k]), mean_life_D * 1000, '-'])
            DCtable_shut.append([i+1, self.pinf[i], 1000*self.state_lifetimes()[i], 1000*mean_latencies[i]])
        dc_str += tabulate(DCtable_shut, headers=header_shut, tablefmt='orgtbl', floatfmt=(None, '.6f', '.6g', '.6g',))
        return dc_str
    
    @property
    def print_initial_vectors_for_openings_shuttings(self):
        open_str = ('\nInitial vector for ideal openings =\n')
        for i in range(self.kA):
            open_str += ('\t{0:.5g}'.format(self.phiA[i]))
        shut_str = ('\nInitial vector for ideal shuttings =\n')
        for i in range(self.kF):
            shut_str += ('\t{0:.5g}'.format(self.phiF[i]))
        return open_str + shut_str + '\n'
    
    @property
    def print_open_time_pdf(self):
        e, w = self.ideal_open_time_pdf_components()
        return ExpPDF(1/e, w/e).printout('\nIdeal open time PDF components, unconditional')
   
    @property
    def print_shut_time_pdf(self):
        e, w = self.ideal_shut_time_pdf_components()
        return ExpPDF(1/e, w/e).printout('\nIdeal shut time PDF components, unconditional')


class AsymptoticPDFPrints(AsymptoticPDF):
    """ 
    Class to print asymptotic PDF components for open and shut times.
    Inherits from AsymptoticPDF.
    """

    def __init__(self, mec, tres=0.0):
        """
        Initialize the AsymptoticPDFPrints class.

        Parameters
        ----------
        Q : ndarray
            Transition rate matrix.
        kA, kB, kC, kD : int, optional
            Dimensions of different state subspaces. Defaults are 1 for kA and kB, 0 for kC and kD.
        tres : float, optional
            Resolution time. Default is 0.0.
        """
        super().__init__(mec, tres=tres) #Q, kA=kA, kB=kB, kC=kC, kD=kD, tres=tres)

    @property
    def print_all(self):
        """ Print the results of the asymptotic PDF calculations including open and shut times. """
        return (
            '\n*******************************************\n'
            'CALCULATED SINGLE CHANNEL ASYMPTOTIC DWELL MEANS, PDFS, ETC....\n'
            f'{self.print_initial_HJC_vectors}'
            f'{self.print_asymptotic_open_time_pdf}'
            f'{self.print_asymptotic_shut_time_pdf}'
        )

    @property
    def print_initial_HJC_vectors(self):
        """ Print the initial HJC vectors for openings and shuttings.   """
        initial_str = f'\nInitial vector for HJC openings (tres={1e6 * self.tres:.0f} us):\n'
        initial_str += '\n'.join([f'\t{self.HJCphiA[i]:.5g}' for i in range(self.kA)])
        initial_str += '\nInitial vector for HJC shuttings:\n'
        initial_str += '\n'.join([f'\t{self.HJCphiF[i]:.5g}' for i in range(self.kF)])
        return initial_str + '\n'

    @property
    def print_asymptotic_open_time_pdf(self):
        """ Print the asymptotic open time PDF components.  """
        e, a = self.HJC_asymptotic_open_time_pdf_components()
        pdf_str = ExpPDF(1 / e, a).printout_asymptotic(self.tres, '\nASYMPTOTIC OPEN TIME DISTRIBUTION')
        pdf_str += f'\nApparent mean open time (ms): {self.apparent_mean_open_time * 1000:.5g}\n'
        return pdf_str

    @property
    def print_asymptotic_shut_time_pdf(self):
        """ Print the asymptotic shut time PDF components. """
        e, a = self.HJC_asymptotic_shut_time_pdf_components()
        pdf_str = ExpPDF(1 / e, a).printout_asymptotic(self.tres, '\nASYMPTOTIC SHUT TIME DISTRIBUTION')
        pdf_str += f'\nApparent mean shut time (ms): {self.apparent_mean_shut_time * 1000:.5g}\n'
        return pdf_str


class ExactPDFPrints(ExactPDFCalculator):
    """ 
    Print exact PDF coefficients for open and shut times.
    
    This class inherits from ExactPDFCalculator and adds functionality to print 
    the exact probability density function (PDF) coefficients for both open and shut times.
    """

    def __init__(self, mec, tres=0.0):
        """
        Initialize the ExactPDFPrints class.

        Parameters
        ----------
        Q : ndarray
            The Q matrix representing the transition rates.
        kA, kB, kC, kD : int, optional
            Dimensions of different state subspaces. Defaults are 1 for kA and kB, 0 for kC and kD.
        """
        super().__init__(mec, tres=tres) #.Q, kA=mec.kA, kB=mec.kB, kC=mec.kC, kD=mec.kD, tres=tres)

    @property
    def open_time_pdf(self):
        """
        Property to calculate and print the exact open time PDF coefficients.

        Returns
        -------
        str
            Formatted string containing the exact open time PDF coefficients and eigenvalues.
        """
        eigvals, gamma00, gamma10, gamma11 = self.exact_GAMAxx(open=True)
        return self._expPDF_exact_printout(eigvals, gamma00, gamma10, gamma11, 'EXACT OPEN TIME DISTRIBUTION')

    @property
    def shut_time_pdf(self):
        """
        Property to calculate and print the exact shut time PDF coefficients.

        Returns
        -------
        str
            Formatted string containing the exact shut time PDF coefficients and eigenvalues.
        """
        eigvals, gamma00, gamma10, gamma11 = self.exact_GAMAxx(open=False)
        return self._expPDF_exact_printout(eigvals, gamma00, gamma10, gamma11, 'EXACT SHUT TIME DISTRIBUTION')

    def _expPDF_exact_printout(self, eigs, gamma00, gamma10, gamma11, title):
        """
        Helper method to format and print the exact PDF coefficients.

        Parameters
        ----------
        eigs : ndarray
            Eigenvalues from the Q matrix.
        gamma00, gamma10, gamma11 : ndarrays
            PDF coefficients corresponding to the eigenvalues.
        title : str
            Title for the table output (e.g., 'EXACT OPEN TIME DISTRIBUTION').

        Returns
        -------
        str
            Formatted table of eigenvalues and PDF coefficients.
        """
        header = ['Eigenvalue', 'g00(m)', 'g10(m)', 'g11(m)']
        table = [[eigs[i], gamma00[i], gamma10[i], gamma11[i]] for i in range(len(eigs))]
        return f"\n{title}\n" + tabulate(table, headers=header, tablefmt='orgtbl')


class TCritPrints(QMatrix):
    def __init__(self, mec):
        QMatrix.__init__(self, mec.Q, kA=mec.kA, kB=mec.kB, kC=mec.kC, kD=mec.kD)
        e, w = self.ideal_shut_time_pdf_components()
        self.tcrits = TCrits(1 / e, w / e)

    @property
    def print_all(self):
        return self.tcrits.print_critical_times_summary()


############################   FUNCTIONS TO REVIEW   ########################################

def open_time_pdf(mec, tres, tmin=0.00001, tmax=1000, points=512, unit='ms'):
    """
    Calculate ideal asymptotic and exact open time distributions.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution.
    tmin, tmax : floats
        Time range for burst length ditribution.
    points : int
        Number of points per plot.
    unit : str
        'ms'- milliseconds.

    Returns
    -------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    ipdf, epdf, apdf : ndarrays of floats, shape (num of points)
        Ideal, exact and asymptotic open time distributions.
    """

    open = True

    # Asymptotic pdf
    roots = asymptotic_roots(tres,
        mec.QAA, mec.QII, mec.QAI, mec.QIA, mec.kA, mec.kI)

    tmax = (-1 / roots.max()) * 20
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    # Ideal pdf.
    eigs, w = ideal_dwell_time_pdf_components(mec.QAA, qml.phiA(mec))
    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
    ipdf = t * pdfs.ExpPDF(1 / eigs, w / eigs).calculate(t) * fac
    

    # Asymptotic pdf
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kI)
    areas = asymptotic_areas(tres, roots,
        mec.QAA, mec.QII, mec.QAI, mec.QIA,
        mec.kA, mec.kI, GAF, GFA)
    apdf = asymptotic_pdf(t, tres, -1 / roots, areas)

    # Exact pdf
    eigvals, gamma00, gamma10, gamma11 = exact_GAMAxx(mec,
        tres, open)
    epdf = np.zeros(points)
    for i in range(points):
        epdf[i] = (t[i] * exact_pdf(t[i], tres,
            roots, areas, eigvals, gamma00, gamma10, gamma11))
            
    if unit == 'ms':
        t = t * 1000 # x scale in millisec

    return t, ipdf, epdf, apdf


def scaled_pdf(t, pdf, dt, n):
    """
    Scale pdf to the data histogram.

    Parameters
    ----------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    pdf : ndarray of floats, shape (num of points)
        pdf to scale.
    dt : float
        Histogram bin width in log10 units.
    n : int
        Total number of events.

    Returns
    -------
    spdf : ndarray of floats, shape (num of points)
        Scaled pdf.
    """

    spdf = n * dt * 2.30259 * pdf
    #spdf = n * dt * pdf
    return spdf

def shut_time_pdf(mec, tres, tmin=0.00001, tmax=1000, points=512, unit='ms'):
    """
    Calculate ideal asymptotic and exact shut time distributions.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution.
    tmin, tmax : floats
        Time range for burst length ditribution.
    points : int
        Number of points per plot.
    unit : str
        'ms'- milliseconds.

    Returns
    -------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    ipdf, epdf, apdf : ndarrays of floats, shape (num of points)
        Ideal, exact and asymptotic shut time distributions.
    """

    open = False

    # Asymptotic pdf
    roots = asymptotic_roots(tres, mec.QII, mec.QAA, mec.QIA, mec.QAI,
        mec.kI, mec.kA)

    tmax = (-1 / roots.max()) * 20
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    # Ideal pdf.
    eigs, w = ideal_dwell_time_pdf_components(mec.QII, qml.phiF(mec))
    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
    ipdf = t * pdfs.ExpPDF(1 / eigs, w / eigs).calculate(t) *fac #pdfs.expPDF(t, 1 / eigs, w / eigs) * fac

    # Asymptotic pdf
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kI)
    areas = asymptotic_areas(tres, roots,
        mec.QII, mec.QAA, mec.QIA, mec.QAI,
        mec.kI, mec.kA, GFA, GAF)
    apdf = asymptotic_pdf(t, tres, -1 / roots, areas)

    # Exact pdf
    eigvals, gamma00, gamma10, gamma11 = exact_GAMAxx(mec, tres, open)
    epdf = np.zeros(points)
    for i in range(points):
        epdf[i] = (t[i] * exact_pdf(t[i], tres,
            roots, areas, eigvals, gamma00, gamma10, gamma11))

    if unit == 'ms':
        t = t * 1000 # x scale in millisec

    return t, ipdf, epdf, apdf

def subset_time_pdf(mec, tres, state1, state2,
    tmin=0.00001, tmax=1000, points=512, unit='ms'):
    """
    Calculate ideal pdf of any subset dwell times.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution.
    state1, state2 : ints
    tmin, tmax : floats
        Time range for burst length ditribution.
    points : int
        Number of points per plot.
    unit : str
        'ms'- milliseconds.

    Returns
    -------
    t : ndarray of floats, shape (num of points)
        Time in millisec.
    spdf : ndarray of floats, shape (num of points)
        Subset dwell time pdf.
    """

    open = False
    if open:
        eigs, w = ideal_dwell_time_pdf_components(mec.QAA, qml.phiA(mec))
    else:
        eigs, w = ideal_dwell_time_pdf_components(mec.QII, qml.phiF(mec))

    tau = 1 / eigs

    tmax = tau.max() * 20
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    # Ideal pdf.
    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
    #ipdf = t * pdfs.expPDF(t, 1 / eigs, w / eigs) * fac
    ipdf = t * pdfs.ExpPDF(1 / eigs, w / eigs).calculate(t) * fac

    spdf = np.zeros(points)
    for i in range(points):
        spdf[i] = t[i] * ideal_subset_time_pdf(mec.Q,
            state1, state2, t[i]) * fac

    if unit == 'ms':
        t = t * 1000 # x scale in millisec

    return t, ipdf, spdf

def png_save_pdf_fig(outfile, ints, mec, conc, tres, type):
    x, y, dx = prepare_hist(ints, tres)
    mec.set_eff('c', conc)
    if type == 'open':
        t, ipdf, epdf, apdf = open_time_pdf(mec, tres)
    elif type == 'shut':
        t, ipdf, epdf, apdf = shut_time_pdf(mec, tres)
    else:
        print ('Wrong type.')

    sipdf = scaled_pdf(t, ipdf, math.log10(dx), len(ints))
    sepdf = scaled_pdf(t, epdf, math.log10(dx), len(ints))
    figure(figsize=(6, 4))
    semilogx(x*1000, y, 'k-', t, sipdf, 'r--', t, sepdf, 'b-')
    savefig(outfile, bbox_inches=0)

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


if __name__ == '__main__':
    mec = samples.CH82()
    mec.set_eff('c', 0.0000001) 
    tres = 0.0001 # 10 us

    q_matrix = QMatrixPrints(mec)
    print(q_matrix.print_Q)
    print(q_matrix.print_pinf) # print equilibrium state occupancies
    print(q_matrix.print_Popen)
    print(q_matrix.print_state_lifetimes)
    print(q_matrix.print_transition_matrices)
    print(q_matrix.print_subset_probabilities)
    print(q_matrix.print_initial_vectors)
    print(q_matrix.print_DC_table)
    print(q_matrix.print_initial_vectors_for_openings_shuttings)
    print(q_matrix.print_open_time_pdf)
    print(q_matrix.print_shut_time_pdf)

    q_asymp = AsymptoticPDFPrints(mec, tres=tres)
    print(q_asymp.print_all)

    q_exact = ExactPDFPrints(mec, tres=tres)
    print(q_exact.open_time_pdf)
    print(q_exact.shut_time_pdf)

    tcrits = TCritPrints(mec)
    print(tcrits.print_all)
