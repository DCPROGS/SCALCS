import sys
import math
#from decimal import*
from deprecated import deprecated
from tabulate import tabulate
import scipy.optimize as so
import numpy as np
from numpy import linalg as nplin
import matplotlib.pyplot as plt

from pylab import figure, semilogx, savefig

from samples import samples
from scalcs import qmatlib as qml
from scalcs import pdfs
from scalcs import hjclib as hjc
from scalcs.pdfs import TCrits, ExpPDF


class AsymptoticPDF(hjc.AsymptoticPDFCalculator):
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


############################   PRINT GENERATORS PDF   #######################################

class QMatrixPrints(qml.QMatrix):
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
    def print_ideal_open_time_pdf(self):
        e, w = self.ideal_open_time_pdf_components()
        return ExpPDF(1/e, w/e).printout('\nIdeal open time PDF components, unconditional')
   
    @property
    def print_ideal_shut_time_pdf(self):
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


class ExactPDFPrints(hjc.ExactPDFCalculator):
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
        super().__init__(mec, tres=tres)

    @property
    def print_exact_open_time_pdf(self):
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
    def print_exact_shut_time_pdf(self):
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


class TCritPrints(qml.QMatrix):
    def __init__(self, mec):
        qml.QMatrix.__init__(self, mec.Q, kA=mec.kA, kB=mec.kB, kC=mec.kC, kD=mec.kD)
        e, w = self.ideal_shut_time_pdf_components()
        self.tcrits = TCrits(1 / e, w / e)

    @property
    def print_all(self):
        return self.tcrits.print_critical_times_summary()


class DwellsPDFDisplay:
    """ 
    Class to (print and) plot dwell time PDF's.
    """

    def __init__(self, mec, tres=0.0):
        """        """
        self.exact = ExactPDFPrints(mec, tres)
        self.asymptotic = AsymptoticPDFPrints(mec, tres)
        self.ideal = QMatrixPrints(mec)
        self.tcrits = TCritPrints(mec)
        self.mec = mec
        self.tres = tres

    def calculate_open_time_pdf(self, tmin=0.00001, tmax=1000, points=512):
        """
        Calculate ideal asymptotic and exact open time distributions.

        Parameters
        ----------
        tmin, tmax : floats
            Time range for burst length ditribution.
        points : int
            Number of points per plot.

        Returns
        -------
        t : ndarray of floats, shape (num of points)
            Time in seconds.
        ipdf, epdf, apdf : ndarrays of floats, shape (num of points)
            Ideal, exact and asymptotic open time distributions.
        """

        # Ideal pdf.
        eigs, w = self.ideal.ideal_open_time_pdf_components()

        tmax = (1 / eigs.min()) * 20
        t = np.logspace(math.log10(tmin), math.log10(tmax), points)
        
        fac = 1 / np.sum((w / eigs) * np.exp(-self.tres * eigs)) # Scale factor
        ipdf = t * pdfs.ExpPDF(1 / eigs, w / eigs).calculate(t) * fac
        
        # Asymptotic pdf
        Aroots, Aareas = self.asymptotic.HJC_asymptotic_open_time_pdf_components()
        apdf = self.asymptotic_pdf(t, 1 / Aroots, Aareas)

        # Exact pdf
        Aeigs, Agamma00, Agamma10, Agamma11 = self.exact.exact_GAMAxx(open=True)
        epdf = np.zeros(points)
        for i in range(points):
            epdf[i] = (t[i] * self.exact_pdf(t[i], Aroots, Aareas, Aeigs, Agamma00, Agamma10, Agamma11))
                
        return t, ipdf, epdf, apdf


    def calculate_shut_time_pdf(self, tmin=0.00001, tmax=1000, points=512):
        """
        Calculate ideal asymptotic and exact shut time distributions.

        Parameters
        ----------
        tmin, tmax : floats
            Time range for burst length ditribution.
        points : int
            Number of points per plot.

        Returns
        -------
        t : ndarray of floats, shape (num of points)
            Time in seconds.
        ipdf, epdf, apdf : ndarrays of floats, shape (num of points)
            Ideal, exact and asymptotic shut time distributions.
        """

        # Ideal pdf
        eigs, w = self.ideal.ideal_shut_time_pdf_components()

        tmax = (1 / eigs.min()) * 20
        t = np.logspace(math.log10(tmin), math.log10(tmax), points)

        fac = 1 / np.sum((w / eigs) * np.exp(-self.tres * eigs)) # Scale factor
        ipdf = t * pdfs.ExpPDF(1 / eigs, w / eigs).calculate(t) *fac

        # Asymptotic pdf
        Froots, Fareas = self.asymptotic.HJC_asymptotic_shut_time_pdf_components()
        apdf = self.asymptotic_pdf(t, 1 / Froots, Fareas)

        # Exact pdf
        Feigs, Fgamma00, Fgamma10, Fgamma11 = self.exact.exact_GAMAxx(open=False)
        epdf = np.zeros(points)
        for i in range(points):
            epdf[i] = (t[i] * self.exact_pdf(t[i], Froots, Fareas, Feigs, Fgamma00, Fgamma10, Fgamma11))

        return t, ipdf, epdf, apdf

    def asymptotic_pdf(self, t, tau, area):
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
        t1 = np.extract(t[:] < self.tres, t)
        t2 = np.extract(t[:] >= self.tres, t)
        apdf2 = t2 * pdfs.ExpPDF(tau, area).calculate(t2 - self.tres)
        return np.append(t1 * 0.0, apdf2)

    def exact_pdf(self, t, roots, areas, eigvals, gamma00, gamma10, gamma11):
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

        if t < self.tres:
            f = 0
        elif ((self.tres < t) and (t < (2 * self.tres))):
            f = hjc.f0((t - self.tres), eigvals, gamma00)
        elif ((self.tres * 2) < t) and (t < (3 * self.tres)):
            f = (hjc.f0((t - self.tres), eigvals, gamma00) -
                hjc.f1((t - 2 * self.tres), eigvals, gamma10, gamma11))
        else:
            f = pdfs.ExpPDF(1 / roots, areas).calculate(t - self.tres)
        return f


    def plot_open_time_pdf(self, xlabel="Time (ms)", ylabel="PDF", title="Open time PDF"):

        t, ipdf, epdf, apdf = self.calculate_open_time_pdf()
        fig, ax = plt.subplots()
        ax.semilogx(t * 1000, ipdf, 'r--', label="Ideal open time PDF")
        ax.semilogx(t * 1000, apdf, 'g-', label="Asymptotic open time PDF")
        ax.semilogx(t * 1000, epdf, 'b-', label="Exact open time PDF")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        plt.show()

    def plot_shut_time_pdf(self, xlabel="Time (ms)", ylabel="PDF", title="Shut time PDF"):
        t, ipdf, epdf, apdf = self.calculate_shut_time_pdf()
        fig, ax = plt.subplots()
        ax.semilogx(t * 1000, ipdf, 'r--', label="Ideal shut time PDF")
        ax.semilogx(t * 1000, apdf, 'g-', label="Asymptotic shut time PDF")
        ax.semilogx(t * 1000, epdf, 'b-', label="Exact shut time PDF")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        plt.show()


############################   FUNCTIONS TO REVIEW   ########################################

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

    #spdf = n * dt * pdf
    return n * dt * 2.30259 * pdf

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
        eigs, w = qml.ideal_dwell_time_pdf_components(mec.QAA, qml.phiA(mec))
    else:
        eigs, w = qml.ideal_dwell_time_pdf_components(mec.QII, qml.phiF(mec))

    tau = 1 / eigs

    tmax = tau.max() * 20
    t = np.logspace(math.log10(tmin), math.log10(tmax), points)

    # Ideal pdf.
    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
    ipdf = t * pdfs.ExpPDF(1 / eigs, w / eigs).calculate(t) * fac

    spdf = np.zeros(points)
    for i in range(points):
        spdf[i] = t[i] * qml.ideal_subset_time_pdf(mec.Q,
            state1, state2, t[i]) * fac

    if unit == 'ms':
        t = t * 1000 # x scale in millisec

    return t, ipdf, spdf


if __name__ == '__main__':
    mec = samples.CH82()
    mec.set_eff('c', 0.0000001) 
    tres = 0.0001 # 100 us
    
    dwells = DwellsPDFDisplay(mec, tres)

    print(dwells.ideal.print_Q)
    print(dwells.ideal.print_pinf) # print equilibrium state occupancies
    print(dwells.ideal.print_Popen)
    print(dwells.ideal.print_state_lifetimes)
    print(dwells.ideal.print_transition_matrices)
    print(dwells.ideal.print_subset_probabilities)
    print(dwells.ideal.print_initial_vectors)
    print(dwells.ideal.print_DC_table)
    print(dwells.ideal.print_initial_vectors_for_openings_shuttings)

    print(dwells.ideal.print_ideal_open_time_pdf)
    print(dwells.ideal.print_ideal_shut_time_pdf)
    print(dwells.asymptotic.print_asymptotic_open_time_pdf)
    print(dwells.asymptotic.print_asymptotic_shut_time_pdf)
    print(dwells.exact.print_exact_open_time_pdf)
    print(dwells.exact.print_exact_shut_time_pdf)
    
    print(dwells.tcrits.print_all)
    
    dwells.plot_open_time_pdf()
    dwells.plot_shut_time_pdf()



