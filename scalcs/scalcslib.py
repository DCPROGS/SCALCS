import math
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from samples import samples
from scalcs import qmatlib as qml
from scalcs import pdfs
from scalcs import hjclib as hjc


class AsymptoticPDF(hjc.AsymptoticPDFCalculator):
    '''
    Class to calculate dwell-time distributions (open and shut times) using
    HJC models from the Q matrix.
    '''

    def __init__(self, mec, tres=0.0): 
        super().__init__(mec, tres=tres)
        
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
        super().__init__(mec)

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
        return pdfs.ExpPDF(1/e, w/e).printout('\nIdeal open time PDF components, unconditional')
   
    @property
    def print_ideal_shut_time_pdf(self):
        e, w = self.ideal_shut_time_pdf_components()
        return pdfs.ExpPDF(1/e, w/e).printout('\nIdeal shut time PDF components, unconditional')


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
        pdf_str = pdfs.ExpPDF(1 / e, a).printout_asymptotic(self.tres, '\nASYMPTOTIC OPEN TIME DISTRIBUTION')
        pdf_str += f'\nApparent mean open time (ms): {self.apparent_mean_open_time * 1000:.5g}\n'
        return pdf_str

    @property
    def print_asymptotic_shut_time_pdf(self):
        """ Print the asymptotic shut time PDF components. """
        e, a = self.HJC_asymptotic_shut_time_pdf_components()
        pdf_str = pdfs.ExpPDF(1 / e, a).printout_asymptotic(self.tres, '\nASYMPTOTIC SHUT TIME DISTRIBUTION')
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
        qml.QMatrix.__init__(self, mec)
        e, w = self.ideal_shut_time_pdf_components()
        self.tcrits = pdfs.TCrits(1 / e, w / e)

    @property
    def print_all(self):
        return self.tcrits.print_critical_times_summary()


class DwellsPDFDisplay:
    """ 
    Class to calculate and plot dwell time Probability Density Functions (PDFs).
    """
    def __init__(self, mec, tres: float = 0.0):
        """ 
        Parameters
        ----------
        mec : channel activation mechanism
            Activation mechanism Markov chain type.
        tres : float, optional
            Time resolution (dead time). Default is 0.0.       
        """
        self.exact = ExactPDFPrints(mec, tres)
        self.asymptotic = AsymptoticPDFPrints(mec, tres)
        self.ideal = QMatrixPrints(mec)
        self.tcrits = TCritPrints(mec)
        self.mec = mec
        self.tres = tres

    def _calculate_pdf(self, is_open: bool, tmin: float = 0.00001, 
                        tmax: Optional[float] = None, points: int = 512) -> Tuple[np.ndarray, ...]:
        """
        Generic method to calculate PDF distributions.

        Parameters
        ----------
        is_open : bool
            Whether to calculate open (True) or shut (False) time distribution.
        tmin : float, optional
            Minimum time for distribution. Default is 0.00001.
        tmax : float, optional
            Maximum time for distribution. If None, calculated from eigenvalues.
        points : int, optional
            Number of points in distribution. Default is 512.

        Returns
        -------
        Tuple of numpy arrays: (time, ideal PDF, exact PDF, asymptotic PDF)
        """
        # Select appropriate methods based on open/shut time
        if is_open:
            ideal_components = self.ideal.ideal_open_time_pdf_components
            asymp_components = self.asymptotic.HJC_asymptotic_open_time_pdf_components
            exact_components = self.exact.exact_GAMAxx_open = lambda: self.exact.exact_GAMAxx(open=True)
        else:
            ideal_components = self.ideal.ideal_shut_time_pdf_components
            asymp_components = self.asymptotic.HJC_asymptotic_shut_time_pdf_components
            exact_components = self.exact.exact_GAMAxx_shut = lambda: self.exact.exact_GAMAxx(open=False)

        # Ideal PDF
        e, w = ideal_components()
        
        # Determine time range
        tmax = tmax or (1 / e.min()) * 20
        t = np.logspace(math.log10(tmin), math.log10(tmax), points)
        
        # Scale factor
        fac = 1 / np.sum((w / e) * np.exp(-self.tres * e))
        ipdf = t * pdfs.ExpPDF(1 / e, w / e).calculate(t) * fac
        
        # Asymptotic PDF
        roots, areas = asymp_components()
        apdf = self._asymptotic_pdf(t, 1 / roots, areas)
        
        # Exact PDF
        eigs, gamma00, gamma10, gamma11 = exact_components()
        epdf = np.array([
            t[i] * self._exact_pdf(t[i], roots, areas, eigs, gamma00, gamma10, gamma11)
            for i in range(points)
        ])
                
        return t, ipdf, epdf, apdf

    def calculate_open_time_pdf(self, **kwargs):
        """Calculate open time PDF distribution."""
        return self._calculate_pdf(is_open=True, **kwargs)

    def calculate_shut_time_pdf(self, **kwargs):
        """Calculate shut time PDF distribution."""
        return self._calculate_pdf(is_open=False, **kwargs)

    def _asymptotic_pdf(self, t: np.ndarray, tau: np.ndarray, area: np.ndarray) -> np.ndarray:
        """
        Calculate asymptotic probability density function.

        Parameters
        ----------
        t : ndarray
            Time values
        tau : ndarray
            Time constants
        area : ndarray
            Component relative areas

        Returns
        -------
        ndarray
            Asymptotic PDF values
        """
        t1 = np.extract(t[:] < self.tres, t)
        t2 = np.extract(t[:] >= self.tres, t)
        apdf2 = t2 * pdfs.ExpPDF(tau, area).calculate(t2 - self.tres)
        return np.append(t1 * 0.0, apdf2)

    def _exact_pdf(self, t: float, roots: np.ndarray, areas: np.ndarray, 
                   eigvals: np.ndarray, gamma00: np.ndarray, 
                   gamma10: np.ndarray, gamma11: np.ndarray) -> float:
        """
        Calculate exact probability density function with missed events correction.

        Parameters
        ----------
        t : float
            Time point
        roots : ndarray
            Roots of the PDF
        areas : ndarray
            Component areas
        eigvals : ndarray
            Eigenvalues
        gamma00, gamma10, gamma11 : ndarray
            PDF coefficients

        Returns
        -------
        float
            PDF value at time t
        """
        if t < self.tres:
            return 0
        elif self.tres <= t < (2 * self.tres):
            return hjc.f0((t - self.tres), eigvals, gamma00)
        elif (2 * self.tres) <= t < (3 * self.tres):
            return (hjc.f0((t - self.tres), eigvals, gamma00) -
                    hjc.f1((t - 2 * self.tres), eigvals, gamma10, gamma11))
        else:
            return pdfs.ExpPDF(1 / roots, areas).calculate(t - self.tres)

    def _plot_pdf(self, t: np.ndarray, ipdf: np.ndarray, epdf: np.ndarray, apdf: np.ndarray, 
                  xlabel: str = "Time (ms)", ylabel: str = "PDF", title: str = "PDF Plot"):
        """
        Plot Probability Density Function with multiple components.

        Parameters
        ----------
        t : ndarray
            Time values
        ipdf, epdf, apdf : ndarray
            Ideal, exact, and asymptotic PDF values
        xlabel, ylabel : str, optional
            Axis labels
        title : str, optional
            Plot title
        """
        plt.figure(figsize=(6, 4))
        plt.semilogx(t * 1000, ipdf, 'r--', label="Ideal PDF", linewidth=2)
        plt.semilogx(t * 1000, apdf, 'g-', label="Asymptotic PDF", linewidth=2)
        plt.semilogx(t * 1000, epdf, 'b-', label="Exact PDF", linewidth=2)
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.show()

    def plot_open_time_pdf(self, **kwargs):
        """Plot Open Time Probability Density Function."""
        t, ipdf, epdf, apdf = self.calculate_open_time_pdf(**kwargs)
        self._plot_pdf(t, ipdf, epdf, apdf, title="Open Time PDF")

    def plot_shut_time_pdf(self, **kwargs):
        """Plot Shut Time Probability Density Function."""
        t, ipdf, epdf, apdf = self.calculate_shut_time_pdf(**kwargs)
        self._plot_pdf(t, ipdf, epdf, apdf, title="Shut Time PDF")

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



