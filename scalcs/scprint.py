from tabulate import tabulate

from scalcs.qmatlib import QMatrix
from scalcs.scalcslib import ExactPDFCalculator, AsymptoticPDF, AdjacentPDF
from scalcs.pdfs import TCrits, ExpPDF

class QMatrixPrints(QMatrix):
    """
    Provides printable representations of Q-matrix properties and related calculations, including equilibrium occupancies, transition matrices,
    and PDF components for open and shut times.
    """

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0):
        # Initialize the QMatrix superclass.
        super().__init__(Q, kA=kA, kB=kB, kC=kC, kD=kD)

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





class TCritPrints(QMatrix):
    def __init__(self, mec):
        QMatrix.__init__(self, mec.Q, kA=mec.kA, kB=mec.kB, kC=mec.kC, kD=mec.kD)
        e, w = self.ideal_shut_time_pdf_components()
        self.tcrits = TCrits(1 / e, w / e)

    @property
    def print_all(self):
        return self.tcrits.print_critical_times_summary()

class AsymptoticPDFPrints(AsymptoticPDF):
    """ 
    Class to print asymptotic PDF components for open and shut times.
    Inherits from AsymptoticPDF.
    """

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0, tres=0.0):
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
        super().__init__(Q, kA=kA, kB=kB, kC=kC, kD=kD, tres=tres)

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

class AdjacentPDFPrints(AdjacentPDF):
    """ Prints adjacent PDF. """
    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0, tres=0.0):
        super().__init__(Q, kA=kA, kB=kB, kC=kC, kD=kD, tres=tres)

    def ideal_adjacent_dwells(self, t1, t2):
        
        adjacent_str = ('\nPDF of open times that precede shut times between {0:.3f} and {1:.3f} ms'.
                         format(t1 * 1000, t2 * 1000))
        e, a = self.adjacent_open_to_shut_range_pdf_components(t1, t2)
        adjacent_str += ExpPDF(1 / e, a / e).printout('\nOPEN TIMES ADJACENT TO SPECIFIED SHUT TIME RANGE')
        #adjacent_str += expPDF_printout(e, a, 'OPEN TIMES ADJACENT TO SPECIFIED SHUT TIME RANGE')
        mean = self.adjacent_open_to_shut_range_mean(t1, t2) #     mec.QAA, mec.QAF, mec.QFF, mec.QFA, phiA)
        adjacent_str += ('Mean from direct calculation (ms) = {0:.6f}\n'.format(mean * 1000))
        return adjacent_str

class ExactPDFPrints(ExactPDFCalculator):
    """ 
    Print exact PDF coefficients for open and shut times.
    
    This class inherits from ExactPDFCalculator and adds functionality to print 
    the exact probability density function (PDF) coefficients for both open and shut times.
    """

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0, tres=0.0):
        """
        Initialize the ExactPDFPrints class.

        Parameters
        ----------
        Q : ndarray
            The Q matrix representing the transition rates.
        kA, kB, kC, kD : int, optional
            Dimensions of different state subspaces. Defaults are 1 for kA and kB, 0 for kC and kD.
        """
        super().__init__(Q, kA=kA, kB=kB, kC=kC, kD=kD, tres=tres)

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


