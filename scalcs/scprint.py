from math import sqrt
from tabulate import tabulate

from scalcs.qmatlib import QMatrix
from scalcs.scburst import SCBurst
from scalcs.scalcslib import SCCorrelations, ExactPDFCalculator, AsymptoticPDF, AdjacentPDF
from scalcs.pdfs import TCrits, ExpPDF, GeometricPDF

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


class SCBurstPrints(SCBurst):
    """
    Print Q-Matrix calculations for single-channel burst analysis.
    """

    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0):
        super().__init__(Q, kA=kA, kB=kB, kC=kC, kD=kD)

    @property
    def print_all(self):
        """
        Output burst calculations.
        """
        sections = [
            '\n*******************************************',
            'CALCULATED SINGLE CHANNEL BURST MEANS, PDFS, ETC....',
            self.print_start_end_vectors,
            self.print_popen,
            self.print_length_pdf,
            self.print_openings_pdf,
            self.print_shuttings_pdf
        ]
        return '\n'.join(sections)

    @property
    def print_start_end_vectors(self):
        """
        Print the burst start and end vectors.
        """
        start_vector = '\nBurst start vector = ' + '\t'.join(f'{x:.5g}' for x in self.start_burst)
        end_vector = '\nBurst end vector =' + '\t'.join(f'{self.end_burst[i, 0]:.5g}' for i in range(self.kA))
        return start_vector + end_vector

    @property
    def print_means(self):
        """
        Print mean values related to burst behavior.
        """
        sections = [
            f'\n\nMean number of openings per burst = {self.mean_number_of_openings:.6g}',
            f'Mean burst length (ms) = {1000 * self.mean_length:.6g}',
            f'Mean open time per burst (ms) = {1000 * self.mean_open_time:.6g}',
            f'Mean shut time per burst (ms; all bursts) = {1000 * self.mean_shut_time:.6g}',
            f'Mean shut time per burst (ms; excluding single opening bursts) = {1000 * self.mean_shut_time / self.probability_more_than_one_opening:.6g}',
            f'Mean shut time between bursts (ms) = {1000 * self.mean_shut_times_between_bursts:.6g}'
        ]
        return '\n'.join(sections)

    @property
    def print_popen(self):
        """
        Print Popen (probability of being open) values.
        """
        sections = [
            f'\n\nPopen WITHIN BURST = (open time/burst)/(burst length) = {self.burst_popen:.5g}',
            f'Total Popen = (open time/burst)/(burst length + mean gap between bursts) = {self.total_popen:.5g}'
        ]
        return '\n'.join(sections)

    @property
    def print_length_pdf(self):
        """
        Print the PDF of burst lengths.
        """
        e1, w1 = self.length_pdf_components()
        e2, w2 = self.length_pdf_no_single_openings_components()
        sections = [
            ExpPDF(1 / e1, w1 / e1).printout('\nPDF of total burst length, unconditional'),
            ExpPDF(1 / e2, w2 / e2).printout('\nPDF of burst length for bursts with 2 or more openings')
        ]
        return ''.join(sections)

    @property
    def print_openings_pdf(self):
        """
        Print the PDF of openings within bursts.
        """
        e1, w1 = self.total_open_time_pdf_components()
        e2, w2 = self.first_opening_length_pdf_components()
        rho, w = self.openings_distr_components()
        sections = [
            ExpPDF(1 / e1, w1 / e1).printout('\nPDF of total open time per burst'),
            ExpPDF(1 / e2, w2 / e2).printout('\nPDF of first opening in a burst with 2 or more openings'),
            GeometricPDF(rho, w).printout('\nGeometric PDF of number (r) of openings per burst (unconditional)')
        ]
        return ''.join(sections)

    @property
    def print_shuttings_pdf(self):
        """
        Print the PDF of shuttings within bursts and between bursts.
        """
        e1, w1 = self.shut_times_inside_burst_pdf_components()
        e2, w2 = self.shut_times_between_burst_pdf_components()
        e3, w3 = self.shut_time_total_pdf_components_2more_openings()
        sections = [
            ExpPDF(1 / e1, w1 / e1).printout('\nPDF of gaps inside burst'),
            ExpPDF(1 / e2, w2 / e2).printout('\nPDF of gaps between bursts'),
            ExpPDF(1 / e3, w3 / e3).printout('\nPDF of total shut time per burst for bursts with at least 2 openings')
        ]
        return ''.join(sections)


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


class CorrelationPrints(SCCorrelations):
    """ Prints various correlation and Q-matrix calculations. """
    def __init__(self, Q, kA=1, kB=1, kC=0, kD=0):
        super().__init__(Q, kA=kA, kB=kB, kC=kC, kD=kD)
        self.varA, self.varF = self._variance(open=True), self._variance(open=False)

    @property
    def print_all(self):
        return (f"\n***** CORRELATIONS *****\n" +
                self.print_ranks +
                self.print_open_correlations +
                self.print_shut_correlations +
                self.print_open_shut_correlations)

    @property
    def print_ranks(self):
        """ Print ranks and eigenvalues of the matrices. """
        return (f"\n Ranks of GAF, GFA = {self.rank_GAF}, {self.rank_GFA}"
                f"\n Rank of GFA * GAF = {self.rank_XFF}"
                f"\n Rank of GAF * GFA = {self.rank_XAA}")

    def _format_correlation_info(self, var, var_n, n, correlation_limit, correlation_type):
        """ Helper method to format correlation information for open and shut times. """
        percentage_diff = 100 * (sqrt(var_n / (n * n)) - sqrt(var / n)) / sqrt(var / n)
        limiting_percentage = 100 * (sqrt(1 + 2 * correlation_limit / var) - 1)
        
        return (f"\nVariance of {correlation_type} time = {var:.5g}\n"
                f"SD of all {correlation_type} times = {sqrt(var) :.5g}\n"
                f"SD of means of {n} {correlation_type} times if uncorrelated = {sqrt(var / n) :.5g}\n"
                f"Actual SD of mean = {sqrt(var_n / (n * n)) :.5g}\n"
                f"Percentage difference as result of correlation = {percentage_diff:.5g}\n"
                f"Limiting value of percent difference for large n = {limiting_percentage:.5g}")

    def _format_correlation_coefficients(self, var, n, open=True):
        """
        Helper method to format correlation coefficients for open or shut times.
        """
        correlation_str = '\nCorrelation coefficients, r(k), for up to lag k = 5:'
        for i in range(n):
            cov = self._covariance(i + 1, open=open)
            corr_coeff = self._coefficient(cov, var, var)
            correlation_str += f"\nr({i+1}) = {corr_coeff:.5g}"
        return correlation_str

    @property
    def print_open_correlations(self):
        """ Print open-open time correlations. """

        varA_n = self.variance_n(50, open=True)
        correlation_limit_A = self.correlation_limit(open=True)

        open_str = '\n\n OPEN-OPEN TIME CORRELATIONS'
        open_str += self._format_correlation_info(self.varA, varA_n, 50, correlation_limit_A, 'open')
        open_str += self._format_correlation_coefficients(self.varA, 5, open=True)
        return open_str

    @property
    def print_shut_correlations(self):
        """ Print shut-shut time correlations. """

        varF_n = self.variance_n(50, open=False)
        correlation_limit_F = self.correlation_limit(open=False)

        shut_str = '\n\n SHUT-SHUT TIME CORRELATIONS'
        shut_str += self._format_correlation_info(self.varF, varF_n, 50, correlation_limit_F, 'shut')
        shut_str += self._format_correlation_coefficients(self.varF, 5, open=False)
        return shut_str

    @property
    def print_open_shut_correlations(self):
        """ Print open-shut time correlations. """
        open_shut_str = '\n\n OPEN - SHUT TIME CORRELATIONS'
        open_shut_str += '\nCorrelation coefficients, r(k), for up to lag k = 5:'
        
        for i in range(5):
            covAF = self.covariance_AF(i + 1)
            open_shut_str += f"\nr({i+1}) = {self._coefficient(covAF, self.varA, self.varF):.5g}"
        return open_shut_str


