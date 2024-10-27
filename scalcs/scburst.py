import numpy as np
from numpy.linalg import inv, matrix_power
import matplotlib.pyplot as plt
#from matplotlib.transforms import FuncTransform

from samples import samples
from scalcs import qmatlib as qml
from scalcs import pdfs

class SCBurst(qml.QMatrix):
    """Calculates burst-related quantities from a Q-matrix as in Colquhoun & Hawkes (1982)."""

    def __init__(self, mec):
        super().__init__(mec.Q, mec.kA, mec.kB, mec.kC, mec.kD)
        self.GABAG = self.GAB @ self.GBA
        self.probability_of_ending = self.IA - self.GABAG
        self.invQAA, self.invQBB, self.invQFF = map(inv, [self.QAA, self.QBB, self.QFF])
        self.pinfC, self.pinfA = self.pinf[self.kE:self.kG], self.pinf[:self.kA]
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
        """
        return (self.start_burst @ inv(self.probability_of_ending) @ self.uA)[0]


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
        w = (self.start_burst @ A @ self.end_burst).flatten()
        return rho, w

    def openings_distr(self, r):
        """
        The distribution of openings per burst (Eq. 3.5, CH82).
        P(r) = phiB * (GAB * GBA)^(r-1) * eB

        Parameters
        ----------
        r : int
            Number of openings per burst.
        """

        if r == 1:
            interm = self.IA
        else:
            interm = matrix_power(self.GABAG, r - 1)
        return self.start_burst @ interm @ self.end_burst

    def openings_cond_distr_depend_on_start_state(self, r):
        """
        The distribution of openings per burst coditional on starting state.

        Parameters
        ----------
        r : int
            Number of openings per burst.
        """
        if r == 1:
            interm = self.IA
        else:
            interm = matrix_power(self.GABAG, r - 1)
        return (interm @ self.end_burst).transpose()

    @property
    def mean_length(self):
        """
        Calculate the mean burst length (Eq. 3.19, CH82).
        m = PhiB * (I - GAB * GBA)^(-1) * (-QAA^(-1)) * \
            (I - QAB * (QBB^(-1)) * GBA) * uA
        """

        inv_ending = inv(self.probability_of_ending)
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
        w = np.array([(self.start_burst @ A[i][:self.kA, :self.kA] @ (-self.QAA) @ self.end_burst)[0] for i in range(self.kE)])
        #for i in range(self.kE):
        #    w[i] = (self.start_burst @ A[i][:self.kA, :self.kA] @ (-self.QAA) @ self.end_burst)[0]
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
        """

        expQEEA = qml.expQ(self.QEE, t)[ : self.kA, : self.kA]
        return self.start_burst @ expQEEA @ -self.QAA @ self.end_burst

    def length_cond_pdf(self, t):
        """
        The distribution of burst length coditional on starting state.

        Parameters
        ----------
        t : float
            Burst length.
        """

        expQEEA = qml.expQ(self.QEE, t)[ : self.kA, : self.kA]
        return (expQEEA @ -self.QAA @ self.end_burst).transpose()

    @property
    def mean_open_time(self):
        """
        Calculate the mean total open time per burst (Eq. 3.26, CH82).
        """

        VAA = self.QAA + self.QAB @ self.GBA
        return (self.start_burst @ -inv(VAA) @ self.uA)[0]
    
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
        """ Calculate the mean total shut time per burst (Eq. 3.41, CH82) for all bursts (including one opening bursts). """
        return (self.start_burst @ self.GAB @ -inv(self.WBB) @ self.GBA @ self.uA)[0]
    
    @property
    def mean_shut_times_between_bursts(self):
        """ Calculate the mean length of the gap between bursts (Eq. 3.86, CH82). """

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
        w = -np.array([ (self.start_burst @ inv(self.probability_of_ending) @ self.GAB @ A[i] @ self.QBB @ self.GBA @ self.uA)[0] 
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
        """ Calculate the burst open probability. """
        return self.mean_open_time / self.mean_length

    @property
    def total_popen(self):
        """ Calculate the total open probability. """
        return self.mean_open_time / (self.mean_length + self.mean_shut_times_between_bursts)
   

class BurstDisplay(SCBurst):
    """ Print Q-Matrix calculations for single-channel burst analysis. """

    @property
    def print_all(self):
        """ Output burst calculations. """
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
            pdfs.ExpPDF(1 / e1, w1 / e1).printout('\nPDF of total burst length, unconditional'),
            pdfs.ExpPDF(1 / e2, w2 / e2).printout('\nPDF of burst length for bursts with 2 or more openings')
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
            pdfs.ExpPDF(1 / e1, w1 / e1).printout('\nPDF of total open time per burst'),
            pdfs.ExpPDF(1 / e2, w2 / e2).printout('\nPDF of first opening in a burst with 2 or more openings'),
            pdfs.GeometricPDF(rho, w).printout('\nGeometric PDF of number (r) of openings per burst (unconditional)')
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
            pdfs.ExpPDF(1 / e1, w1 / e1).printout('\nPDF of gaps inside burst'),
            pdfs.ExpPDF(1 / e2, w2 / e2).printout('\nPDF of gaps between bursts'),
            pdfs.ExpPDF(1 / e3, w3 / e3).printout('\nPDF of total shut time per burst for bursts with at least 2 openings')
        ]
        return ''.join(sections)
    
    def calculate_burst_length_pdf(self, multicomp=False, conditional=False, tmin=0.00001, points=512):
        """
        Calculate the points for burst length pdf.

        Parameters
        ----------
        multicomp : bool, optional
            True if multiple components are considered.
        conditional : bool, optional
            True if conditional distribution is calculated.
        tmin : float, optional
            Minimum time in seconds for the time range.
        points : int, optional
            Number of points in the plot.

        Returns
        -------
        t : ndarray
            Time in seconds.
        fbst : ndarray
            Burst length probability distribution function (PDF).
        Optional:
        mfbst : ndarray
            Burst length PDF for multiple components.
        cfbrst : ndarray
            Conditional burst length PDF.
        """
        eigs, w = self.length_pdf_components()
        t = np.logspace(np.log10(tmin), np.log10(20 / min(eigs)), points)
        fbst = t * pdfs.ExpPDF(1 / eigs, w / eigs).calculate(t)

        if multicomp:
            for i in range(self.kE):
                fbst = np.vstack((fbst, t * pdfs.ExpPDF(1 / eigs[i], w[i] / eigs[i]).calculate(t)))

        if conditional:
            cfbst = np.zeros((points, self.kA))
            for i in range(points):
                cfbst[i] = t[i] * self.length_cond_pdf(t[i])
            cfbrst = cfbst.T  # Transpose for correct shape
            return t, fbst, cfbrst

        return t, fbst

    def calculate_burst_openings_pdf(self, n, conditional=False):
        """
        Calculate the distribution of openings per burst.

        Parameters
        ----------
        n  : int
            Number of openings.
        conditional : bool
            True if conditional distribution is plotted.

        Returns
        -------
        r : ndarray of ints, shape (num of points,)
            Number of openings per burst.
        Pr : ndarray of floats, shape (num of points,)
            Fraction of bursts.
        cPr : ndarray of floats, shape (num of open states, num of points)
            Fraction of bursts for conditional distribution.
        """
        
        r = np.arange(1, n + 1)  # Opening counts from 1 to n
        Pr = np.array([self.openings_distr(openings) for openings in r])  # Unconditional probabilities

        if conditional:
            cPr = np.zeros((n, self.kA))
            for openings in r:
                cPr[openings-1] = self.openings_cond_distr_depend_on_start_state(openings)
            return r, Pr, cPr.transpose()

        return r, Pr

    def calculate_burst_length_versus_conc_plot(self, mec, cmin, cmax):
        """
        Calculate data for the plot of burst length versus concentration.

        Parameters
        ----------
        mec : instance of type Mechanism
        cmin, cmax : float
            Range of concentrations in M.

        Returns
        -------
        c : ndarray of floats, shape (num of points,)
            Concentration in Moles
        br : ndarray of floats, shape (num of points,)
            Mean burst length in seconds.
        brblk : ndarray of floats, shape (num of points,)
            Mean burst length in seconds corrected for fast block.
        """

        points = 100
        c = np.linspace(cmin, cmax, points)
        br = np.zeros(points)
        brblk = np.zeros(points)

        for i in range(points):
            mec.set_eff('c', c[i])
            super().__init__(mec)
            br[i] = self.mean_length
            if mec.fastblock:
                brblk[i] = br[i] * (1 + c[i] / mec.KBlk)
            else:
                brblk[i] = br[i]
        c = c # x axis scale in mikroMoles
        br = br
        brblk= brblk

        return c, br, brblk

    def plot_burst_length_vs_concentration(self, mec, cmin=10e-9, cmax=1e-3):
        """
        Plot mean burst length versus concentration.

        Parameters
        ----------
        mec : instance of Mechanism
            Mechanism instance with required methods to adjust concentration.
        cmin : float, optional
            Minimum concentration in M (default is 10e-9 M).
        cmax : float, optional
            Maximum concentration in M (default is 1e-3 M).
        """
        # Calculate burst length data over the concentration range
        c, br, brblk = self.calculate_burst_length_versus_conc_plot(mec, cmin, cmax)

        # Plot the results
        plt.figure(figsize=(8, 6))
        plt.plot(c * 1e6, br * 1000, 'r-', label="Mean Burst Length")
        plt.xlabel("Concentration (µM)")
        plt.ylabel("Mean Burst Length (ms)")
        plt.title("Mean Burst Length vs. Concentration")
        plt.legend()
        plt.show()


    def plot_burst_length_pdf(self, multicomp=True):
        """Generate and display the burst length PDF plot with multicomp=True."""
        
        # Retrieve data with multicomp enabled
        t, fbst = self.calculate_burst_length_pdf(multicomp=multicomp)
        
        # Create the plot
        fig, ax = plt.subplots()
        
        # Plot the main burst length PDF
        ax.semilogx(t * 1000, fbst[0], 'b-', label='Burst Length PDF')
        
        # Plot additional components if available
        if fbst.shape[0] > 1: #is not None:
            for i, mf in enumerate(fbst[1:]):
                ax.semilogx(t * 1000, mf, 'b--', label=f'Component {i+1}')

        # Apply square-root transformation to the y-axis
        #sqrt_transform = FuncTransform(lambda y: np.sqrt(y), lambda y: y**2)
        #ax.set_yscale('function', functions=(sqrt_transform.transform, sqrt_transform.inverted))

        # Labeling
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('PDF') # (sqrt scale)')
        ax.set_title('Burst Length PDF')
        ax.legend()

        plt.show()
        #return fig

    def plot_conditional_burst_length_pdf(self):
        """Generate and display the conditional burst length PDF plot."""
        
        # Retrieve data with conditional=True
        t, fbst, cfbst = self.calculate_burst_length_pdf(conditional=True)
        
        # Define color scheme for each starting state
        colors = ['g', 'r', 'c', 'm', 'y', 'k']
        plots = []
        
        # Plot each conditional burst length distribution by starting state
        for i in range(self.kA):
            color = colors[i % len(colors)]
            handle, = plt.semilogx(t * 1000, cfbst[i], color + '-', label=f"State {i+1}")
            plots.append(handle)
        
        # Plot the non-conditional burst length distribution
        handle, = plt.semilogx(t * 1000, fbst, 'b-', label="Not conditional")
        plots.append(handle)

        # Add legend and labels
        plt.xlabel('Time (ms)')
        plt.ylabel('PDF')
        plt.title('Conditional Burst Length PDF by Starting State')
        plt.legend(handles=plots)

        plt.show()

    def plot_openings_per_burst(self, n=10, conditional=False, colors=None):
        """
        Plot the distribution of the number of openings per burst.
        
        Parameters
        ----------
        n : int, optional
            Maximum number of openings to consider for the plot. Default is 10.
        conditional : bool, optional
            If True, plot conditional distributions based on starting state.
        colors : list of str, optional
            List of color codes for each starting state if conditional is True.
            Default color is assigned if not provided.
        """
        # Calculate the number of openings distribution
        if conditional:
            r, Pr, cPr = self.calculate_burst_openings_pdf(n, conditional=True)
        else:
            r, Pr = self.calculate_burst_openings_pdf(n)
        
        # Set default colors if not provided
        if colors is None:
            colors = ["r", "g", "b", "m", "c", "y"] * ((self.kA + 5) // 6)

        # Plot the conditional or unconditional distributions
        plots = []
        
        if conditional:
            
            for i in range(len(cPr)):
                color = colors[i % len(colors)]
                handle, = plt.plot(r, cPr[i], color +'o', label=f"State {i+1}")
                plots.append(handle)
        
        # Plot the overall distribution
        handle, = plt.plot(r, Pr, 'ko', label="Not conditional")
        plots.append(handle)
        
        # Configure plot settings
        plt.xlim([0, n + 1])
        plt.xlabel('Number of Openings per Burst')
        plt.ylabel('Probability')
        plt.title('Distribution of Openings per Burst' + (' (Conditional on Start State)' if conditional else ''))
        plt.legend(handles=plots)
        
        plt.show()


if __name__ == '__main__':
    c = 0.0000001 # 0.1 uM
    mec = samples.CH82()
    mec.set_eff('c', c)
    disp = BurstDisplay(mec)
    print(disp.print_all)

    disp.plot_burst_length_pdf()
    disp.plot_conditional_burst_length_pdf()
    disp.plot_openings_per_burst(n=10, conditional=True)
    disp.plot_burst_length_vs_concentration(mec, cmin=10e-9, cmax=1e-3)

