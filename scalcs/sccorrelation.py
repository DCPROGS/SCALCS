import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from functools import cached_property

from samples import samples
from scalcs import qmatlib as qml

class CorrelationCalculator(qml.QMatrix):
    """
    SCCorrelations handles the computation of correlation coefficients, variances, and covariances
    using Q-matrices for open and shut time PDFs.
    """

    def __init__(self, mec):
        super().__init__(mec.Q, mec.kA, mec.kB, mec.kC, mec.kD)

        self.uA: np.ndarray = np.ones((self.kA, 1))
        self.uF: np.ndarray = np.ones((self.kF, 1))
        self.phiAr: np.ndarray = self.phiA.reshape(1, self.kA)
        self.phiFr: np.ndarray = self.phiF.reshape(1, self.kF)

        self.rank_GAF: int = np.linalg.matrix_rank(self.GAF)
        self.rank_GFA: int = np.linalg.matrix_rank(self.GFA)

        self.XFF: np.ndarray = self.GFA @ self.GAF
        self.rank_XFF: int = np.linalg.matrix_rank(self.XFF)
        self.eigs_XFF, self.A_XFF = qml.eigenvalues_and_spectral_matrices(self.XFF)

        self.XAA: np.ndarray = self.GAF @ self.GFA
        self.rank_XAA: int = np.linalg.matrix_rank(self.XAA)
        self.eigs_XAA, self.A_XAA = qml.eigenvalues_and_spectral_matrices(self.XAA)

        self.varianceA: float = self.variance(open=True)
        self.varianceF: float = self.variance(open=False)

    @cached_property
    def varianceA(self) -> float:
        """Cached variance for open times."""
        return self.variance(open=True)

    @cached_property
    def varianceF(self) -> float:
        """Cached variance for shut times."""
        return self.variance(open=False)

    @cached_property
    def eigs_XFF(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cached eigenvalues and spectral matrices for XFF."""
        return qml.eigenvalues_and_spectral_matrices(self.XFF)

    @cached_property
    def eigs_XAA(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cached eigenvalues and spectral matrices for XAA."""
        return qml.eigenvalues_and_spectral_matrices(self.XAA)

    def variance(self, open: bool = True) -> float:
        """Calculate variance for open (or shut) times."""
        u, phi, invQxx, _ = self._get_open_or_shut_params(open)
        M = 2 * np.eye(u.shape[0]) - u @ phi
        row = phi @ invQxx
        col = invQxx @ u
        return (row @ M @ col)[0, 0]


    def variance_n(self, n: int, open: bool = True) -> float:
        """
        Calculate the variance of the nth event for open (or shut) times.
        """
        variance = self.varianceA if open else self.varianceF
        total_covariance = sum((n - i) * self.correlation_coefficient(self.covariance(i + 1, open=open), variance, variance) * variance for i in range(1, n))
        return n * variance + 2 * total_covariance

    def correlation_coefficient(self, cov: float, var1: float, var2: float) -> float:
        """Calculate correlation coefficient given covariance and variances."""
        return cov / np.sqrt(var1 * var2)

    def covariance(self, lag: int, open: bool = True) -> float:
        """
        Vectorized covariance calculation for open (or shut) times using cached matrices.
        """
        if lag < 0:
            raise ValueError("Lag must be a non-negative integer.")
        u, phi, invQxx, Xn = self._get_open_or_shut_params(open, lag)
        M2 = Xn - u @ phi
        row = phi @ invQxx
        col = invQxx @ u
        return (row @ M2 @ col)[0, 0]

    def correlation_limit(self, open: bool = True) -> float:
        """
        Calculate the correlation limit for open (or shut) times.

        Parameters:
        open : bool (default=True)
            Set to True for open time correlation limit, False for shut time.

        Returns:
        float : The correlation limit.
        """
        u, phi, invQxx, eigs, A = self._get_eigen_and_matrices(open)
        row = phi @ invQxx
        col = invQxx @ u
        M = np.einsum('i,ijk->jk', eigs[:-1] / (1 - eigs[:-1]), A[:-1, :, :])
        return (row @ M @ col)[0, 0]

    def covariance_AF(self, lag: int) -> float:
        """Vectorized covariance between open and shut times."""
        if lag < 0:
            raise ValueError("Lag must be a non-negative integer.")

        uA, uF = self.uA, self.uF
        phiA = self.phiAr
        MAF = qml.powQ(self.XAA, lag - 1) - uA @ phiA

        row = phiA @ -np.linalg.inv(self.QAA)
        col = (self.GAF @ -np.linalg.inv(self.QFF)) @ uF
        return (row @ MAF @ col)[0, 0]

    def decay_amplitude_A(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate decay amplitude for open-open correlation decay."""
        eigs, A = self.eigs_XAA
        invQAA = -np.linalg.inv(self.QAA)
        row = self.phiAr @ invQAA
        col = invQAA @ self.uA

        valid_indices = (np.abs(eigs) > 1e-12) & (np.abs(eigs - 1) > 1e-12)
        weights = (row @ A[valid_indices, :, :] @ col)[:, 0] / self.varianceA

        return weights, eigs[valid_indices]
    
    def calculate_correlation_coefficients(self, lag: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized calculation of correlation coefficients for open, shut, and open-shut times."""
        r = np.arange(1, lag + 1)
        roA = np.array([self.correlation_coefficient(self.covariance(i + 1, open=True), self.varianceA, self.varianceA) for i in range(lag)])
        roF = np.array([self.correlation_coefficient(self.covariance(i + 1, open=False), self.varianceF, self.varianceF) for i in range(lag)])
        roAF = np.array([self.correlation_coefficient(self.covariance_AF(i + 1), self.varianceA, self.varianceF) for i in range(lag)])
        return roA, roF, roAF

    
    def _get_eigen_and_matrices(self, open: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper method to retrieve eigenvalues and matrices based on open or shut states.

        Parameters:
        open : bool
            Determines if the parameters for open (True) or shut (False) times should be retrieved.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] : Tuple of u, phi, inverse of Qxx, eigenvalues, and matrix A for correlation limit computation.
        """
        if open:
            u, phi, invQxx, eigs, A = self.uA, self.phiAr, -np.linalg.inv(self.QAA), self.eigs_XAA, self.A_XAA
        else:
            u, phi, invQxx, eigs, A = self.uF, self.phiFr, -np.linalg.inv(self.QFF), self.eigs_XFF, self.A_XFF
        return u, phi, invQxx, eigs, A

    def _get_open_or_shut_params(self, open: bool, lag: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cached parameters for open or shut times based on the state."""
        if open:
            u, phi = self.uA, self.phiAr
            invQxx = -np.linalg.inv(self.QAA)
            Xn = qml.powQ(self.XAA, lag)
        else:
            u, phi = self.uF, self.phiFr
            invQxx = -np.linalg.inv(self.QFF)
            Xn = qml.powQ(self.XFF, lag)
        return u, phi, invQxx, Xn


class CorrelationDisplay(CorrelationCalculator):
    """ Prints various correlation and Q-matrix calculations. """

    @property
    def print_all(self) -> str:
        """Returns all correlation information."""
        return (f"\n***** CORRELATIONS *****\n"
                f"{self.print_ranks}"
                f"{self.print_open_correlations}"
                f"{self.print_shut_correlations}"
                f"{self.print_open_shut_correlations}")

    @property
    def print_ranks(self) -> str:
        """Prints ranks and eigenvalues of the matrices."""
        return (f"\nRanks of GAF, GFA = {self.rank_GAF}, {self.rank_GFA}"
                f"\nRank of GFA * GAF = {self.rank_XFF}"
                f"\nRank of GAF * GFA = {self.rank_XAA}")

    def _format_correlation_info(self, var: float, var_n: float, n: int, correlation_limit: float, correlation_type: str) -> str:
        """Helper method to format correlation information for open and shut times."""
        percentage_diff = 100 * (np.sqrt(var_n / n ** 2) - np.sqrt(var / n)) / np.sqrt(var / n)
        limiting_percentage = 100 * (np.sqrt(1 + 2 * correlation_limit / var) - 1)

        return (f"\nVariance of {correlation_type} time = {var:.5g}\n"
                f"SD of all {correlation_type} times = {np.sqrt(var):.5g}\n"
                f"SD of means of {n} {correlation_type} times if uncorrelated = {np.sqrt(var / n):.5g}\n"
                f"Actual SD of mean = {np.sqrt(var_n / n ** 2):.5g}\n"
                f"Percentage difference as result of correlation = {percentage_diff:.5g}\n"
                f"Limiting value of percent difference for large n = {limiting_percentage:.5g}")

    def _format_correlation_coefficients(self, var: float, n: int, open: bool = True) -> str:
        """Helper method to format correlation coefficients for open or shut times."""
        correlation_str = '\nCorrelation coefficients, r(k), for up to lag k = 5:'
        for i in range(n):
            corr_coeff = self.correlation_coefficient(self.covariance(i + 1, open=open), var, var)
            correlation_str += f"\nr({i + 1}) = {corr_coeff:.5g}"
        return correlation_str

    @property
    def print_open_correlations(self) -> str:
        """Prints open-open time correlations."""
        varA_n = self.variance_n(50, open=True)
        correlation_limit_A = self.correlation_limit(open=True)

        open_str = '\n\nOPEN-OPEN TIME CORRELATIONS'
        open_str += self._format_correlation_info(self.varianceA, varA_n, 50, correlation_limit_A, 'open')
        open_str += self._format_correlation_coefficients(self.varianceA, 5, open=True)
        return open_str

    @property
    def print_shut_correlations(self) -> str:
        """Prints shut-shut time correlations."""
        varF_n = self.variance_n(50, open=False)
        correlation_limit_F = self.correlation_limit(open=False)

        shut_str = '\n\nSHUT-SHUT TIME CORRELATIONS'
        shut_str += self._format_correlation_info(self.varianceF, varF_n, 50, correlation_limit_F, 'shut')
        shut_str += self._format_correlation_coefficients(self.varianceF, 5, open=False)
        return shut_str

    @property
    def print_open_shut_correlations(self) -> str:
        """Prints open-shut time correlations."""
        open_shut_str = '\n\nOPEN - SHUT TIME CORRELATIONS'
        open_shut_str += '\nCorrelation coefficients, r(k), for up to lag k = 5:'

        for i in range(5):
            covAF = self.covariance_AF(i + 1)
            open_shut_str += f"\nr({i + 1}) = {self.correlation_coefficient(covAF, self.varianceA, self.varianceF):.5g}"
        return open_shut_str

    def plot_corr_open_shut(self, lag: int) -> None:
        """
        Plot the correlation coefficients for open, shut, and open-shut times over a given number of lags.
        
        Parameters
        ----------
        lag : int
            Number of lags to plot.
        """
        if lag < 1:
            raise ValueError("Lag must be a positive integer.")

        r = np.arange(1, lag + 1)
        roA, roF, roAF = self.calculate_correlation_coefficients(lag)

        plt.figure(figsize=(6, 4))
        plt.plot(r, roA, 'go-', label='Open-Open Correlations (roA)')
        plt.plot(r, roF, 'ro-', label='Shut-Shut Correlations (roF)')
        plt.plot(r, roAF, 'bo-', label='Open-Shut Correlations (roAF)')

        plt.title(f'Correlation Coefficients for Open, Shut, and Open-Shut Times (Lag {lag})')
        plt.xlabel('Lag')
        plt.ylabel('Correlation Coefficient')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    c = 0.0000001 # 0.1 uM
    mec = samples.CH82()
    mec.set_eff('c', c)
    #calc = CorrelationCalculator(mec)
    disp = CorrelationDisplay(mec)
    print(disp.print_all)
    disp.plot_corr_open_shut(5)
