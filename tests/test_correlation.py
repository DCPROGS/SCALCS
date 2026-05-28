import unittest
import numpy as np

from samples import samples
from scalcs.sccorrelation import CorrelationCalculator

class TestSCCorrelations(unittest.TestCase):
    
    def setUp(self):
        # Setup CH82 test data
        c = 0.0000001 # 0.1 uM
        mec = samples.CH82()
        mec.set_eff('c', c)
        self.scc = CorrelationCalculator(mec) #.Q, mec.kA, mec.kB, mec.kC, mec.kD)

    def test_open_time_variance(self):
        # Expected variance for open times
        expected_variance = 3.8957e-06
        calculated_variance = self.scc.variance(open=True)
        self.assertAlmostEqual(calculated_variance, expected_variance, places=7,
                               msg=f"Expected variance of open time: {expected_variance}, got: {calculated_variance}")

    def test_shut_time_variance(self):
        # Expected variance for shut times
        expected_variance = 6.5374058
        calculated_variance = self.scc.variance(open=False)
        self.assertAlmostEqual(calculated_variance, expected_variance, places=5,
                               msg=f"Expected variance of shut time: {expected_variance}, got: {calculated_variance}")
    
    def test_open_sd_of_means(self):
        # Expected SD of means for open times if uncorrelated
        expected_sd_means = 0.27913
        calculated_variance = self.scc.variance(open=True)
        calculated_sd_means = np.sqrt(calculated_variance / 50) * 1000  # Convert to ms
        self.assertAlmostEqual(calculated_sd_means, expected_sd_means, places=5,
                               msg=f"Expected SD of means for open times: {expected_sd_means}, got: {calculated_sd_means}")

    def test_shut_sd_of_means(self):
        # Expected SD of means for shut times if uncorrelated
        expected_sd_means = 361.59
        calculated_variance = self.scc.variance(open=False)
        calculated_sd_means = np.sqrt(calculated_variance / 50) * 1000  # Convert to ms
        self.assertAlmostEqual(calculated_sd_means, expected_sd_means, places=2,
                               msg=f"Expected SD of means for shut times: {expected_sd_means}, got: {calculated_sd_means}")

    def test_open_correlation_coefficients(self):
        # Expected correlation coefficients for open times up to lag k = 5
        expected_correlations = [0.010259, 0.0021908, 0.00046783, 9.9904e-05, 2.1334e-05]
        variance = self.scc.variance(open=True)
        for i in range(5):
            cov = self.scc.covariance(i + 1, open=True)
            corr_coeff = self.scc.correlation_coefficient(cov, variance, variance)
            self.assertAlmostEqual(corr_coeff, expected_correlations[i], places=6,
                                   msg=f"Expected r({i+1}) for open times: {expected_correlations[i]}, got: {corr_coeff}")

    def test_shut_correlation_coefficients(self):
        # Expected correlation coefficients for shut times up to lag k = 5
        expected_correlations = [0.087454, 0.018676, 0.0039881, 0.00085165, 0.00018187]
        variance = self.scc.variance(open=False)
        for i in range(5):
            cov = self.scc.covariance(i + 1, open=False)
            corr_coeff = self.scc.correlation_coefficient(cov, variance, variance)
            self.assertAlmostEqual(corr_coeff, expected_correlations[i], places=5,
                                   msg=f"Expected r({i+1}) for shut times: {expected_correlations[i]}, got: {corr_coeff}")

    def test_open_shut_correlation_coefficients(self):
        # Expected correlation coefficients for open-shut times up to lag k = 5
        expected_correlations = [-0.064817, -0.013842, -0.0029558, -0.00063121, -0.00013479]
        varA = self.scc.variance(open=True)
        varF = self.scc.variance(open=False)
        for i in range(5):
            covAF = self.scc.covariance_AF(i + 1)
            corr_coeff = self.scc.correlation_coefficient(covAF, varA, varF)
            self.assertAlmostEqual(corr_coeff, expected_correlations[i], places=5,
                                   msg=f"Expected r({i+1}) for open-shut times: {expected_correlations[i]}, got: {corr_coeff}")

    def test_limiting_percentage_open(self):
        # Expected limiting percentage difference for open times for large n
        expected_limiting_percentage = 1.296
        varA = self.scc.variance(open=True)
        correlation_limit_A = self.scc.correlation_limit(open=True)
        calculated_limiting_percentage = 100 * (np.sqrt(1 + 2 * correlation_limit_A / varA) - 1)
        self.assertAlmostEqual(calculated_limiting_percentage, expected_limiting_percentage, places=3,
                               msg=f"Expected limiting percentage for open times: {expected_limiting_percentage}, got: {calculated_limiting_percentage}")

    def test_limiting_percentage_shut(self):
        # Expected limiting percentage difference for shut times for large n
        expected_limiting_percentage = 10.562
        varF = self.scc.variance(open=False)
        correlation_limit_F = self.scc.correlation_limit(open=False)
        calculated_limiting_percentage = 100 * (np.sqrt(1 + 2 * correlation_limit_F / varF) - 1)
        self.assertAlmostEqual(calculated_limiting_percentage, expected_limiting_percentage, places=3,
                               msg=f"Expected limiting percentage for shut times: {expected_limiting_percentage}, got: {calculated_limiting_percentage}")

if __name__ == '__main__':
    unittest.main()
