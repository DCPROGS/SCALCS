import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from samples import samples
from scalcs.qmatlib import eigenvalues_and_spectral_matrices, expQ, powQ 
from scalcs.qmatlib import pinf_extendQ, pinf_reduceQ

from scalcs.qmatlib import QMatrix

class TestQMatrixRegression(unittest.TestCase):
    """ Unit tests for the QMatrix class. """

    def setUp(self):
        # Set up your QMatrix instance here
        self.mec = samples.CH82()
        self.mec.set_eff('c', 0.0000001) 
        self.qmatrix = QMatrix(self.mec)
        self.pinf_expected_result = np.array([2.48271431e-05, 0.00186203552, 0.00496542821, 6.20678511e-05, 0.993085641])
 
    def test_pinf_method1(self):
        # Test Q extention method to calculate Pinf
        result = pinf_extendQ(self.mec.Q)
        np.testing.assert_allclose(result, self.pinf_expected_result, rtol=1e-6)

    def test_pinf_method2(self):
        # Test Q reduction method to calculate Pinf
        result = pinf_reduceQ(self.mec.Q)
        np.testing.assert_allclose(result, self.pinf_expected_result, rtol=1e-6)

    def test_pinf(self):
        """ Test equilibrium state occupancies. """
        np.testing.assert_array_almost_equal(self.qmatrix.pinf, self.pinf_expected_result, decimal=6)
        
    def test_Popen(self):
        """ Test equilibrium open probability. """
        expected_Popen = 0.001887
        self.assertAlmostEqual(self.qmatrix.Popen(), expected_Popen, places=6) 
        
    def test_state_lifetimes(self):
        """ Test mean lifetimes of each individual state in milliseconds. """
        expected_lifetimes = [0.0003278688524590164, 0.001997336870856612, 0.00048426150121065375, 5.2631578947368424e-05, 0.1]
        computed_lifetimes = [lifetime for lifetime in self.qmatrix.state_lifetimes()]
        np.testing.assert_array_almost_equal(computed_lifetimes, expected_lifetimes, decimal=6)
        
    def test_transition_probability(self):
        """ Test probability of transitions regardless of time. """
        expected_pi = np.array([
            [0, 0.0164, 0.9836, 0, 0],
            [0.0013, 0, 0, 0.9987, 0],
            [0.0073, 0, 0, 0.0242, 0.9685],
            [0, 0.7895, 0.2105, 0, 0],
            [0, 0, 1, 0, 0] ])
        np.testing.assert_array_almost_equal(self.qmatrix.transition_probability(), expected_pi, decimal=4)

    def test_transition_frequency(self):
        """ Test frequency of transitions per second. """
        expected_fi = np.array([
            [0, 0.0012, 0.0745, 0, 0],
            [0.0012, 0, 0, 0.931, 0],
            [0.0745, 0, 0, 0.2483, 9.9309],
            [0, 0.931, 0.2483, 0, 0],
            [0, 0, 9.9309, 0, 0] ])
        np.testing.assert_array_almost_equal(self.qmatrix.transition_frequency(), expected_fi, decimal=4)

    def test_subset_probabilities(self):
        """ Test initial probabilities of being in various subsets (A, B, C, F). """
        self.assertAlmostEqual(self.qmatrix.P("A"), 0.001887, places=6)
        self.assertAlmostEqual(self.qmatrix.P("B"), 0.005027, places=6)
        self.assertAlmostEqual(self.qmatrix.P("C"), 0.993085641279747, places=6)
        self.assertAlmostEqual(self.qmatrix.P("F"), 0.9981131373372089, places=6)

    def test_conditional_probabilities(self):
        """ Test conditional probabilities P(B|F) and P(C|F). """
        self.assertAlmostEqual(self.qmatrix.P("B|F"), 0.005037, places=6)
        self.assertAlmostEqual(self.qmatrix.P("C|F"), 0.9949629998149565, places=6)
        
    def test_open_time_pdf(self):
        """ Test open time PDF (unconditional). """
        expected_e_open = [500.65359469, 3050.01307531]
        expected_w_open = [464.41452879, 220.77066058]
        e, w = self.qmatrix.ideal_open_time_pdf_components()
        np.testing.assert_array_almost_equal(e, expected_e_open, decimal=6)
        np.testing.assert_array_almost_equal(w, expected_w_open, decimal=6)

    def test_shut_time_pdf(self):
        """ Test shut time PDF (unconditional). """
        expected_e_shut = [0.263895376, 2062.93374, 19011.8024]
        expected_w_shut = [6.91262570e-02, 1.72606505e+01, 1.38726701e+04]
        e, w = self.qmatrix.ideal_shut_time_pdf_components()
        np.testing.assert_array_almost_equal(e, expected_e_shut, decimal=4)
        np.testing.assert_array_almost_equal(w, expected_w_shut, decimal=4)


class TestEquilibriumOccupancies(unittest.TestCase):

    def test_pinf_extendQ(self):
        Q = np.array([[0, 1, -1],
                      [-1, 0, 1],
                      [1, -1, 0]])
        expected_pinf = np.array([0.33333333, 0.33333333, 0.33333333])
        pinf = pinf_extendQ(Q)
        assert_almost_equal(pinf, expected_pinf, decimal=5)

    def test_pinf_reduceQ(self):
        Q = np.array([[0, 1, -1],
                      [-1, 0, 1],
                      [1, -1, 0]])
        expected_pinf = np.array([0.33333333, 0.33333333, 0.33333333])
        pinf = pinf_reduceQ(Q)
        assert_almost_equal(pinf, expected_pinf, decimal=5)

class TestMatrixFunctions(unittest.TestCase):

    def test_eigenvalues_and_spectral_matrices(self):
        Q = np.array([[1, 2], [3, 4]])
        eigvals, A = eigenvalues_and_spectral_matrices(Q)
        
        expected_eigvals, expected_vectors = np.linalg.eig(Q)
        expected_inv_vectors = np.linalg.inv(expected_vectors)
        expected_A = np.einsum('ij,kj->kij', expected_vectors, expected_inv_vectors)
        
        sorted_indices = expected_eigvals.real.argsort()
        expected_eigvals = expected_eigvals[sorted_indices]
        expected_A = expected_A[sorted_indices]

        assert_almost_equal(eigvals, expected_eigvals)
        #assert_almost_equal(A, expected_A)

    def test_powQ(self):
        Q = np.array([[2, 0], [0, 3]])
        n = 3
        result = powQ(Q, n)
        expected = np.array([[8, 0], [0, 27]])

        assert_almost_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
