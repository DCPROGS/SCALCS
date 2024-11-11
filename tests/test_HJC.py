import unittest
import numpy as np

from samples import samples
from scalcs.scalcslib import ExactPDFCalculator, AsymptoticPDF

class TestExactPDFCalculator(unittest.TestCase):
    
    def setUp(self):
        # Set up your QMatrix instance here
        self.mec = samples.CH82()
        self.mec.set_eff('c', 0.0000001) 
        tres = 0.0001 # 10 us
        self.calculator_exact = ExactPDFCalculator(self.mec, tres=tres) 
        self.calculator_asymp = AsymptoticPDF(self.mec, tres=tres)

    def test_exact_open_time_pdf(self):
        # Expected open time results based on provided data
        expected_eigvals = np.array([6.06779790e-14, 1.01817908e+02, 2.02211927e+03, 3.09352724e+03, 1.94082023e+04])
        expected_gamma00 = np.array([3.79833091e-01, 1.37628426e+02, 1.50832630e+01, 4.43278099e+02, 2.82348045e+00])
        expected_gamma10 = np.array([  0.75378189,  -0.30035418,  10.81345106, -11.66360017,   0.39672141])
        expected_gamma11 = np.array([ 1.44273177e-01,  8.22575034e+03,  4.81181018e+02, -1.50898006e+04, -1.63784217e+02])

        eigvals, gamma00, gamma10, gamma11 = self.calculator_exact.exact_GAMAxx(open=True)

        # Verify the calculated values against the expected ones
        np.testing.assert_allclose(eigvals, expected_eigvals, rtol=1e-6)
        np.testing.assert_allclose(gamma00, expected_gamma00, rtol=1e-6)
        np.testing.assert_allclose(gamma10, expected_gamma10, rtol=1e-6)
        np.testing.assert_allclose(gamma11, expected_gamma11, rtol=1e-6)

    def test_exact_shut_time_pdf(self):
        # Expected shut time results based on provided data
        expected_eigvals = np.array([6.06779790e-14, 1.01817908e+02, 2.02211927e+03, 3.09352724e+03,  1.94082023e+04])
        expected_gamma00 = np.array([9.40819385e-01, 1.17815511e+02, 2.48961904e+01, 1.28843436e+00, 5.37018080e+03])
        expected_gamma10 = np.array([4.57791957, 100.2107672, -5.49855015, 0.67154784, -99.96168445])
        expected_gamma11 = np.array([ 8.85141115e-01,  4.36349919e+04,  7.18068399e+02, -3.97437239e+01, -1.98322881e+06])

        eigvals, gamma00, gamma10, gamma11 = self.calculator_exact.exact_GAMAxx(open=False)

        # Verify the calculated values against the expected ones       
        np.testing.assert_allclose(eigvals, expected_eigvals, rtol=1e-6)
        np.testing.assert_allclose(gamma00, expected_gamma00, rtol=1e-6)
        np.testing.assert_allclose(gamma10, expected_gamma10, rtol=1e-6)
        np.testing.assert_allclose(gamma11, expected_gamma11, rtol=1e-6)

    def test_asymptotic_open_time_pdf(self):
        # Expected open time results based on provided data
        expected_roots = np.array([3045.2857761 ,  162.92946866])
        expected_areas = np.array([0.15074747, 0.84920398])

        e, a = self.calculator_asymp.HJC_asymptotic_open_time_pdf_components()

        # Verify the calculated values against the expected ones
        np.testing.assert_allclose(e, expected_roots, rtol=1e-6)
        np.testing.assert_allclose(a, expected_areas, rtol=1e-6)

    def test_asymptotic_shut_time_pdf(self):
        # Expected open time results based on provided data
        expected_roots = np.array([1.70901928e+04, 2.05808129e+03, 2.43565355e-01])
        expected_areas = np.array([0.28581509, 0.01673115, 0.68354203])

        e, a = self.calculator_asymp.HJC_asymptotic_shut_time_pdf_components()

        # Verify the calculated values against the expected ones
        np.testing.assert_allclose(e, expected_roots, rtol=1e-6)
        np.testing.assert_allclose(a, expected_areas, rtol=1e-6)

    def test_HJC_vectors(self):
        # Expected open time results based on provided data
        expected_phiA = np.array([0.1539661, 0.8460339])
        expected_phiF = np.array([0.53036915, 0.3861155 , 0.08351535])

        A = self.calculator_asymp.HJCphiA
        F = self.calculator_asymp.HJCphiF

        # Verify the calculated values against the expected ones
        np.testing.assert_allclose(A, expected_phiA, rtol=1e-6)
        np.testing.assert_allclose(F, expected_phiF, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
