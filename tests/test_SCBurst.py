import unittest
import numpy as np

from samples import samples
from scalcs.scburst import SCBurst

class TestSCBurst(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        # Setup CH82 test data
        c = 0.0000001 # 0.1 uM
        mec = samples.CH82()
        mec.set_eff('c', c)
        self.q_burst = SCBurst(mec)

        # Mocking values from the output example
        self.expected_start_burst = np.array([0.27536232, 0.72463768])
        self.expected_end_burst = np.array([[0.9609028], [0.20595089]])
        self.expected_mean_number_of_openings = 3.81864
        self.expected_mean_burst_length = 7.3281
        self.expected_mean_open_time = 7.16585
        self.expected_mean_shut_time = 0.162258
        self.expected_mean_shut_time_excl_single = 0.276814
        self.expected_mean_shut_time_between_bursts = 3790.43

        self.popen_within_burst = 0.97786
        self.total_popen = 0.0018869

        # PDF values for different properties (mocking expected output)
        self.total_burst_length_pdf_eigs = [  101.600835,  2012.597763,  3093.266488, 19408.201584]
        self.total_burst_length_pdf_amps = [ 74.738405,  28.668297, 773.533787,   1.498222]

        self.burst_length_no_single_openings_pdf_eigs = [101.600835, 2012.597763, 3093.266488, 19408.201584,  500.653595,  3050.013075]
        self.burst_length_no_single_openings_pdf_amps = [127.504299,   48.908337, 1319.654649,     2.555979, -128.574549, -1370.048716]

        self.total_open_time_pdf_eigs = [ 103.854522, 3028.14941 ]
        self.total_open_time_pdf_amps = [ 76.345644, 802.093067]

        self.first_opening_in_burst_pdf_eigs = [ 500.653595, 3050.013075]
        self.first_opening_in_burst_pdf_amps = [495.12486, 33.681397]

        self.openings_per_burst_pdf_rhos = [0.007144, 0.792567]
        self.openings_per_burst_pdf_amps = [0.260915, 0.152921]

        self.gaps_inside_burst_pdf_eigs = [2053.198363, 19011.801637]
        self.gaps_inside_burst_pdf_amps = [23.475151, 18794.431058]

        self.gaps_between_burst_pdf_eigs = [2.053198e+03, 1.901180e+04, 2.638954e-01, 2.062934e+03, 1.901180e+04]
        self.gaps_between_burst_pdf_amps = [-6.616803e+01, -5.297476e+04,  2.639684e-01,  6.591223e+01, 5.297475e+04]

    def test_popen_within_burst(self):
        # Test Popen within burst
        self.assertAlmostEqual(self.q_burst.burst_popen, self.popen_within_burst, places=5)

    def test_total_popen(self):
        # Test total Popen
        self.assertAlmostEqual(self.q_burst.total_popen, self.total_popen, places=7)

    def test_start_burst(self):
        result = self.q_burst.start_burst
        np.testing.assert_almost_equal(result, self.expected_start_burst, decimal=5)

    def test_end_burst(self):
        result = self.q_burst.end_burst
        np.testing.assert_almost_equal(result, self.expected_end_burst, decimal=5)

    def test_mean_number_of_openings(self):
        result = self.q_burst.mean_number_of_openings
        self.assertAlmostEqual(result, self.expected_mean_number_of_openings, places=5)

    def test_mean_burst_length(self):
        result = 1000 * self.q_burst.mean_length
        self.assertAlmostEqual(result, self.expected_mean_burst_length, places=4)

    def test_mean_open_time(self):
        result = 1000 * self.q_burst.mean_open_time
        self.assertAlmostEqual(result, self.expected_mean_open_time, places=4)

    def test_mean_shut_time(self):
        result = 1000 * self.q_burst.mean_shut_time
        self.assertAlmostEqual(result, self.expected_mean_shut_time, places=5)

    def test_mean_shut_time_excl_single(self):
        result = 1000 * self.q_burst.mean_shut_time / self.q_burst.probability_more_than_one_opening
        self.assertAlmostEqual(result, self.expected_mean_shut_time_excl_single, places=5)

    def test_mean_shut_time_between_bursts(self):
        result = 1000 * self.q_burst.mean_shut_times_between_bursts
        self.assertAlmostEqual(result, self.expected_mean_shut_time_between_bursts, places=2)

    def test_total_burst_length_pdf(self):
        # Test PDF of total burst length (unconditional)
        eigs, amps = self.q_burst.length_pdf_components()
        np.testing.assert_almost_equal(eigs, self.total_burst_length_pdf_eigs, decimal=6)
        np.testing.assert_almost_equal(amps, self.total_burst_length_pdf_amps, decimal=6)

    def test_burst_length_no_single_openings_pdf(self):
        # Test PDF of burst length for bursts with 2 or more openings
        eigs, amps = self.q_burst.length_pdf_no_single_openings_components()
        np.testing.assert_almost_equal(eigs, self.burst_length_no_single_openings_pdf_eigs, decimal=6)
        np.testing.assert_almost_equal(amps, self.burst_length_no_single_openings_pdf_amps, decimal=6)

    def test_total_open_time_pdf(self):
        # Test PDF of total open time per burst
        eigs, amps = self.q_burst.total_open_time_pdf_components()
        np.testing.assert_almost_equal(eigs, self.total_open_time_pdf_eigs, decimal=6)
        np.testing.assert_almost_equal(amps, self.total_open_time_pdf_amps, decimal=6)

    def test_first_opening_in_burst_pdf(self):
        # Test PDF of first opening in a burst with 2 or more openings     
        eigs, amps = self.q_burst.first_opening_length_pdf_components()
        np.testing.assert_almost_equal(eigs, self.first_opening_in_burst_pdf_eigs, decimal=6)
        np.testing.assert_almost_equal(amps, self.first_opening_in_burst_pdf_amps, decimal=6)   

    def test_openings_per_burst_pdf(self):
        # Test geometric PDF of number (r) of openings per burst (unconditional)
        rhos, amps = self.q_burst.openings_distr_components()
        np.testing.assert_almost_equal(rhos, self.openings_per_burst_pdf_rhos, decimal=6)
        np.testing.assert_almost_equal(amps, self.openings_per_burst_pdf_amps, decimal=6)
        
    def test_gaps_inside_burst_pdf(self):
        # Test PDF of gaps inside burst
        eigs, amps = self.q_burst.shut_times_inside_burst_pdf_components()
        np.testing.assert_almost_equal(eigs, self.gaps_inside_burst_pdf_eigs, decimal=6)
        np.testing.assert_almost_equal(amps, self.gaps_inside_burst_pdf_amps, decimal=6) 

    def test_gaps_between_burst_pdf(self):
        # Test PDF of gaps between bursts
        eigs, amps = self.q_burst.shut_times_between_burst_pdf_components()
        np.testing.assert_almost_equal(eigs, self.gaps_between_burst_pdf_eigs, decimal=2)
        np.testing.assert_almost_equal(amps, self.gaps_between_burst_pdf_amps, decimal=2)

if __name__ == '__main__':
    unittest.main()
