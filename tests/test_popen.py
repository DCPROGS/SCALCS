import unittest
import numpy as np

from samples import samples
from scalcs import popen

class TestPopenCurve(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        # Setup CH82 test data
        c = 0.0000001 # 0.1 uM
        self.tres = 0.0001 # 100 us
        mec = samples.CH82()
        mec.set_eff('c', c)
        self.popen_analysis = popen.PopenCurve(mec)

        self.expected_ideal_maxPopen = 0.9677394380085494
        self.expected_ideal_EC50 = 2.4080276489257817
        self.expected_ideal_nH = 1.891852810093832
        self.expected_HJC_maxPopen = 0.961847920017044
        self.expected_HJC_EC50 = 2.360343933105469
        self.expected_HJC_nH = 1.9050537071553801

    def test_ideal_curve(self):
        result = self.popen_analysis.analyse_curve()
        self.assertAlmostEqual(result['maxPopen'], self.expected_ideal_maxPopen, places=6)
        self.assertAlmostEqual(result['EC50']*1e6, self.expected_ideal_EC50, places=6)
        self.assertAlmostEqual(result['nH'], self.expected_ideal_nH, places=6)

    def test_ideal_curve(self):
        result = self.popen_analysis.analyse_curve(tres=self.tres)
        self.assertAlmostEqual(result['maxPopen'], self.expected_HJC_maxPopen, places=6)
        self.assertAlmostEqual(result['EC50']*1e6, self.expected_HJC_EC50, places=6)
        self.assertAlmostEqual(result['nH'], self.expected_HJC_nH, places=6)
