import dcpyps
from dcpyps import samples
from dcpyps import popen
from dcpyps import scplotlib as scpl

import sys
import unittest

class TestDC_PyPs(unittest.TestCase):

    def setUp(self):
        self.mec = samples.CH82()
        
    def test_popen(self):
        sys.stdout.write('%s' % self.mec)

        tres = 0.0001  # resolution in seconds
        self.mec.fastBlk = False
        self.mec.KBlk = 0.01
        conc = 100e-9    # 100 nM

        # POPEN CURVE CALCULATIONS
        sys.stdout.write('\n\nCalculating Popen curve parameters:')
        popen.printout(self.mec, tres)
        c, pe, pi = scpl.Popen(self.mec, tres)
        self.assertTrue(pi[-1]>0.967 and pi[-1]<0.969)

if __name__ == '__main__':
    unittest.main()
