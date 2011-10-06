
from dcpyps import samples
from dcpyps import popen
from dcpyps import pdfs
from dcpyps import scburst
from dcpyps import scplotlib as scpl

import sys
import unittest

class TestDC_PyPs(unittest.TestCase):

    def setUp(self):
        self.mec = samples.CH82()

    def runTest(self):
        pass

    def test_burst(self):
        sys.stdout.write('\n%s' % self.mec)
        sys.stdout.write('\n******************\n')
        sys.stdout.write('TESTING SCBURST...\n')
        conc = 100e-9
        self.mec.set_eff('c', conc)

        # # # Burst initial vector.
        phiB = scburst.phiBurst(self.mec)
        sys.stdout.write('\nInitial vector for burst (phiB) = \n')
        for i in range(self.mec.kA):
            sys.stdout.write('{0:.5g}\t'.format(phiB[i]))
        self.assertAlmostEqual(phiB[0], 0.275362, 6)
        self.assertAlmostEqual(phiB[1], 0.724638, 6)

        # # # Openings per burst.
        rho, w = scburst.openings_distr_components(self.mec)
        mean, sd = pdfs.geometricPDF_mean_sd(rho, w)
        mu = scburst.openings_mean(self.mec)
        sys.stdout.write('\nMean number of openings per burst calculated:')
        sys.stdout.write('\n\tfrom geometric pdf components = {0:.5g}'.
            format(mean))
        sys.stdout.write('\n\tdirectly from Q matrix = {0:.5g}'. format(mu))
        self.assertAlmostEqual(mean, 3.81864, 5)
        self.assertAlmostEqual(mu, 3.81864, 5)

        # # # Burst length.
        eigs, w = scburst.length_pdf_components(self.mec)
        mean, sd = pdfs.expPDF_mean_sd(1 / eigs, w / eigs)
        mbl = scburst.length_mean(self.mec)
        sys.stdout.write('\nMean burst length (ms) calculated:')
        sys.stdout.write('\n\tfrom exponential pdf components = {0:.5g}'.
            format(mean * 1000))
        sys.stdout.write('\n\tdirectly from Q matrix = {0:.5g}'. format(mbl * 1000))
        self.assertAlmostEqual(mean * 1000, 7.32810, 5)
        self.assertAlmostEqual(mbl * 1000, 7.32810, 5)

        # # # Total open time.
        eigs, w = scburst.open_time_total_pdf_components(self.mec)
        mean, sd = pdfs.expPDF_mean_sd(1 / eigs, w / eigs)
        mop = scburst.open_time_mean(self.mec)
        sys.stdout.write('\nMean total open time (ms) calculated:')
        sys.stdout.write('\n\tfrom exponential pdf components = {0:.5g}'.
            format(mean * 1000))
        sys.stdout.write('\n\tdirectly from Q matrix = {0:.5g}'. format(mop * 1000))
        self.assertAlmostEqual(mean * 1000, 7.16585, 5)
        self.assertAlmostEqual(mop * 1000, 7.16585, 5)

        
    def test_popen(self):

        sys.stdout.write('\n\n******************\n')
        sys.stdout.write('TESTING POPEN...\n')

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
