
from dcpyps import samples
from dcpyps import popen
from dcpyps import pdfs
from dcpyps import scburst
from dcpyps import scalcslib as scl
from dcpyps import scplotlib as scpl
from dcpyps import qmatlib as qml

import sys
import unittest
import numpy as np

class TestDC_PyPs(unittest.TestCase):

    def setUp(self):
        self.mec = samples.CH82()
        self.conc = 100e-9 # 100 nM
        self.mec.set_eff('c', self.conc)
        self.tres = 0.0001 # 100 microsec

    # def runTest(self):
    #     pass

    def test_burst(self):
        sys.stdout.write('\n%s' % self.mec)
        sys.stdout.write('\n******************\n')
        sys.stdout.write('TESTING SCBURST...\n')

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

    def test_openshut(self):

        sys.stdout.write('\n\n******************\n')
        sys.stdout.write('TESTING QMATLIB AND SCALCSLIB...')

        # # # Initial HJC vectors.
        expQFF = qml.expQt(self.mec.QFF, self.tres)
        expQAA = qml.expQt(self.mec.QAA, self.tres)
        GAF, GFA = qml.iGs(self.mec.Q, self.mec.kA, self.mec.kF)
        eGAF = qml.eGs(GAF, GFA, self.mec.kA, self.mec.kF, expQFF)
        eGFA = qml.eGs(GFA, GAF, self.mec.kF, self.mec.kA, expQAA)
        phiA = qml.phiHJC(eGAF, eGFA, self.mec.kA)
        phiF = qml.phiHJC(eGFA, eGAF, self.mec.kF)

        sys.stdout.write('\n\nInitial HJC vector for openings phiOp =\n')
        for i in range(phiA.shape[0]):
            sys.stdout.write('\t{0:.6g}'.format(phiA[i]))
        sys.stdout.write('\n\nInitial HJC vector for shuttings phiSh =\n')
        for i in range(phiF.shape[0]):
            sys.stdout.write('\t{0:.6g}'.format(phiF[i]))
        self.assertAlmostEqual(phiA[0], 0.153966, 6)
        self.assertAlmostEqual(phiA[1], 0.846034, 6)
        self.assertAlmostEqual(phiF[0], 0.530369, 6)
        self.assertAlmostEqual(phiF[1], 0.386116, 6)
        self.assertAlmostEqual(phiF[2], 0.0835153, 6)
        
        # Ideal shut time pdf
        eigs, w = scl.ideal_dwell_time_pdf_components(self.mec.QFF,
            qml.phiF(self.mec))
        sys.stdout.write('\n\nIdeal shut time pdf')
        pdfs.expPDF_printout(eigs, w, sys.stdout)
        self.assertAlmostEqual(eigs[0], 0.263895, 6)
        self.assertAlmostEqual(eigs[1], 2062.93, 2)
        self.assertAlmostEqual(eigs[2], 19011.8, 1)
        self.assertAlmostEqual(w[0], 0.0691263, 6)
        self.assertAlmostEqual(w[1], 17.2607, 4)
        self.assertAlmostEqual(w[2], 13872.7, 1)

        # Asymptotic shut time pdf
        roots = scl.asymptotic_roots(self.tres,
            self.mec.QFF, self.mec.QAA, self.mec.QFA, self.mec.QAF,
            self.mec.kF, self.mec.kA)
        areas = scl.asymptotic_areas(self.tres, roots,
            self.mec.QFF, self.mec.QAA, self.mec.QFA, self.mec.QAF,
            self.mec.kF, self.mec.kA, GFA, GAF)
        sys.stdout.write('\n\nAsymptotic shut time pdf')
        pdfs.expPDF_printout(-roots, -roots * areas, sys.stdout)
        mean = scl.exact_mean_time(self.tres,
                self.mec.QFF, self.mec.QAA, self.mec.QFA, self.mec.kF,
                self.mec.kA, GFA, GAF)
        sys.stdout.write('\nMean shut time (ms) = {0:.6f}'.format(mean * 1000))
        self.assertAlmostEqual(-roots[0], 17090.2, 1)
        self.assertAlmostEqual(-roots[1], 2058.08, 2)
        self.assertAlmostEqual(-roots[2], 0.243565, 6)
        self.assertAlmostEqual(areas[0] * 100, 28.5815, 4)
        self.assertAlmostEqual(areas[1] * 100, 1.67311, 5)
        self.assertAlmostEqual(areas[2] * 100, 68.3542, 4)

        # Exact pdf
        eigvals, gamma00, gamma10, gamma11 = scl.exact_GAMAxx(self.mec,
            self.tres, False)
        sys.stdout.write('\n\nExact shut time pdf')
        sys.stdout.write('\neigen\tg00(m)\tg10(m)\tg11(m)')
        for i in range(self.mec.k):
            sys.stdout.write('\n{0:.5g}'.format(eigvals[i]) +
            '\t{0:.5g}'.format(gamma00[i]) +
            '\t{0:.5g}'.format(gamma10[i]) +
            '\t{0:.5g}'.format(gamma11[i]))

        self.assertAlmostEqual(gamma00[0], 0.940819, 6)
        self.assertAlmostEqual(gamma00[1], 117.816, 3)
        self.assertAlmostEqual(gamma00[2], 24.8962, 4)
        self.assertAlmostEqual(gamma00[3], 1.28843, 5)
        self.assertAlmostEqual(gamma00[4], 5370.18, 2)
        self.assertAlmostEqual(gamma10[0], 4.57792, 5)
        self.assertAlmostEqual(gamma10[1], 100.211, 3)
        self.assertAlmostEqual(gamma10[2], -5.49855, 4)
        self.assertAlmostEqual(gamma10[3], 0.671548, 6)
        self.assertAlmostEqual(gamma10[4], -99.9617, 4)
        self.assertAlmostEqual(gamma11[0], 0.787344, 6)
        self.assertAlmostEqual(gamma11[1], 42275.7, 1)
        self.assertAlmostEqual(gamma11[2], 3.39701, 5)
        self.assertAlmostEqual(gamma11[3], -7.81406, 5)
        self.assertAlmostEqual(gamma11[4], -1.99149e+06, 0)
        
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
