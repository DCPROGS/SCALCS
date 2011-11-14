
from dcpyps import samples
from dcpyps import popen
from dcpyps import pdfs
from dcpyps import scburst
from dcpyps import cjumps
from dcpyps import scalcslib as scl
from dcpyps import scplotlib as scpl
from dcpyps import qmatlib as qml

import sys
import time
import unittest
import numpy as np

class TestDC_PyPs(unittest.TestCase):

    def setUp(self):
        self.mec = samples.CH82()
        self.conc = 100e-9 # 100 nM
        self.mec.set_eff('c', self.conc)
        self.tres = 0.0001 # 100 microsec

    def test_burst(self):

        # # # Burst initial vector.
        phiB = scburst.phiBurst(self.mec)
        self.assertAlmostEqual(phiB[0], 0.275362, 6)
        self.assertAlmostEqual(phiB[1], 0.724638, 6)

        # # # Openings per burst.
        rho, w = scburst.openings_distr_components(self.mec)
        mean, sd = pdfs.geometricPDF_mean_sd(rho, w)
        mu = scburst.openings_mean(self.mec)
        self.assertAlmostEqual(mean, 3.81864, 5)
        self.assertAlmostEqual(mu, 3.81864, 5)

        # # # Burst length.
        eigs, w = scburst.length_pdf_components(self.mec)
        mean, sd = pdfs.expPDF_mean_sd(1 / eigs, w / eigs)
        mbl = scburst.length_mean(self.mec)
        self.assertAlmostEqual(mean * 1000, 7.32810, 5)
        self.assertAlmostEqual(mbl * 1000, 7.32810, 5)

        # # # Total open time.
        eigs, w = scburst.open_time_total_pdf_components(self.mec)
        mean, sd = pdfs.expPDF_mean_sd(1 / eigs, w / eigs)
        mop = scburst.open_time_mean(self.mec)
        self.assertAlmostEqual(mean * 1000, 7.16585, 5)
        self.assertAlmostEqual(mop * 1000, 7.16585, 5)

    def test_openshut(self):

        # # # Initial HJC vectors.
        expQFF = qml.expQt(self.mec.QFF, self.tres)
        expQAA = qml.expQt(self.mec.QAA, self.tres)
        GAF, GFA = qml.iGs(self.mec.Q, self.mec.kA, self.mec.kF)
        eGAF = qml.eGs(GAF, GFA, self.mec.kA, self.mec.kF, expQFF)
        eGFA = qml.eGs(GFA, GAF, self.mec.kF, self.mec.kA, expQAA)
        phiA = qml.phiHJC(eGAF, eGFA, self.mec.kA)
        phiF = qml.phiHJC(eGFA, eGAF, self.mec.kF)

        self.assertAlmostEqual(phiA[0], 0.153966, 6)
        self.assertAlmostEqual(phiA[1], 0.846034, 6)
        self.assertAlmostEqual(phiF[0], 0.530369, 6)
        self.assertAlmostEqual(phiF[1], 0.386116, 6)
        self.assertAlmostEqual(phiF[2], 0.0835153, 6)
        
        # Ideal shut time pdf
        eigs, w = scl.ideal_dwell_time_pdf_components(self.mec.QFF,
            qml.phiF(self.mec))
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
        mean = scl.exact_mean_time(self.tres,
                self.mec.QFF, self.mec.QAA, self.mec.QFA, self.mec.kF,
                self.mec.kA, GFA, GAF)
        self.assertAlmostEqual(-roots[0], 17090.2, 1)
        self.assertAlmostEqual(-roots[1], 2058.08, 2)
        self.assertAlmostEqual(-roots[2], 0.243565, 6)
        self.assertAlmostEqual(areas[0] * 100, 28.5815, 4)
        self.assertAlmostEqual(areas[1] * 100, 1.67311, 5)
        self.assertAlmostEqual(areas[2] * 100, 68.3542, 4)

        # Exact pdf
        eigvals, gamma00, gamma10, gamma11 = scl.exact_GAMAxx(self.mec,
            self.tres, False)
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

    def test_cjumps(self):

        start = time.time()
        t, c, Popen, P = cjumps.solve_jump(self.mec, 0.05, 0.000005,
            cjumps.pulse_instexp, (0.00001, 0.0, 0.005, 0.0025))
        elapsed1 = time.time() - start
        maxP1 = max(Popen)
        start = time.time()
        t, c, Popen, P = cjumps.calc_jump(self.mec, 0.05, 0.000005,
            cjumps.pulse_instexp, (0.00001, 0.0, 0.005, 0.0025))
        elapsed2 = time.time() - start
        print ('\ntesting jump calculation...' +
            '\ndirect matrix calculation took {0:.6f} s'.format(elapsed2) +
            '\nintegration took {0:.6f} s'.format(elapsed1))
        maxP2 = max(Popen)
        self.assertAlmostEqual(maxP1, maxP2, 3)

        start = time.time()
        t, c, Popen, P  = cjumps.solve_jump(self.mec, 0.05, 0.000005,
            cjumps.pulse_erf, (0.000001, 0.0, 0.01, 0.01, 0.0002, 0.0002))
        elapsed1 = time.time() - start
        maxP1 = max(Popen)
        start = time.time()
        t, c, Popen, P = cjumps.calc_jump(self.mec, 0.05, 0.000005,
            cjumps.pulse_erf, (0.000001, 0.0, 0.01, 0.01, 0.0002, 0.0002))
        elapsed2 = time.time() - start
        print ('\ntesting jump calculation...' +
            '\ndirect matrix calculation took {0:.6f} s'.format(elapsed2) +
            '\nintegration took {0:.6f} s'.format(elapsed1))
        maxP2 = max(Popen)
        self.assertAlmostEqual(maxP1, maxP2, 3)

    def test_popen(self):

        self.mec.fastBlk = False
        self.mec.KBlk = 0.01

        # POPEN CURVE CALCULATIONS
        c, pe, pi = scpl.Popen(self.mec, self.tres)
        self.assertTrue(pi[-1]>0.967 and pi[-1]<0.969)
