
from dcpyps import samples
from dcpyps import popen
from dcpyps import pdfs
from dcpyps import scburst
from dcpyps import cjumps
from dcpyps import scalcslib as scl
from dcpyps import scplotlib as scpl
from dcpyps import qmatlib as qml
from dcpyps import dcio
from dcpyps import dataset

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
        self.tcrit = 0.004

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
        self.assertAlmostEqual(gamma11[0], 0.885141, 6)
        self.assertAlmostEqual(gamma11[1], 43634.99, 1)
        self.assertAlmostEqual(gamma11[2], 718.068, 3)
        self.assertAlmostEqual(gamma11[3], -39.7437, 3)
        self.assertAlmostEqual(gamma11[4], -1.9832288e+06, 0)

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
        print ('\ntesting jump calculation speed...' +
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
        print ('\ntesting jump calculation speed...' +
            '\ndirect matrix calculation took {0:.6f} s'.format(elapsed2) +
            '\nintegration took {0:.6f} s'.format(elapsed1))
        maxP2 = max(Popen)
        self.assertAlmostEqual(maxP1, maxP2, 3)

    def test_likelihood(self):

        GAF, GFA = qml.iGs(self.mec.Q, self.mec.kA, self.mec.kF)
        expQFF = qml.expQt(self.mec.QFF, self.tres)
        expQAA = qml.expQt(self.mec.QAA, self.tres)
        eGAF = qml.eGs(GAF, GFA, self.mec.kA, self.mec.kF, expQFF)
        eGFA = qml.eGs(GFA, GAF, self.mec.kF, self.mec.kA, expQAA)
        phiF = qml.phiHJC(eGFA, eGAF, self.mec.kF)
        eigs, A = qml.eigs(-self.mec.Q)

        Aeigvals, AZ00, AZ10, AZ11 = qml.Zxx(self.mec.Q, eigs, A, self.mec.kA,
            self.mec.QFF, self.mec.QAF, self.mec.QFA, expQFF, True)
        Aroots = scl.asymptotic_roots(self.tres, self.mec.QAA, self.mec.QFF,
            self.mec.QAF, self.mec.QFA, self.mec.kA, self.mec.kF)
        AR = qml.AR(Aroots, self.tres, self.mec.QAA, self.mec.QFF,
            self.mec.QAF, self.mec.QFA, self.mec.kA, self.mec.kF)
        Feigvals, FZ00, FZ10, FZ11 = qml.Zxx(self.mec.Q, eigs, A, self.mec.kA,
            self.mec.QAA, self.mec.QFA, self.mec.QAF, expQAA, False)
        Froots = scl.asymptotic_roots(self.tres, self.mec.QFF, self.mec.QAA,
            self.mec.QFA, self.mec.QAF, self.mec.kF, self.mec.kA)
        FR = qml.AR(Froots, self.tres, self.mec.QFF, self.mec.QAA,
            self.mec.QFA, self.mec.QAF, self.mec.kF, self.mec.kA)

        startB, endB = qml.CHSvec(Froots, self.tres, self.tcrit,
            self.mec.QFA, self.mec.kA, expQAA, phiF, FR)
        self.assertAlmostEqual(startB[0], 0.22041811, 6)
        self.assertAlmostEqual(startB[1], 0.77958189, 6)
        self.assertAlmostEqual(endB[0, 0], 0.97485171, 6)
        self.assertAlmostEqual(endB[1, 0], 0.21346049, 6)
        self.assertAlmostEqual(endB[2, 0], 0.99917949, 6)

#        t = 0.0010134001973
#        # Open time. Asymptotic solution.
#        eGAFt = qml.eGAF(t, self.tres, Aeigvals, AZ00, AZ10, AZ11, Aroots,
#            AR, self.mec.QAF, expQFF)
#        self.assertAlmostEqual(eGAFt[0,0], 152.44145644, 6)
#        self.assertAlmostEqual(eGAFt[0,1], 1.51526687, 6)
#        self.assertAlmostEqual(eGAFt[0,2], 33.72923718, 6)
#        self.assertAlmostEqual(eGAFt[1,0], 67.29121672, 6)
#        self.assertAlmostEqual(eGAFt[1,1], 63.80986196, 6)
#        self.assertAlmostEqual(eGAFt[1,2], 9.27940684, 6)
#        # Shut time. Asymptotic solution.
#        eGAFt = qml.eGAF(t, self.tres, Feigvals, FZ00, FZ10, FZ11, Froots,
#            FR, self.mec.QFA, expQAA)
#        self.assertAlmostEqual(eGAFt[0,0], 1.73048864, 6)
#        self.assertAlmostEqual(eGAFt[0,1], 6.9056438, 6)
#        self.assertAlmostEqual(eGAFt[1,0], 0.42762192, 6)
#        self.assertAlmostEqual(eGAFt[1,1], 1.70926206, 6)
#        self.assertAlmostEqual(eGAFt[2,0], 0.04548765, 6)
#        self.assertAlmostEqual(eGAFt[2,1], 0.15704816, 6)
#
#        t = 0.0001
#        # Open time. Exact solution.
#        eGAFt = qml.eGAF(t, self.tres, Aeigvals, AZ00, AZ10, AZ11, Aroots,
#            AR, self.mec.QAF, expQFF)
#        self.assertAlmostEqual(eGAFt[0,0], 2442.03379283, 6)
#        self.assertAlmostEqual(eGAFt[0,1], 5.88221827, 6)
#        self.assertAlmostEqual(eGAFt[0,2], 541.9589599, 6)
#        self.assertAlmostEqual(eGAFt[1,0], 78.429577, 5)
#        self.assertAlmostEqual(eGAFt[1,1], 74.92749546, 6)
#        self.assertAlmostEqual(eGAFt[1,2], 10.76602524, 6)
#        # Shut time. Exact solution.
#        eGAFt = qml.eGAF(t, self.tres, Feigvals, FZ00, FZ10, FZ11, Froots,
#            FR, self.mec.QFA, expQAA)
#        self.assertAlmostEqual(eGAFt[0,0], 1.10568526e+01, 6)
#        self.assertAlmostEqual(eGAFt[0,1], 6.29701830e-02, 6)
#        self.assertAlmostEqual(eGAFt[1,0], 8.39602440e-01, 6)
#        self.assertAlmostEqual(eGAFt[1,1], 1.42674924e+04, 4)
#        self.assertAlmostEqual(eGAFt[2,0], 5.06525702e-18, 15)
#        self.assertAlmostEqual(eGAFt[2,1], -9.02489888e-15, 12)

        filename = "./dcpyps/samples/CH82.scn"
        rec1 = dataset.SCRecord([filename], self.conc, self.tres, self.tcrit)

        # Check if burst separation is done right.
        self.assertEqual(len(rec1.bursts), 572)
        blength = rec1.get_burst_length_list()
        self.assertAlmostEqual(np.average(blength)*1000, 8.425049335, 8)
        openings = rec1.get_openings_burst_list()
        self.assertAlmostEqual(np.average(openings), 1.461538462, 8)

        opts = {}
        opts['mec'] = self.mec
        opts['conc'] = self.conc
        opts['tres'] = self.tres
        opts['tcrit'] = self.tcrit
        opts['isCHS'] = True
        opts['data'] = rec1.bursts
        rates = np.log(self.mec.theta())
        lik, theta = scl.HJClik(rates, opts)
        print 'lik=', lik
        self.assertAlmostEqual(-lik, 5265.9536156, 5)

    def test_popen(self):

        self.mec.fastBlk = False
        self.mec.KBlk = 0.01

        # POPEN CURVE CALCULATIONS
        c, pe, pi = scpl.Popen(self.mec, self.tres)
        self.assertTrue(pi[-1]>0.967 and pi[-1]<0.969)
