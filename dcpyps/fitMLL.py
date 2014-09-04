"""A collection of some MLL (maximum log likelihood) fitting related functions."""

import sys
import time
import numpy as np
from scipy.optimize import minimize
from math import*
import numpy as np

from dcpyps.reports import FitReportHTML
from dcprogs.likelihood import Log10Likelihood

class FittingSession():
    """
    """


    def __init__(self, mec, recs):
        self.mec = mec
        self.recs = recs
        self.LL = []
        kwargs = {'nmax': 2, 'xtol': 1e-12, 'rtol': 1e-12, 'itermax': 100,
            'lower_bound': -1e6, 'upper_bound': 0}
        print self.recs[0].bursts.intervals()
        for i in range(len(self.recs)):
            self.LL.append(Log10Likelihood(self.recs[i].bursts.intervals(), self.mec.kA,
                self.recs[i].tres, self.recs[i].tcrit, **kwargs))
        self.report = FitReportHTML()
        self.report.dataset(self.recs)
        start_lik = self.dcprogslik(np.log(self.mec.theta()))
        print ("Starting likelihood = {0:.6f}".format(-start_lik))
        self.report.rates(self.mec, start_lik)
        self.iternum = 0

    def likelihood(self):
        
        for i in range(len(self.recs)):
            self.LL.append(Log10Likelihood(self.recs[i].bursts, self.mec.kA,
                self.recs[i].tres, self.recs[i].tcrit, **self.kwargs))

    def run(self):
        # FITTING BIT
        start = time.clock()
        result = minimize(self.dcprogslik, np.log(self.mec.theta()),
            method='Nelder-Mead', callback=self.printiter,
            options={'xtol':1e-4, 'ftol':1e-4, 'maxiter': 5000, 'maxfev': 10000,
            'disp': True})
        end = time.clock() - start
        # FITTING BIT
        print ("\n\nFitting with DCPROGS likelihood finished: %4d/%02d/%02d %02d:%02d:%02d"
            %time.localtime()[0:6])
        print '\n Time in simplex=', end
        print ('\n Final log-likelihood = {0:.6f}'.format(-result.fun))
        print ('\n Number of iterations = {0:d}'.format(result.nit))
        self.mec.theta_unsqueeze(np.exp(result.x))
        print "\n Final rate constants:"
        self.mec.printout(sys.stdout)

        self.report.fit_result(self.mec, start, end, result.fun, result.nit)
        self.report.rates(self.mec, result.fun, True)
        self.report.distributions(self.mec, self.recs)
        self.report.finalise()

    def printiter(self, theta):
        self.iternum += 1
        lik = self.dcprogslik(theta)
        if self.iternum % 10 == 0:
            print("iteration # {0:d}; log-lik = {1:.6f}".format(self.iternum, -lik))
            print(np.exp(theta))

    def dcprogslik(self, x, args=None):
        self.mec.theta_unsqueeze(np.exp(x))
        lik = 0
        for i in range(len(self.recs)):
            self.mec.set_eff('c', self.recs[i].conc)
            lik += -self.LL[i](self.mec.Q) * log(10)
        return lik

    def dcprogslik_all(x, args=None):
        """
        Return a list of separate likelihoods for each patch.
        """
        self.mec.theta_unsqueeze(np.exp(x))
        lik = []
        for i in range(len(conc)):
            self.mec.set_eff('c', self.recs[i].conc)
            lik.append(-self.LL[i](self.mec.Q) * log(10))
        return lik

# end of FittingSession #
