#! /usr/bin/python
"""
Maximum likelihood fit demo.
"""

import sys
import time
import math
import numpy as np

from dcpyps import optimize
from dcpyps import samples
from dcpyps import dataset
from dcpyps import mechanism
from dcpyps.reports import FitReportHTML
from dcprogs.likelihood import Log10Likelihood

report = FitReportHTML()

print('\n\nTesting single channel data:')
# LOAD DEMO MECHANISM (C&H82 numerical example).
mec = samples.CH82()
tres = 0.0001
tcrit = 0.004
conc = 100e-9
chs = True

# LOAD DATA.
filename = ["./dcpyps/samples/scn/CH82.scn"]
rec1 = dataset.SCRecord(filename, conc, tres, tcrit)
rec1.record_type = 'recorded'
rec1.printout()
report.dataset([rec1])

# PREPARE RATE CONSTANTS.
# Fixed rates.
mec.Rates[7].fixed = True
# Constrained rates.
mec.Rates[5].is_constrained = True
mec.Rates[5].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[5].constrain_args = [4, 2]
mec.Rates[6].is_constrained = True
mec.Rates[6].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[6].constrain_args = [8, 2]
mec.update_constrains()

# Initial guesses. Now using rate constants from numerical example.
rates = mec.unit_rates()
rates = [100, 3000, 10000, 100, 1000, 1000, 1e+7, 5e+7, 6e+7, 10]
mec.set_rateconstants(rates)
mec.update_mr()
mec.printout(sys.stdout)
theta = mec.theta()
print '\ntheta=', theta

# Prepare parameter dict for DC-Pyps simplex
opts = {}
opts['mec'] = mec
opts['conc'] = conc
opts['tres'] = tres
opts['tcrit'] = tcrit
opts['isCHS'] = True
opts['data'] = rec1.bursts

#######   DCPROGS likelihood
likelihood = Log10Likelihood(rec1.bursts, mec.kA, tres, tcrit)
def dcprogslik(x, args=None):
    mec.theta_unsqueeze(np.exp(x))
    mec.set_eff('c', conc)
    return -likelihood(mec.Q) * math.log(10), np.log(mec.theta())

start_lik, th = dcprogslik(np.log(theta))
print ("Starting likelihood = {0:.6f}".format(-start_lik))
report.rates(mec, start_lik)

start = time.clock()
xout, lik, niter, neval = optimize.simplex(dcprogslik,
    np.log(theta), args=opts, display=True)
end = time.clock() - start
print ("\nFitting with DCPROGS likelihood finished: %4d/%02d/%02d %02d:%02d:%02d\n"
        %time.localtime()[0:6])
print 'time in simplex=', end
lik2 = -lik
print ('\n Final log-likelihood = {0:.6f}'.format(lik2))
print ('\n Number of iterations = {0:d}'.format(niter))
print 'xout', xout
mec.theta_unsqueeze(np.exp(xout))
print "\n Final rate constants:"
report.fit_result(mec, start, end, lik, niter)

mec.printout(sys.stdout)
report.rates(mec, lik, True)
report.distributions(mec, [rec1])
report.finalise()



