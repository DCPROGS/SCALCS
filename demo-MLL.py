#! /usr/bin/python
"""
Maximum likelihood fit demo.
"""

import sys
import time
import math
import numpy as np
import cProfile
from scipy.optimize import minimize

from dcpyps import optimize
from dcpyps import samples
from dcpyps import dcio
from dcpyps import dataset
from dcpyps import scalcslib as scl
from dcpyps import mechanism

from dcprogs.likelihood import Log10Likelihood

def main():

    print('\n\nTesting single channel data:')
    # LOAD DEMO MECHANISM (C&H82 numerical example).
    mec = samples.CH82()
    #mec.printout(sys.stdout)
    tres = 0.0001
    tcrit = 0.004
    conc = 100e-9

    # LOAD DATA.
    filename = "./dcpyps/samples/scn/CH82.scn"
    rec1 = dataset.SCRecord([filename], conc, tres, tcrit)
    rec1.record_type = 'recorded'
    rec1.printout()

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
#    rates = [6.5, 14800, 3640, 362, 1220, 2440, 1e+7, 5e+8, 2.5e+8, 55]
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
    opts['data'] = rec1.bursts.intervals()

    # MAXIMUM LIKELIHOOD FIT.
    start_lik, th = scl.HJClik(np.log(theta), opts)
    print ("Starting likelihood = {0:.6f}".format(-start_lik))
    
    ############ DC-Pyps likelihood
    print ("\n\nFitting with DC-Pyps likelihood started: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])
    start = time.clock()
    xout, fout, niter, neval = optimize.simplex(scl.HJClik,
        np.log(theta), args=opts, display=True)
    t1 = time.clock() - start
    print ("\nFitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])
    print 'time in simplex=', t1
    # Display results.
    mec.theta_unsqueeze(np.exp(xout))
    print "\n Final rate constants:"
    mec.printout(sys.stdout)
    lik1 = -fout
    print ('\n Final log-likelihood = {0:.6f}'.format(lik1))
    print ('\n Number of evaluations = {0:d}'.format(neval))
    print ('\n Number of iterations = {0:d}'.format(niter))
    print '\n\n'
    
    #######   DCPROGS likelihood
    bursts = rec1.bursts.intervals()
    logfac = math.log(10)
    likelihood = Log10Likelihood(bursts, mec.kA, tres, tcrit)
    def dcprogslik(x, args=None):
        mec.theta_unsqueeze(np.exp(x))
        mec.set_eff('c', opts['conc'])
        return -likelihood(mec.Q) * logfac, np.log(mec.theta())
    
    def dcprogslik1(x, args=None):
        mec.theta_unsqueeze(np.exp(x))
        mec.set_eff('c', opts['conc'])
        return -likelihood(mec.Q) * logfac
    
    start = time.clock()
    xout, lik, niter, neval = optimize.simplex(dcprogslik,
        np.log(theta), args=opts, display=True)
    t2 = time.clock() - start
    print ("\nFitting with DCPROGS likelihood finished: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])
    print 'time in simplex=', t2
    lik2 = -lik
    print ('\n Final log-likelihood = {0:.6f}'.format(lik2))
    print ('\n Number of iterations = {0:d}'.format(niter))
    print 'xout', xout
    mec.theta_unsqueeze(np.exp(xout))
    print "\n Final rate constants:"
    mec.printout(sys.stdout)
    
    
    start = time.clock()
    res = minimize(dcprogslik1, np.log(theta), method='Powell')
    t3 = time.clock() - start
    print ("\n\n\nScyPy.minimize (Powell) Fitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])
    print 'time in ScyPy.minimize (Powell)=', t3
    print 'xout', res.x
    mec.theta_unsqueeze(np.exp(res.x))
    print "\n Final rate constants:"
    mec.printout(sys.stdout)
    lik, th = scl.HJClik(res.x, opts)
    lik3 = -lik
    print ("\n likelihood = {0:.6f}".format(-lik3))
    

try:
    cProfile.run('main()')
except KeyboardInterrupt:
    pass