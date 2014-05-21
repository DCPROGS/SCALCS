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
from dcpyps import dcio
from dcpyps import dataset
from dcpyps import scalcslib as scl
from dcpyps import mechanism

from dcprogs.likelihood import Log10Likelihood

def main():

    # LOAD MECHANISM USED IN COLQUHOUN et al 2003.
    mecfn = "./dcpyps/samples/mec/demomec.mec"
    version, meclist, max_mecnum = dcio.mec_get_list(mecfn)
    mec = dcio.mec_load(mecfn, meclist[1][0])

    mec.printout(sys.stdout)
    tres = 0.000025
    tcrit = 0.0035
    conc = 50e-9

    # LOAD DATA.
    filename = "./dcpyps/samples/scn/AChsim.scn"
    rec1 = dataset.SCRecord([filename], conc, tres, tcrit)
    rec1.record_type = 'recorded'
    rec1.printout()

    # PREPARE RATE CONSTANTS.
    # Fixed rates.
    fixed = np.array([False, False, False, False, False, False, False, True,
        False, False, False, False, False, False])
    if fixed.size == len(mec.Rates):
        for i in range(len(mec.Rates)):
            mec.Rates[i].fixed = fixed[i]

    # Constrained rates.
    mec.Rates[6].is_constrained = True
    mec.Rates[6].constrain_func = mechanism.constrain_rate_multiple
    mec.Rates[6].constrain_args = [10, 1]
    mec.Rates[8].is_constrained = True
    mec.Rates[8].constrain_func = mechanism.constrain_rate_multiple
    mec.Rates[8].constrain_args = [12, 1]
    mec.Rates[9].is_constrained = True
    mec.Rates[9].constrain_func = mechanism.constrain_rate_multiple
    mec.Rates[9].constrain_args = [13, 1]
    mec.update_constrains()

    #mec.Rates[11].mr=True
    mec.set_mr(True, 11)
    mec.update_mr()

    # Initial guesses. Now using rate constants from numerical example.
    rates = np.log(mec.unit_rates())
    mec.set_rateconstants(np.exp(rates))
    mec.printout(sys.stdout)
    theta = mec.theta()
    print '\n\ntheta=', theta

    # Prepare parameter dict for simplex
    opts = {}
    opts['mec'] = mec
    opts['conc'] = conc
    opts['tres'] = tres
    opts['tcrit'] = tcrit
    opts['isCHS'] = True
    opts['data'] = rec1.bursts.intervals()

#################################################

    start_lik, th = scl.HJClik(np.log(theta), opts)
    print ("Starting likelihood = {0:.6f}".format(-start_lik))#

    bursts = rec1.bursts.intervals()
    likelihood = Log10Likelihood(bursts, mec.kA, tres, tcrit)

    def dcprogslik(x, args=None):
        mec.theta_unsqueeze(np.exp(x))
        mec.set_eff('c', opts['conc'])
        return -likelihood(mec.Q) * math.log(10), np.log(mec.theta())
    
    def dcprogslik1(x, args=None):
        mec.theta_unsqueeze(np.exp(x))
        mec.set_eff('c', opts['conc'])
        return -likelihood(mec.Q) * math.log(10)

#### testing DC-Pyps simplex
    start = time.clock()
    xout, fout, niter, neval = optimize.simplex(dcprogslik,
        np.log(theta), display=True)
    print ("\nDCPROGS Fitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])
    t1 = time.clock() - start
    mec.theta_unsqueeze(np.exp(xout))
    print "\n Final rate constants:"
    mec.printout(sys.stdout)
    print ('\n Number of iterations = {0:d}'.format(niter))
    print ('\n Number of evaluations = {0:d}'.format(neval))

#### testing SciPy optimize.minimize
    start = time.clock()
    res = minimize(dcprogslik1, np.log(theta), method='Powell')
#    res = minimize(dcprogslik1, res.x, method='Nelder-Mead')
    
    print ("\nDCPROGS Fitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])
    t2 = time.clock() - start
    mec.theta_unsqueeze(np.exp(res.x))
    print "\n Final rate constants:"
    mec.printout(sys.stdout)
    theta = mec.theta()
    lik2, th = scl.HJClik(np.log(theta), opts)

    print '\n\nresult='
    print res
    
    print '\n\n\ntime in DC-Pyps simplex=', t1
    print ('\n Final log-likelihood = {0:.6f}'.format(-fout))
    print '\ntime in SciPy simplex=', t2
    print ("\n likelihood = {0:.6f}".format(-lik2))
    
try:
    cProfile.run('main()')
except KeyboardInterrupt:
    pass
