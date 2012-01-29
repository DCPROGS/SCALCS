#! /usr/bin/python
"""
Maximum likelihood fit demo.
"""

import sys
import time
import numpy as np
import cProfile

from scipy import optimize as so

from dcpyps import optimize
from dcpyps import samples
from dcpyps import dcio
from dcpyps import dataset
from dcpyps import scalcslib as scl

def main():
    # Load demo mechanism (C&H82 numerical example).
    mec = samples.CH82()
    mec.printout(sys.stdout)

    tres = 0.0001
#    tres = 0.000001 # 1 microsec
    tcrit = 0.004
#    tcrit = 300
#    tcrit = 1000000
    conc = 100e-9

    

    # Load data.
    filename = "./dcpyps/samples/CH82.scn"
#    filename = "./dcpyps/samples/CO.SCN"
    ioffset, nint, calfac, header = dcio.scn_read_header(filename)
    tint, iampl, iprops = dcio.scn_read_data(filename, ioffset, nint, calfac)
    rec1 = dataset.TimeSeries(filename, header, tint, iampl, iprops)

    # Impose resolution, get open/shut times and bursts.
    rec1.impose_resolution(tres)
    rec1.get_open_shut_periods()
    rec1.get_bursts(tcrit)

    print('\nNumber of resolved intervals = {0:d}'.format(len(rec1.rtint)))
    print('\nNumber of bursts = {0:d}'.format(len(rec1.bursts)))
    blength = rec1.get_burst_length_list()
    print('Average length = {0:.9f} millisec'.format(np.average(blength)))
#    print('Range: {0:.3f}'.format(min(blength)) +
#            ' to {0:.3f} millisec'.format(max(blength)))
    #rec1.print_bursts()
    openings = rec1.get_openings_burst_list()
    print('Average number of openings= {0:.9f}'.format(np.average(openings)))

#    rec1.print_resolved_intervals()
#    rec1.print_bursts()

#    mll, rts = scl.HJClik(rates, rec1.bursts, opts)
#    print ("\nStarting likelihood = {0:.6f}\n".format(mll))

    # Here should go initial guesses. Now using rate constants from example.
    rates = np.log(mec.unit_rates())
#    rates = np.log([1000, 30000, 10000, 100, 1000, 1000, 1e+7, 5e+7, 6e+7, 10])
    mec.set_rateconstants(np.exp(rates))

    fixed = np.array([False, False, False, False, False, False, True, True, True, False])
    if fixed.size == len(mec.Rates):
        for i in range(len(mec.Rates)):
            mec.Rates[i].fixed = fixed[i]

    theta = mec.theta()
    print 'theta=', theta

#    rates = np.log([20, 50])
#    rates = np.log([10, 60, 2, 2e+06])
#    optimize.test_CHS(rates, opts)

# Prepare parameter dict for simplex
    opts = {}
    opts['mec'] = mec
    opts['conc'] = conc
    opts['tres'] = tres
    opts['tcrit'] = tcrit
    opts['isCHS'] = True

    # Maximum likelihood fit.
    print ("\nFitting started: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])

#    xopt, fopt, iter, funcalls, warnflag, allvecs = so.fmin(optimize.HJClik,
#        rates, args=(rec1.bursts, opts),
#        full_output=1, maxiter=10000, maxfun=10000, retall=1,
#        callback=optimize.printit)

    theta, fopt, neval = optimize.simplexHJC(scl.HJClik, np.log(theta), rec1.bursts, opts)


    print ("\nFitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])

#    newrates = np.exp(xopt)
#    mec.set_rateconstants(newrates)
    mec.theta_unsqueeze(np.exp(theta))
    print "\n Final rate constants:"
    mec.printout(sys.stdout)
    print ('\n Final log-likelihood = {0:.6f}'.format(-fopt))
    print ('\n Number of evaluations = {0:d}'.format(neval))
#    print ('\n {0:d} iterations and {1:d} function calls.\n'.format(iter, funcalls))
#    print 'warnflag=', warnflag
    print '\n\n'

try:
    cProfile.run('main()')
except KeyboardInterrupt:
    pass