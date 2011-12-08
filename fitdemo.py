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

def main():
    # Load demo mechanism (C&H82 numerical example).
    mec = samples.CO()
    mec.printout(sys.stdout)

    tres = 0.0001
    tres = 0.000001 # 1 microsec
    tcrit = 0.004
    tcrit = 1000000
    conc = 100e-9

    # Prepare parameter dict for simplex
    opts = {}
    opts['mec'] = mec
    opts['conc'] = conc
    opts['tres'] = tres
    opts['tcrit'] = tcrit
    opts['isCHS'] = False # True

    # Here should go initial guesses. Now using rate constants from example.
    rates = np.log(mec.unit_rates())

    #rates = [1000, 30000, 10000, 100, 1000, 1000, 1e+7, 5e+7, 6e+7, 10]
    rates = [20, 50]

#    optimize.test_CHS(rates, opts)

    # Load data.
    #filename = "./dcpyps/samples/CH82.scn"
    filename = "./dcpyps/samples/CO.SCN"
    ioffset, nint, calfac, header = dcio.scn_read_header(filename)
    tint, iampl, iprops = dcio.scn_read_data(filename, ioffset, nint, calfac)
    rec1 = dataset.TimeSeries(filename, header, tint, iampl, iprops)

    # Impose resolution, get open/shut times and bursts.
    rec1.impose_resolution(tres)
    rec1.get_open_shut_periods()
    rec1.get_bursts(tcrit)
    print('\nNumber of bursts = {0:d}'.format(len(rec1.bursts)))
    blength = rec1.get_burst_length_list()
    print('Average = {0:.3f} millisec'.format(np.average(blength)))
#    print('Range: {0:.3f}'.format(min(blength)) +
#            ' to {0:.3f} millisec'.format(max(blength)))
    #rec1.print_bursts()

    # Maximum likelihood fit.
    print ("\nFitting started: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])

    xopt, fopt, iter, funcalls, warnflag, allvecs = so.fmin(optimize.HJClik,
        rates, args=(rec1.bursts, opts),
        full_output=1, maxiter=10000, maxfun=10000, retall=1,
        callback=optimize.printit)

#    newrates, loglik = optimize.simplexHJC(rates, rec1.bursts, optimize.HJClik,
#        opts, verbose=0)
    print ("\nFitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])


    newrates = np.exp(xopt)
    mec.set_rateconstants(newrates)
    print "\n Final rate constants:"
    mec.printout(sys.stdout)
    print ('\n Final log-likelihood = {0:.6f}'.format(-fopt))
    print ('\n {0:d} iterations and {1:d} function calls.\n'.format(iter, funcalls))
    print 'warnflag=', warnflag
    print '\n\n'

try:
    cProfile.run('main()')
except KeyboardInterrupt:
    pass