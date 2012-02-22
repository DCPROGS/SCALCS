#! /usr/bin/python
"""
Maximum likelihood fit demo.
"""

import sys
import time
import numpy as np
import cProfile

from dcpyps import optimize
from dcpyps import samples
from dcpyps import dcio
from dcpyps import dataset
from dcpyps import scalcslib as scl
from dcpyps import mechanism


def rosen(x, data=None, args=None):
    """The Rosenbrock function"""
    f = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return f, x


def main():

    # Test with Rosenbrock function minimisation.
    print('\nTesting Rosenbrock function minimisation:')
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    xout, fout, niter, neval = optimize.simplex(rosen, x0, args=None)
    print xout, fout, niter, neval
    print('\nFirst test finished.')


    print('\n\nTesting single channel data:')
    # LOAD DEMO MECHANISM (C&H82 numerical example).
    mec = samples.CH82()
    mec.printout(sys.stdout)
    tres = 0.0001
    tcrit = 0.004
    conc = 100e-9

    # LOAD DATA.
    filename = "./dcpyps/samples/CH82.scn"
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
    openings = rec1.get_openings_burst_list()
    print('Average number of openings= {0:.9f}'.format(np.average(openings)))

    # PREPARE RATE CONSTANTS.
    # Fixed rates.
    fixed = np.array([False, False, False, False, False, False, False, False, True, False])
    if fixed.size == len(mec.Rates):
        for i in range(len(mec.Rates)):
            mec.Rates[i].fixed = fixed[i]
    # Constrained rates.
    mec.Rates[5].is_constrained = True
    mec.Rates[5].constrain_func = mechanism.constrain_rate_multiple
    mec.Rates[5].constrain_args = [4, 2]
    mec.Rates[6].is_constrained = True
    mec.Rates[6].constrain_func = mechanism.constrain_rate_multiple
    mec.Rates[6].constrain_args = [8, 2]
    mec.update_constrains()
    # Initial guesses. Now using rate constants from numerical example.
    rates = np.log(mec.unit_rates())
#    rates = np.log([100, 3000, 10000, 100, 1000, 1000, 1e+7, 5e+7, 6e+7, 10])
#    rates = np.log([6.5, 14800, 3640, 362, 1220, 2440, 1e+7, 5e+8, 2.5e+8, 55])
    mec.set_rateconstants(np.exp(rates))
    mec.printout(sys.stdout)
    theta = mec.theta()
    print 'theta=', theta

    # Prepare parameter dict for simplex
    opts = {}
    opts['mec'] = mec
    opts['conc'] = conc
    opts['tres'] = tres
    opts['tcrit'] = tcrit
    opts['isCHS'] = True
    opts['data'] = rec1.bursts

    # MAXIMUM LIKELIHOOD FIT.
    print ("\nFitting started: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])
    #xout, fopt, neval, niter = optimize.simplexHJC(scl.HJClik,
    #    np.log(theta), data=rec1.bursts, args=opts)
    xout, fout, niter, neval = optimize.simplex(scl.HJClik,
        np.log(theta), args=opts, display=True)
    print ("\nFitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])
    # Display results.
    mec.theta_unsqueeze(np.exp(xout))
    print "\n Final rate constants:"
    mec.printout(sys.stdout)
    print ('\n Final log-likelihood = {0:.6f}'.format(-fout))
    print ('\n Number of evaluations = {0:d}'.format(neval))
    print ('\n Number of iterations = {0:d}'.format(niter))
    print '\n\n'

try:
    cProfile.run('main()')
except KeyboardInterrupt:
    pass