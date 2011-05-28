#! /usr/bin/python
"""
Maximum likelihood fit demo.
"""

import sys
import time

import numpy as np

from dcpyps import scalcslib as scl
from dcpyps import usefulib as ufl
from dcpyps import dataset
from dcpyps import io
from dcpyps import samples

# profiler 
import cProfile

def main():
    # Load demo mechanism (C&H82 numerical example).
    mec = samples.CH82()
    mec.printout(sys.stdout)

    tres = 0.0001
    tcrit = 0.004
    conc = 100e-9

    # Prepare parameter dict for simplex
    opts = {}
    opts['mec'] = mec
    opts['conc'] = conc
    opts['tres'] = tres
    opts['tcrit'] = tcrit
    opts['isCHS'] = True

    # Here should go initial guesses. Now using rate constants from example.
    rates = np.log(mec.unit_rates())

    # Load data.
    filename = "./dcpyps/samples/CH82.scn"
    ioffset, nint, calfac, header = io.scn_read_header(filename)
    tint, iampl, iprops = io.scn_read_data(filename, ioffset, nint, calfac)
    rec1 = dataset.TimeSeries(filename, header, tint, iampl, iprops)

    # Impose resolution, get open/shut times and bursts.
    rec1.impose_resolution(tres)
    rec1.get_open_shut_periods()
    rec1.get_bursts(tcrit)
    print('\nNumber of bursts = {0:d}'.format(len(rec1.bursts)))
    blength = rec1.get_burst_length_list()
    print('Average = {0:.3f} millisec'.format(np.average(blength)))
    print('Range: {0:.3f}'.format(min(blength)) +
            ' to {0:.3f} millisec'.format(max(blength)))
    #rec1.print_bursts()

    # Maximum likelihood fit.
    print ("\nFitting started: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])
    newrates, loglik = ufl.simplexHJC(rates, rec1.bursts, scl.HJClik,
        opts, verbose=0)
    print ("\nFitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
            %time.localtime()[0:6])
    mec.set_rateconstants(newrates)
    print "\n Final rate constants:"
    mec.printout(sys.stdout)

try:
    cProfile.run('main()')
except KeyboardInterrupt:
    pass