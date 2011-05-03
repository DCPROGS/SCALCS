#! /usr/bin/python
"""
Maximum likelihood fit demo.
"""

import sys

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

    # Here should go initial guesses. Nau using rate constants from example.
    rates = mec.rates

    # Load data.
    filename = "./dcpyps/samples/CH82.scn"
    ioffset, nint, calfac, header = io.scn_read_header(filename)
    tint, iampl, iprops = io.scn_read_data(filename, ioffset, nint, calfac)
    rec1 = dataset.TimeSeries(filename, header, tint, iampl, iprops)

    # Impose resolution, get open/shut times and bursts.
    rec1.impose_resolution(tres)
    rec1.get_open_shut_periods()
    rec1.get_bursts(tcrit)

    # Maximum likelihood fit.

    newrates, loglik = ufl.simplex(rates, rec1.bursts, scl.HJClik,
        opts, verbose=0)
    mec.rates = newrates
    print "\n Final rate constants:"
    mec.printout(sys.stdout)

cProfile.run('main()')
