#!/usr/bin/python
"""
Maximum likelihood fit demo.
"""
import argparse
import sys
import time
import math
import numpy as np
import cProfile
from scipy.optimize import minimize

from dcpyps import dcio
from dcpyps import dataset
from dcpyps import scalcslib as scl
from dcpyps import mechanism

from dcprogs.likelihood import Log10Likelihood

def main(argv=None):

    if argv is None:
        argv=sys.argv[1:]

    p = argparse.ArgumentParser(description="Example of using argparse")

    p.add_argument('--filename', action='store', default='./dcpyps/samples/test_1.scn', help="first word")
    p.add_argument('--tcrit', action='store', default='0.0035', help="t_crit")

    p.add_argument('--tres', action='store', default='0.000025', help="t_res")

    p.add_argument('--conc', action='store', default='3e-8', help="conc")
    # Parse command line arguments
    args = p.parse_args(argv)

    print "Running dc-pyps tests for filename = " + args.filename, "and t_crit " ,  args.tcrit, "\n"

    # LOAD MECHANISM USED IN FIGURE 2 COLQUHOUN et al 2003.
    mecfn = "./dcpyps/samples/MiloneDes.mec"
    version, meclist, max_mecnum = dcio.mec_get_list(mecfn)
    mec = dcio.mec_load(mecfn, meclist[0][0])

    filename = args.filename
    tcrit = float(args.tcrit)
    tres = float(args.tres)
    conc = float(args.conc)

    # LOAD DATA.
    rec1 = dataset.SCRecord([filename], conc, tres, tcrit)
    rec1.printout()

    # PREPARE RATE CONSTANTS.
    # Fixed rates.
    fixed = np.array([False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False,False])
    if fixed.size == len(mec.Rates):
        for i in range(len(mec.Rates)):
            mec.Rates[i].fixed = fixed[i]

    # Constrained rates.
    mec.set_mr(True, 15)

    # Initial guesses. Now using rate constants from numerical example.
    #rates = np.log(mec.unit_rates())
    #mec.set_rateconstants(np.exp(rates))
    #use rates from True 1, 2003 paper
    rates = [2000.0000000000000000, 52000.0000000000000000, 6000.0000000000000000, 50.0000000000000000, 50000.0000000000000000, 150.0000000000000000, 5.0000000000000000, 1.4, 1500.0000000000000000, 200000000.0000000000000000, 10000.0000000000000000, 400000000.0000000000000000, 1500.0000000000000000, 200000000.0000000000000000, 30.0000000000000000, 400000000.0000000000000000 ];
    mec.set_rateconstants(rates)
    mec.update_constrains()
    mec.update_mr()

    theta = mec.theta()
#    mec.theta_unsqueeze(theta)
    np.set_printoptions(precision=15)
    print(mec)
    print("Q-matrix\n")
    print(mec.Q)

    bursts = rec1.bursts
    likelihood = Log10Likelihood(bursts, mec.kA, tres, tcrit)

    def dcprogslik1(x, args=None):
        mec.theta_unsqueeze(x)
        mec.set_eff('c', conc)
        return -likelihood(mec.Q) * math.log(10)

    print ("\nStarting likelihood = {0:.16f}".format(dcprogslik1(theta)))

    print (mec)
if __name__=="__main__":
    sys.exit(main(sys.argv[1:]))
