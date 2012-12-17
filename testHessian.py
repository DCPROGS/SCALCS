#! /usr/bin/python
"""
Test Hessian matrix calculation.
"""

import sys
import time
import numpy as np
import math

from dcpyps import optimize
from dcpyps import samples
from dcpyps import dcio
from dcpyps import dataset
from dcpyps import scalcslib as scl
from dcpyps import mechanism

def main():

    print('\nRUNNING HESSIAN MATRIX CALCULATION TEST ...')

    # LOAD DATA.
    print('\n\nLOADING DATA:')
    filename = "./dcpyps/samples/CH82.scn"
    tres = 0.0001
    tcrit = 0.004
    conc = 100e-9
    ioffset, nint, calfac, header = dcio.scn_read_header(filename)
    tint, iampl, iprops = dcio.scn_read_data(filename, ioffset, nint, calfac)
    rec1 = dataset.TimeSeries(filename, header, tint, iampl, iprops)
    # Impose resolution, get open/shut times and bursts.
    rec1.impose_resolution(tres)
    rec1.get_open_shut_periods()
    rec1.get_bursts(tcrit)
    print('Number of resolved intervals = {0:d}'.format(len(rec1.rtint)))
    print('Number of bursts = {0:d}'.format(len(rec1.bursts)))
    blength = rec1.get_burst_length_list()
    print('Average length = {0:.9f} millisec'.format(np.average(blength)))
    print('Range: {0:.3f}'.format(min(blength)) +
            ' to {0:.3f} millisec'.format(max(blength)))
    openings = rec1.get_openings_burst_list()
    print('Average number of openings= {0:.9f}'.format(np.average(openings)))


    # LOAD DEMO MECHANISM (C&H82 numerical example).
    print('\n\nLOADING MECHANISM:')
    mec = samples.CH82()
    
    # PREPARE RATE CONSTANTS.
    # Fixed rates.
    fixed = np.array([False, False, False, False, False, False, False, True, False, False])
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
    mec.update_mr()
    # Initial guesses. Now using rate constants from numerical example.
    rates = mec.unit_rates()
    mec.set_rateconstants(rates)
    mec.printout(sys.stdout)
    theta = np.log(mec.theta())
    print '\ntheta=', np.exp(theta)

    # Prepare parameter dict for simplex
    opts = {}
    opts['mec'] = mec
    opts['conc'] = conc
    opts['tres'] = tres
    opts['tcrit'] = tcrit
    opts['isCHS'] = True
    opts['data'] = rec1.bursts
    
    
    
    # ESTIMATE ERROS FOR PARAMETERS
    print('\n\nESTIMATING ERRORS FOR PARAMETERS:')
    
    # Here theta already log(rates)
    lik, theta = scl.HJClik(theta, opts)
    print ("\nStarting likelihood = {0:.6f}".format(-lik))
    
    delta = 0.1 # Decrease log(lik) by delta. 
    # TODO: Allow to change by user.
    print('\nFor Hessian use step size that decreases log(lik)' +
        'by delta = {0:.6f}'.format(delta))
    
    # In this version calculates for log rates directly. 
    #TODO: check if calculation from not log rates gives same answer.
    
    likcrit = -lik - delta
    kfit = len(theta) # Number of estimated parameters.
    
    step = np.ones((kfit)) * 0.01
    
    
    for i in range(kfit):
        print('Seeking increment in parameter {0:d} = {1:.6f}'.
            format(i+1, theta[i]))
        th1 = theta.copy()
        th1[i] = theta[i] + step[i]
        lik1, th1 = scl.HJClik(th1, opts)
        if -lik1 < likcrit:
            while -lik1 < likcrit:
                step[i] = 0.7 * step[i]
                th1[i] = theta[i] + step[i]
                lik1, th1 = scl.HJClik(th1, opts)
        elif -lik1 > likcrit:
            while -lik1 > likcrit:
                step[i] = 1.4 * step[i]
                th1[i] = theta[i] + step[i]
                lik1, th1 = scl.HJClik(th1, opts)
        p = 100.0 * (1 - th1[i] / theta[1])
        print('    Step = {0:.6f} ({1:.6f}%)'.format(step[i], p))
        
        
    # Estimate Hessian.
    
    

main()
