#! /usr/bin/env python
"""
This script uses scalcslib module to calculate maxPopen, EC50 and
nH parameters.
"""

try:
    import matplotlib.pyplot as plt
except:
    raise ImportError("matplotlib module is missing")
import argparse
import sys
import os

from scalcs import scalcsio
from samples import samples
from scalcs import version
from scalcs import scplotlib as scpl
from scalcs import popen, scburst
from scalcs.scalcslib import QMatrixPrints, TCritPrints, ExactPDFPrints, AsymptoticPDFPrints

def create_parser():
    parser = argparse.ArgumentParser(
        description='SCALCS %s demo program.' % (version.full_version))
    parser.add_argument('-f', '--file', action='store', nargs=1, dest='file',
                        help='mechanism file (optional); ' \
                             'will use demo sample if not provided')
    parser.add_argument('-d', '--demo', action='store_true', dest='demo',
                        default=False,
                        help='mechanism file')
    parser.add_argument('-v', '--version', action='version', 
                        version='SCALCS %s' %(version.full_version), 
                        help='print version information')
    
    return parser

def process_args(args):
    # demo = True to run DC82 numerical example
    # demo = False to load a mechanism defined in mec file
    mecfn = None
    if args.demo:
    # Case 1: User chooses demo mode. In that case, everything
    #         else is discarded.
        if args.file is not None:
            sys.stdout.write('Running in demo mode. Ignoring mec file.\n')
        demomec = samples.CH82()
    else:
    # Case 2: Not in demo mode.
        if args.file is not None:
        # Case 2a: User supplies a file on the command line 
            mecfn = args.file[0]        
        else:
            sys.stderr.write("No file provided, couldn't load file dialog, " \
                                "reverting to demo mode\n")
            #sys.exit(1)
            demomec = samples.CH82()

    if mecfn:
        # Check whether the file is available:
        if not os.path.exists(mecfn):
            sys.stderr.write("Couldn't find file %s. Exiting now.\n" % mecfn)
            sys.exit(1)

        version, meclist, max_mecnum = scalcsio.mec_get_list(mecfn)
        sys.stdout.write('mecfile: %s\n' % mecfn)
        sys.stdout.write('version: %s\n' % version)
        mecnum, ratenum = scalcsio.mec_choose_from_list(meclist, max_mecnum)
        sys.stdout.write('\nRead rate set #%d of mec #%d\n' % (ratenum+1, mecnum))
        demomec = scalcsio.mec_load(mecfn, meclist[ratenum][0])

    return demomec

def console_demo(demomec):

    sys.stdout.write('%s' % demomec)

    tres = 0.0001  # resolution in seconds
    demomec.fastBlk = False
    demomec.KBlk = 0.01
    conc = 100e-9    # 100 nM

    #     POPEN CURVE CALCULATIONS
    sys.stdout.write('\n\nCalculating Popen curve parameters:')
    sys.stdout.write(popen.printout(demomec, tres))
    c, pe, pi = popen.calculate_Popen_plot(demomec, tres)

    plt.subplot(221)
    plt.semilogx(c*1e6, pe, 'b-', c*1e6, pi, 'r--')
    plt.ylabel('Popen')
    plt.xlabel('Concentration, uM')
    plt.title('Apparent and ideal Popen curves')

    demomec.set_eff('c', conc)

    #     BURST CALCULATIONS
    sys.stdout.write('\n\nCalculating burst properties:')
    sys.stdout.write('\nAgonist concentration = %e M' %conc)    
    # Calculating burst parameters
    q_burst = scburst.BurstDisplay(demomec)
    print(q_burst.print_all)

    t, fbst = q_burst.calculate_burst_length_pdf()
    plt.subplot(222)
    plt.semilogx(t*1000, fbst, 'b-')
    plt.ylabel('fbst(t)')
    plt.xlabel('burst length, ms')
    plt.title('The burst length pdf')

#    # Calculate mean number of openings per burst.
#    r, Pr = scpl.get_burstopenings_distr(demomec, conc)
#    # Plot distribution of number of openings per burst
#    plt.subplot(223)
#    plt.plot(r, Pr,'ro')
#    plt.ylabel('Pr')
#    plt.xlabel('Openings per burst')
#    plt.title('Openings per burst')
#    plt.axis([0, max(r)+1, 0, 1])

    #     OPEN TIME DISTRIBUTION
    sys.stdout.write('\n\nCalculating open and shut time distributions:')
    q_matrix = QMatrixPrints(demomec) 
    sys.stdout.write(q_matrix.print_DC_table)

    q_asymp = AsymptoticPDFPrints(demomec, tres)
    q_exact = ExactPDFPrints(demomec, tres)
    sys.stdout.write(q_matrix.print_open_time_pdf)
    sys.stdout.write(q_asymp.print_asymptotic_open_time_pdf)
    sys.stdout.write(q_exact.open_time_pdf)
    sys.stdout.write(q_matrix.print_shut_time_pdf)
    sys.stdout.write(q_asymp.print_asymptotic_shut_time_pdf)
    sys.stdout.write(q_exact.shut_time_pdf)

    tcrits = TCritPrints(demomec)
    sys.stdout.write(tcrits.print_all)
    t, ipdf, epdf, apdf = scpl.open_time_pdf(demomec, tres)

    plt.subplot(223)
    plt.semilogx(t, ipdf, 'r--', t, epdf, 'b-', t, apdf, 'g-')
    plt.ylabel('fopen(t)')
    plt.xlabel('Open time, ms')
    plt.title('The open time pdf')

    #     SHUT TIME DISTRIBUTION
    t, ipdf, epdf, apdf = scpl.shut_time_pdf(demomec, tres)
    plt.subplot(224)
    plt.semilogx(t, ipdf, 'r--', t, epdf, 'b-', t, apdf, 'g-')
    plt.ylabel('fshut(t)')
    plt.xlabel('Shut time, ms')
    plt.title('The shut time pdf')

    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None,
        wspace=0.4, hspace=0.5)
    plt.show()
    
if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    demomec = process_args(args)

    console_demo(demomec)
