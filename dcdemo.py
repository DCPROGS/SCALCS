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
try:
    import Tkinter as Tk
    import tkFileDialog
    HASTK = True
except:
    HASTK = False

from dcpyps import scalcslib as scl
from dcpyps import scplotlib as scpl
from dcpyps import dcio
from dcpyps import samples
from dcpyps import popen
from dcpyps import scburst
from dcpyps import version

def create_parser():
    parser = argparse.ArgumentParser(
        description='DC_PyPs %s demo program.' % (version.full_version))
    parser.add_argument('-f', '--file', action='store', nargs=1, dest='file',
                        help='mechanism file (optional); ' \
                             'will use demo sample if not provided')
    parser.add_argument('-d', '--demo', action='store_true', dest='demo',
                        default=False,
                        help='mechanism file')
    parser.add_argument('-v', '--version', action='version', 
                        version='DC_PyPs %s' %(version.full_version), 
                        help='print version information')
    
    return parser

def process_args(args):
    # demo = True to run DC82 numerical example
    # demo = False to load a mechanism defined in mec file
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
        # Case 2b: No file provided. Attempt to get file name from dialog.
            if HASTK:
                mecfn = file_dialog()
            else:
                sys.stderr.write("No file provided, couldn't load file dialog, " \
                                 "aborting now\n")
                sys.exit(1)

        # Check whether the file is available:
        if not os.path.exists(mecfn):
            sys.stderr.write("Couldn't find file %s. Exiting now.\n" % mecfn)
            sys.exit(1)

        version, meclist, max_mecnum = dcio.mec_get_list(mecfn)
        sys.stdout.write('mecfile: %s\n' % mecfn)
        sys.stdout.write('version: %s\n' % version)
        mecnum, ratenum = dcio.mec_choose_from_list(meclist, max_mecnum)
        sys.stdout.write('\nRead rate set #%d of mec #%d\n' % (ratenum+1, mecnum))
        demomec = dcio.mec_load(mecfn, meclist[ratenum][0])

    return demomec

def file_dialog():
    """
    Choose mec file to read.

    Returns
    -------
    mecfile : filename
    """

    root = Tk.Tk()
    mecfile = tkFileDialog.askopenfilename(
        initialdir='',
        filetypes=[("DC mec", "*.mec"),("DC mec", "*.MEC"),
                   ("all files", "*")])
    root.destroy()
    
    return mecfile

def console_demo(demomec):

    sys.stdout.write('%s' % demomec)

    tres = 0.0001  # resolution in seconds
    demomec.fastBlk = False
    demomec.KBlk = 0.01
    conc = 100e-9    # 100 nM

    #     POPEN CURVE CALCULATIONS
    sys.stdout.write('\n\nCalculating Popen curve parameters:')
    popen.printout(demomec, tres)
    c, pe, pi = scpl.Popen(demomec, tres)

    plt.subplot(221)
    plt.semilogx(c, pe, 'b-', c, pi, 'r--')
    plt.ylabel('Popen')
    plt.xlabel('Concentration, M')
    plt.title('Apparent and ideal Popen curves')

    demomec.set_eff('c', conc)

    #     BURST CALCULATIONS
    sys.stdout.write('\n\nCalculating burst properties:')
    sys.stdout.write('\nAgonist concentration = %e M' %conc)
    scburst.printout(demomec)

    t, fbst = scpl.burst_length_pdf(demomec)
    plt.subplot(222)
    plt.semilogx(t, fbst, 'b-')
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
    scl.printout_occupancies(demomec)
    scl.printout_distributions(demomec, tres)
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
