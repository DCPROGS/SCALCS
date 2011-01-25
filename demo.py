#! /usr/bin/python
"""
This script uses scalcslib module to calculate maxPopen, EC50 and
nH parameters.
"""

DCMAJOR = 0
DCMINOR = 1

import matplotlib.pyplot as plt
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
from dcpyps import io
from dcpyps import samples

def create_parser():
    parser = argparse.ArgumentParser(
        description='DC_PyPs %d.%d demo program.' %(DCMAJOR, DCMINOR))
    parser.add_argument('-f', '--file', action='store', nargs=1, dest='file',
                        help='mechanism file (optional); ' \
                             'will use demo sample if not provided')
    parser.add_argument('-d', '--demo', action='store_true', dest='demo',
                        default=False,
                        help='mechanism file')
    parser.add_argument('-v', '--version', action='version', 
                        version='DC_PyPs %d.%d' %(DCMAJOR,DCMINOR), 
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

        version, meclist, max_mecnum = io.get_mec_list(mecfn)
        sys.stdout.write('mecfile: %s\n' % mecfn)
        sys.stdout.write('version: %s\n' % version)
        mecnum, ratenum = io.choose_mec_from_list(meclist, max_mecnum)
        sys.stdout.write('\nRead rate set #%d of mec #%d\n' % (ratenum+1, mecnum))
        demomec = io.load_mec(mecfn, meclist[ratenum][0])

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
        initialdir='/home/remis/pDC/data',
        filetypes=[("DC mec", "*.mec"),("DC mec", "*.MEC"),
                   ("all files", "*")])
    root.destroy()
    
    return mecfile
    
if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    demomec = process_args(args)

    tres = 0.00004  # resolution in seconds
    demomec.fastBlk = False
    demomec.KBlk = 0.01

    conc = 100e-9    # 100 nM
    cmin = 100e-9
    cmax = 0.01
    tmin = tres
    tmax = 100

    #     POPEN CURVE CALCULATIONS
    sys.stdout.write('\nCalculating Popen curve parameters:\n')
    text1, text2, c, pe, pi = scl.get_Popen_plot(demomec, tres, cmin, cmax)

    sys.stdout.write(text1 + text2 + '\n')
    plt.subplot(221)
    plt.semilogx(c, pe, 'b-', c, pi, 'r--')
    plt.ylabel('Popen')
    plt.xlabel('Concentration, M')
    plt.title('Apparent and ideal Popen curves')

    #     BURST CALCULATIONS
    sys.stdout.write('\nCalculating burst properties:\n')
    sys.stdout.write('Agonist concentration = %e M\n' %conc)
    text1, t, fbst = scl.get_burstlen_pdf(demomec, conc, tmin, tmax)
    sys.stdout.write(text1 + '\n')
    plt.subplot(222)
    plt.semilogx(t, fbst, 'b-')
    plt.ylabel('fbst(t)')
    plt.xlabel('burst length, s')
    plt.title('The burst length pdf')

    # Calculate mean number of openings per burst.
    text1, r, Pr = scl.get_burstopenings_distr(demomec, conc)
    sys.stdout.write(text1 + '\n')
    # Plot distribution of number of openings per burst
    plt.subplot(223)
    plt.plot(r, Pr,'ro')
    plt.ylabel('Pr')
    plt.xlabel('Openings per burst')
    plt.title('Openings per burst')
    plt.axis([0, max(r)+1, 0, 1])

    cmin = 1e-6    # in M
    cmax = 1e-2    # in M
    c, b = scl.get_burstlen_conc_plot(demomec, cmin, cmax)
    plt.subplot(224)
    #if mec.fastblk:
    #    plt.plot(c, b, 'b-', c, blk, 'g-')
    #else:
    plt.plot(c, b, 'b-')
    plt.ylabel('Mean burst length, ms')
    plt.xlabel('Concentration, mM')
    plt.title('Mean burst length')

    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None,
        wspace=0.4, hspace=0.5)
    plt.show()
