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

import qmatlib as qml
import scalcslib as scl
import scplotlib as scplot
import mechanism as mec
import readmecfile as readmec

def demoQ():

    RateList = [
         mec.Rate(15.0, 4, 1, name='beta1'),
         mec.Rate(15000.0, 3, 2, name='beta2'),
         mec.Rate(3000.0, 1, 4, name='alpha1'),
         mec.Rate(500.0, 2, 3, name='alpha2'),
         mec.Rate(2000.0, 4, 5, name='k(-1)'),
         mec.Rate(2 * 2000.0, 3, 4, name='k(-2)'),
         mec.Rate(2 * 5.0e07, 5, 4, name='k(+1)', eff='c'),
         mec.Rate(5.0e08, 1, 2, name='k*(+1)', eff='c'),
         mec.Rate(5.0e08, 4, 3, name='k(-2)', eff='c'),
         mec.Rate(2 * 1.0 / 3.0, 2, 1, name='k*(-2)'),
         ]

    StateList = [
        mec.State(1, 'A', 'A2R*', 60e-12),
        mec.State(2, 'A', 'AR*', 60e-12),
        mec.State(3, 'B', 'A2R', 0.0),
        mec.State(4, 'B', 'AR', 0.0),
        mec.State(5, 'C', 'R', 0.0)
        ]

    ncyc = 1
    fastblk = True
    KBlk = 0.001

    return  mec.Mechanism(RateList, StateList, ncyc, fastblk, KBlk)

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
                        version='%(prog)s 2.0', 
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
        demomec = demoQ()
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

        version, meclist, max_mecnum = readmec.get_mec_list(mecfn)
        print 'mecfile:', mecfn
        print 'version:', version
        mecnum, ratenum = readmec.choose_mec_from_list(meclist, max_mecnum)
        print '\nRead rate set #', ratenum + 1, 'of mec #', mecnum
        demomec = readmec.load_mec(mecfn, meclist[ratenum][0])

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
    fastBlk = False
    KBlk = 0.01

    conc = 100e-9    # 100 nM
    cmin = 100e-9
    cmax = 0.01
    tmin = tres
    tmax = 100

    #     POPEN CURVE CALCULATIONS
    print '\nCalculating Popen curve parameters:'
    text1, text2, c, pe, pi = scl.get_Popen_plot(demomec, tres, cmin, cmax,
        fastBlk, KBlk)
    print text1, text2
    plt.subplot(221)
    plt.semilogx(c, pe, 'b-', c, pi, 'r--')
    plt.ylabel('Popen')
    plt.xlabel('Concentration, M')
    plt.title('Apparent and ideal Popen curves')

    #     BURST CALCULATIONS
    print '\nCalculating burst properties:'
    print ('Agonist concentration = %e M' %conc)
    text1, t, fbst = scl.get_burstlen_distr(demomec, conc, tmin, tmax)
    print text1
    plt.subplot(222)
    plt.semilogx(t, fbst, 'b-')
    plt.ylabel('fbst(t)')
    plt.xlabel('burst length, s')
    plt.title('The burst length pdf')

    # Calculate mean number of openings per burst.
    text1, r, Pr = scl.get_burstopenings_distr(demomec, conc)
    print text1
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
