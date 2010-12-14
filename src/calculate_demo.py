#! /usr/bin/python
"""
This script uses scalcslib module to calculate maxPopen, EC50 and
nH parameters.
"""

import matplotlib.pyplot as plt

import qmatlib as qml
import scalcslib as scl
import scplotlib as scplot
import mechanism as mec
import readmecfile as readmec

def demoQ():

    RateList = [
         mec.Rate('beta1', 15.0, 4, 1),
         mec.Rate('beta2', 15000.0, 3, 2),
         mec.Rate('alpha1', 3000.0, 1, 4),
         mec.Rate('alpha2', 500.0, 2, 3),
         mec.Rate('k(-1)', 2000.0, 4, 5),
         mec.Rate('k(-2)', 2 * 2000.0, 3, 4),
         mec.Rate('k(+1)', 2 * 5.0e07, 5, 4, 'c'),
         mec.Rate('k*(+1)', 5.0e08, 1, 2, 'c'),
         mec.Rate('k(-2)', 5.0e08, 4, 3, 'c'),
         mec.Rate('k*(-2)', 2 * 1.0 / 3.0, 2, 1),
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

if __name__ == "__main__":

    tres = 0.00004  # resolution in seconds

    # demo = True to run DC82 numerical example
    # demo = False to load a mechanism defined in mec file
    demo = True
    if demo:
        demomec = demoQ()
    else:
        mecfile, version, meclist, max_mecnum = readmec.choose_mecfile()
        print 'mecfile:', mecfile
        print 'version:', version
        mecnum, ratenum = readmec.choose_mec_from_list(meclist, max_mecnum)
        print '\nRead rate set #', ratenum + 1, 'of mec #', mecnum
        demomec = readmec.load_mec(mecfile, meclist[ratenum][0])

    #     POPEN CURVE CALCULATIONS
    print '\nCalculating Popen curve parameters:'
    # Calculate EC50, nH and maxPopen for ideal Popen curve.
    emaxPopen, econc = scl.get_maxPopen(demomec, tres)
    eEC50 = scl.get_EC50(demomec, tres)
    enH = scl.get_nH(demomec, tres)
    text2 = ('\nHJC Popen curve:\nmaxPopen={0:.3f}'.format(emaxPopen) +
        '\nEC50 = {0:.2e} M'.format(eEC50) + '\nnH = {0:.3f}'.format(enH))
    print text2

    # Calculate EC50, nH and maxPopen for Popen curve
    # corrected for missed events.
    imaxPopen, iconc = scl.get_maxPopen(demomec, 0)
    iEC50 = scl.get_EC50(demomec, 0)
    inH = scl.get_nH(demomec, 0)
    text1 = ('\nideal Popen curve:\nmaxPopen={0:.3f}'.format(imaxPopen) +
        '\nEC50 = {0:.2e} M'.format(iEC50) + '\nnH = {0:.3f}'.format(inH))
    print text1

    # Plot ideal and corrected Popen curves.
    c_start = iEC50 * 0.01
    c_end = iEC50 *100
    plt.subplot(221)
    scplot.popen_curve(c_start, c_end, demomec, tres, text1, text2)

    #     BURST CALCULATIONS
    print '\nCalculating burst properties:'
    conc = 100e-9    # 100 nM
    #demomec.set_eff('c', conc)

    # Calculate mean burst length.
    m = qml.mean_burst_length(demomec, conc)
    print '\nmean burst length =', m
    # Plot burst length distribution
    tmin = tres
    tmax = 0.100
    plt.subplot(222)
    scplot.distr_burst_length(demomec, conc, tmin, tmax)

    # Calculate mean number of openings per burst.
    mu = qml.mean_num_burst_openings(demomec, conc)
    print 'mean number of openings per burst= ', mu
    # Plot distribution of number of openings per burst
    n = 10
    plt.subplot(223)
    scplot.distr_num_burst_openings(n, demomec, conc)

    conc_start = 1e-6    # in M
    conc_end = 1e-2    # in M
    plt.subplot(224)
    scplot.burst_length_versus_conc(demomec, conc_start, conc_end)

    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None,
        wspace=0.4, hspace=0.5)
    plt.show()


    