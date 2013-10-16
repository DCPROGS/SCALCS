#! /usr/bin/python

import sys
import math
import random

import numpy as np
from scipy.special import erf

import dcio
import scalcslib as scl

class SCRecord(object):
    """
    A wrapper over a list of time intervals from idealised single channel
    record.
    """

    def __init__(self, filenames=None, conc=None, tres=None, tcrit=None,
        chs=None, onechan=None, badend=None,
        itint=None, iampl=None, iprops=None):
        
        self.filenames = filenames
        self.itint = itint
        self.iampl = iampl
        self.iprops = iprops
        self.tres = tres
        self.tcrit = tcrit
        self.conc = conc
        self.chs = chs # CHS vectors: yes or no
        self.onechan = onechan # opening from one channel only?
        self.badend = badend # bad shutting can terminate burst?

        self.record_type = None
        self.resolution_imposed = False
        
        if self.filenames and self.tres and self.tcrit:
            self.load_from_file()
            self.impose_resolution(self.tres)
            self.get_open_shut_periods()
            self.get_bursts(self.tcrit)
        

    def load_from_file(self):
        #TODO: enable taking several scan files and join in a single record.
        # Just a single file could be loaded at present.
        ioffset, nint, calfac, header = dcio.scn_read_header(self.filenames[0])
        self.itint, self.iampl, self.iprops = dcio.scn_read_data(self.filenames[0],
            ioffset, nint, calfac)
#        self.record_type = '' # TODO: get from header if simulated or scanned

    def simulate_record(self, mec, tres, state, opamp=5, nintmax=5000):
        """
        """
        picum = np.cumsum(scl.transition_probability(mec.Q), axis=1)
        tmean = -1 / mec.Q.diagonal() # in ms
        itint = [random.expovariate(1 / tmean[state])]
        iampl = [opamp if state < mec.kA else 0]
        while len(itint) < nintmax:
            state, t, a = self.next_state(state, picum, tmean, mec.kA, opamp)
            if t < tres or a == iampl[-1]:
                itint[-1] += t
            else:
                itint.append(t)
                iampl.append(a)
        self.itint = np.array(itint, dtype=np.float)
        self.iampl = np.array(iampl, dtype=np.int16)
        self.iprops = np.zeros((len(itint)), dtype=np.int8)
        self.rtint = self.itint
        self.rampl = self.iampl
        self.rprops = self.iprops
        self.resolution_imposed = True
        self.tres = tres
        self.record_type = 'simulated locally'

    def next_state(self, present, picum, tmean, kA, opamp):
        """
        Get next state, its lifetime and amplitude.
        """
        possible = np.nonzero(picum[present] >= random.random())[0]
        next = np.delete(possible, np.where(possible == present))[0]
        t = random.expovariate(1 / tmean[next])
        a = opamp if next < kA else 0
        return next, t, a

    def print_all_record(self):
        for i in range(len(self.itint)):
            print i, self.itint[i], self.iampl[i], self.iprops[i]

    def print_resolved_intervals(self):
        print('\n#########\nList of resolved intervals:\n')
        for i in range(len(self.rtint)):
            print i, self.rtint[i], self.rampl[i], self.rprops[i]
            if (self.rampl[i] == 0) and (self.rtint > (self.tcrit)):
                print ('\n')
        print('\n###################\n\n')
            
    def impose_resolution(self, tres):
        """
        Impose time resolution.

        First interval to start has to be resolvable, usable and preceded by
        an resolvable interval too. Otherwise its start will be defined by
        unresolvable interval and so will be unreliable.
        (1) A concantenated shut period starts with a good, resolvable
            shutting and ends when first good resolvable opening found.
            Length of concat shut period = sum of all durations before the
            resolved opening. Amplitude of concat shut period = 0.
        (2) A concantenated open period starts with a good, resolvable opening
            and ends when first good resolvable interval is found that
            has a different amplitude (either shut or open but different
            amplitude). Length of concat open period = sum of all concatenated
            durations. Amplitude of concatenated open period = weighted mean
            amplitude of all concat intervals.
        First interval in each concatenated group must be resolvable, but may
        be bad (in which case all group will be bad).
        """

        # If last interval is shut, then set it as unusable.
        if self.iampl[-1] == 0: self.iprops[-1] = 8
        # Check for negative intervals and set them unusable.
        for i in range(len(self.itint)):
            if self.itint[i] < 0: self.iprops[i] = 8

        # Find first resolvable and usable interval.
        n = 0
        firstResolved = False
        while not firstResolved:
            if ((self.itint[n] > tres) and (self.iprops[n] != 8)): # and
                #(self.itint[n-1] > tres) and (self.iprops[n-1] != 8)):
                firstResolved = True # first interval is usable and resolvable
            else:
                n += 1
        # TODO: check if preceeding interval is resolvable and not bad.

        rtint = [self.itint[n]]
        rampl = [self.iampl[n]]
        rprops = [self.iprops[n]]
        ttemp = rtint[-1]
        aavtemp = rampl[-1] * rtint[-1]

        # Start looking for unresolvable intervals.
        n += 1
        while n < (len(self.itint)-1):

            if self.itint[n] < tres:
                rtint[-1] += self.itint[n]
                if (rampl[-1] != 0) and (self.iampl[n] != 0):
                    aavtemp += self.iampl[n] * self.itint[n]
                    ttemp += self.itint[n]
                    rampl[-1] = aavtemp / ttemp

            else:
                if ((self.iampl[n] == 0) and (rampl[-1] == 0)):
                    rtint[-1] += self.itint[n]
                elif (rampl[-1] == self.iampl[n]) and (rampl[-1] != 0):
                    rtint[-1] += self.itint[n]
                    aavtemp += self.iampl[n] * self.itint[n]
                    ttemp += self.itint[n]
                    rampl[-1] = aavtemp / ttemp
                else:
                    rtint.append(self.itint[n])
                    rampl.append(self.iampl[n])
                    rprops.append(self.iprops[n])
                    ttemp = rtint[-1]
                    aavtemp = rampl[-1] * rtint[-1]

            if self.iprops[n] == 8:
                rprops[-1] = 8
            n += 1
        # end of while

        # TODO: check if the very last interval is closed or open.
        if rampl[0] == 0:
            rtint.pop(0)
            rampl.pop(0)
            rprops.pop(0)

        if rampl[-1] != 0:
            rtint.pop()
            rampl.pop()
            rprops.pop()

        self.rtint = rtint
        self.rampl = rampl
        self.rprops = rprops
        self.resolution_imposed = True
        self.tres = tres

    def get_open_shut_periods(self):
        """
        Separate open and shut intervals from the entire record.
        
        There may be many small amplitude transitions during one opening,
        each of which will count as an individual opening, so generally
        better to look at 'open periods'.

        Look for start of a group of openings i.e. any opening that has
        defined duration (i.e. usable).  A single unusable opening in a group
        makes its length undefined so it is excluded.
        NEW VERSION -ENSURES EACH OPEN PERIOD STARTS WITH SHUT-OPEN TRANSITION
        Find start of a group (open period) -valid start must have a good shut
        time followed by a good opening -if a bad opening is found as first (or
        any later) opening then the open period is abandoned altogether, and the
        next good shut time sought as start for next open period, but for the
        purposes of identifying the nth open period, rejected ones must be counted
        as an open period even though their length is undefined.
        """

        opint = []
        oppro = []
        opamp = []
        shint = []
        shpro = []
        n = 0
        tint, prop, ampl = 0, 0, 0
        first = False
        while n < len(self.rtint):
            if self.rampl[n] != 0:
                tint += self.rtint[n]
                ampl += self.rampl[n] * self.rtint[n]
                if self.rprops[n] == 8: prop = 8
                avamp = ampl / tint
                first = True
                if n == (len(self.rtint) - 1):
                    opamp.append(avamp)
                    opint.append(tint)
                    oppro.append(prop)
                n += 1
            else:
                shint.append(self.rtint[n])
                shpro.append(self.rprops[n])
                n += 1
                if first:
                    opamp.append(avamp)
                    opint.append(tint)
                    oppro.append(prop)
                    tint, prop, ampl = 0, 0, 0

        self.opint = opint
        self.opamp = opamp
        self.oppro = oppro
        self.shint = shint
        self.shpro = shpro

    def get_bursts(self, tcrit):
        """
        Cut entire single channel record into bursts using critical shut time
        interval (tcrit).

        Default definition of bursts:
        (1) 'Burst amplitude' defined as mean current (excluding shut
            periods) during burst;
        (2) Bursts with any 'assumed' amplitudes are INCLUDED;
        (3) Require a gap > tcrit before the 1st burst in each file;
        (4) Unusable shut time NOT a valid end of burst;
        (5) No listing of the individual bursts.
        """

        #defaultdef = True # True if default burst definition accepted.
        #badend = True # True if unusable shut time is valid end of burst.
        firstgapfound = False
        i = 0
        if self.rampl[0] != 0:
            firstgapfound = True
        #print("First burst starts only after gap > tcrit in each file.")
        #print("First burst starts with first good opening in each file.")
        #if badend:
        #    print("Unusable shut time treated as a valid end of burst.")
        #else:
        #    print("Unusable shut time aborts a burst.")
        while i < len(self.rtint) and not firstgapfound:
            if self.rampl[i] == 0 and self.rtint[i] > tcrit:
                firstgapfound = True
            i += 1
        #print ("First long gap found: n={0:d}; ".format(gap1) +
        #    "length = {0:.6f} ms".format(self.rtint[gap1]))

        bursts = []
        burst = []
        endburst = False
        openinglength = 0
        while i < len(self.rtint):
            if self.rampl[i] != 0:
                openinglength += self.rtint[i]
                #TODO: if bad opening: set burst bad
            else: # found gap
                if self.rtint[i] < tcrit and not endburst and i != (len(self.rtint)-1) and self.rprops[i] != 8:
                    burst.append(openinglength)
                    burst.append(self.rtint[i])
                    openinglength = 0
                else: # gap is longer than tcrit
                    endburst = True
            if endburst:
                burst.append(openinglength)
                openinglength = 0
                # TODO: bad/unusable gap
                bursts.append(burst)
                endburst = False
                burst = []
            i += 1
        self.bursts = bursts

    def get_burst_length_list(self):
        blength = []
        for ind in range(len(self.bursts)):
            blength.append(np.sum(self.bursts[ind]))
        return blength

    def get_openings_burst_list(self):
        openings = []
        for ind in range(len(self.bursts)):
            openings.append((len(self.bursts[ind]) + 1) / 2)
        return openings

    def print_bursts(self):
        print('\n#####\nList of bursts:\n')
        for ind in range(len(self.bursts)):
            print '{0:d} '.format(ind), self.bursts[ind]
        print('\n###############\n\n')
        
    def set_tres(self, tres):
        self.tres = tres
        
    def set_tcrit(self, tcrit):
#        self.chs = False if (tcrit < 0) else True
        self.tcrit = math.fabs(tcrit)

    def set_conc(self, conc):
        self.conc = conc
        
    def set_chs(self, chs):
        self.chs = chs # CHS vectors: yes or no
        
    def set_onechan(self, onechan):
        self.onechan = onechan # opening from one channel only?
        
    def set_badend(self, badend):
        self.badend = badend # bad shutting can terminate burst?
        
    def __repr__(self):
        
        str_repr = '\n\n Data loaded from file: '
        str_repr += self.filenames[0]
        
#        if self.record_type:
#            str_repr += '\n'
#            if self.record_type == 'simulated':
#                str_repr += '\nSimulated data loaded from file: '
#            elif self.record_type == 'recorded':
#                str_repr += '\nRecorded data loaded from file: '
#            str_repr += self.filenames[0]
#        else:
#            str_repr += '\nData not loaded...'

        if self.tres:
            str_repr += '\nNumber of resolved intervals = {0:d}'.format(len(self.rtint))
            str_repr += '\nNumber of resolved periods = {0:d}'.format(len(self.opint) + len(self.shint))
            str_repr += '\n\nNumber of open periods = {0:d}'.format(len(self.opint))
            str_repr += ('\nMean and SD of open periods = {0:.9f} +/- {1:.9f} ms'.
                format(np.average(self.opint)*1000, np.std(self.opint)*1000))
            str_repr += ('\nRange of open periods from {0:.9f} ms to {1:.9f} ms'.
                format(np.min(self.opint)*1000, np.max(self.opint)*1000))
            str_repr += ('\n\nNumber of shut intervals = {0:d}'.format(len(self.shint)))
            str_repr += ('\nMean and SD of shut periods = {0:.9f} +/- {1:.9f} ms'.
                format(np.average(self.shint)*1000, np.std(self.shint)*1000))
            str_repr += ('\nRange of shut periods from {0:.9f} ms to {1:.9f} ms'.
                format(np.min(self.shint)*1000, np.max(self.shint)*1000))
            str_repr += ('\nLast shut period = {0:.9f} ms'.format(self.shint[-1]*1000))
        else:
            str_repr += '\n\nTemporal resolution not imposed...'

        if self.tcrit:
            str_repr += ('\n\nNumber of bursts = {0:d}'.format(len(self.bursts)))
            blength = self.get_burst_length_list()
            str_repr += ('\nAverage length = {0:.9f} ms'.
                format(np.average(blength)*1000))
            str_repr += ('\nRange: {0:.3f}'.format(min(blength)*1000) +
                ' to {0:.3f} millisec'.format(max(blength)*1000))
            openings = self.get_openings_burst_list()
            str_repr += ('\nAverage number of openings= {0:.9f}'.
                format(np.average(openings)))
        else:
            str_repr += '\n\nBursts not separated...'

        return str_repr

    def printout(self, output=sys.stdout):
        output.write('%s' % self)
        
def false_events(tres, fc, rms, amp):
    """
    Version for EKDIST/new SCAN (avamp, rms already in pA). \
    To calc false event rate (per sec) in EKDIST (from RESINT.) \
    First calc threshold as amp attained by pulse of length=tres (in ms).
    """
    u = erf(2.668 * fc * tres)
    phi = u * amp    #'threshold' (pA)
    var = (rms) ** 2    # noise variance (pA)**2
    # Calc rate from C & Sigworth eq. 9, with k=1
    frate = fc * np.exp(-(phi * phi) / (2. * var))
    return frate    # false event rate

def prepare_hist(X, tres):
    """

    """

    n = len(X)
    xmax = max(X)
    xstart = tres #* 1000    # histogramm starts at

    # Defines bin width and number of bins.
    # Number of bins/decade
    if (n <= 300): nbdec = 5
    if (n > 300) and (n <= 1000): nbdec = 8
    if (n > 1000) and (n <= 3000): nbdec = 10
    if (n > 3000): nbdec = 12

    # round down minimum value, so get Xmin for distribution
    # round up maximum value, so get Xmax for distribution
    #xmin1 = int(xmin - 1)
    #xmax1 = int(xmax + 1)
    
    xend = 1. + xmax - math.fmod(xmax, 1.)    # last x value
    dx = math.exp(math.log(10.0) / float(nbdec))
    nbin = 1 + int(math.log(xend / xstart) / math.log(dx))

    # Make bins.
    xaxis = np.zeros(nbin+1)
    xaxis[0] = xstart
    # For log scale.
    for i in range(1, nbin+1):
        xaxis[i] = xstart * (dx**i)

    # Sorts data into bins.
    freq = np.zeros(nbin)
    for i in range(n):
        for j in range(nbin):
            if X[i] >= xaxis[j] and X[i] < xaxis[j+1]:
                freq[j] = freq[j] + 1

    xout = np.zeros((nbin + 1) * 2)
    yout = np.zeros((nbin + 1) * 2)

    xout[0] = xaxis[0]
    yout[0] = 0
    for i in range(0, nbin):
        xout[2*i+1] = xaxis[i]
        xout[2*i+2] = xaxis[i+1]
        yout[2*i+1] = freq[i]
        yout[2*i+2] = freq[i]
    xout[-1] = xaxis[-1]
    yout[-1] = 0

    return xout, yout, dx

