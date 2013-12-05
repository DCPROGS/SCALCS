#! /usr/bin/python

import sys
import math
import random

import numpy as np

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
        if tcrit:
            self.tcrit = math.fabs(tcrit)
        else:
            self.tcrit = tcrit
        self.conc = conc
        self.chs = chs # CHS vectors: yes or no
        self.onechan = onechan # opening from one channel only?
        self.badend = badend # bad shutting can terminate burst?

        self.record_type = None
        self.resolution_imposed = False

        self.bursts = []
        
        if self.filenames and self.tres and self.tcrit:
            self.load_from_file()
            self.impose_resolution(self.tres)
            self.get_periods()
            self.get_bursts(self.tcrit)
        

    def load_from_file(self):
        #TODO: enable taking several scan files and join in a single record.
        # Just a single file could be loaded at present.
        ioffset, nint, calfac, header = dcio.scn_read_header(self.filenames[0])
        self.itint, self.iampl, self.iprops = dcio.scn_read_data(
            self.filenames[0], header)
        if header['iscanver'] == -103:
            self.record_type = 'simulated'

    def simulate_record(self, mec, tres, state, opamp=5, nintmax=5000):
        """
        """
        picum = np.cumsum(scl.transition_probability(mec.Q), axis=1)
        tmean = -1 / mec.Q.diagonal() # in s
        itint = [random.expovariate(1 / tmean[state])]
        iampl = [opamp if state < mec.kA else 0]
        iprops = [0]
        while len(itint) < nintmax:
            state, t, a = self.next_state(state, picum, tmean, mec.kA, opamp)
            if t < tres or a == iampl[-1]:
                itint[-1] += t
            else:
                itint.append(t)
                iampl.append(a)
                iprops.append(0)
        self.itint = itint
        self.iampl = iampl
        self.iprops = iprops
        self.rint = self.itint
        self.ramp = self.iampl
        self.ropt = self.iprops
        self.resolution_imposed = True
        self.tres = tres
        self.record_type = 'simulated'
        self.get_periods()
#        self.get_bursts(self.tcrit)

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
        for i in range(len(self.rint)):
            print i+1, self.rint[i]*1000, self.ramp[i], self.ropt[i]
            if (self.ramp[i] == 0) and (self.rint[i] > (self.tcrit)):
                print ('\n')
        print('\n###################\n\n')
        
    def print_resolved_periods(self):
        print 'tcrit=', self.tcrit
        print('\n#########\nList of resolved periods:\n')
        for i in range(len(self.pint)):
            print i+1, self.pint[i], self.pamp[i], self.popt[i]
            if self.pamp[i] == 0 and self.pint[i] > self.tcrit:
                print ('\n')
        print('\n###################\n\n')
        
    def impose_resolution(self, tres, acrit=0):
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
        
        for i in range(len(self.itint)):
            if self.itint[i] < 0: self.iprops[i] = 8
            
        # Find first resolvable and usable interval.
        n = 0
        firstResolved = False
        if ((self.itint[n] > tres) and (self.iprops[n] != 8)):
            firstResolved = True
        else:
            n += 1
            
        while not firstResolved:
            if ((self.itint[n] > tres) and (self.iprops[n] != 8) and
#                (self.iampl[n] != 0) and 
                (self.itint[n-1] > tres) and (self.iprops[n-1] != 8)):
                    firstResolved = True # first interval is usable and resolvable
            else:
                n += 1
        
        rtint, rampl, rprops = [], [], []
        ttemp, otemp = self.itint[n], self.iprops[n]
        if (self.iampl[n] == 0):
            atemp = 0
        elif self.record_type == 'simulated':
            atemp = self.iampl[n]
        else:
            atemp = self.iampl[n] * self.itint[n]
        isopen = True if (self.iampl[n] != 0) else False
        n += 1
        
        # Start looking for unresolvable intervals.
        while n < (len(self.itint)):
            if self.itint[n] < tres: # interval is unresolvable
            
                if (len(self.itint) == n + 1) and self.iampl[n] == 0 and isopen:
                    rtint.append(ttemp)
                    if self.record_type == 'simulated':
                        rampl.append(atemp)
                    else:
                        rampl.append(atemp / ttemp)
                    rprops.append(otemp)
                    isopen = False
                    ttemp = self.itint[n]
                    atemp = 0
                    otemp = 8
                    
                else:
                    ttemp += self.itint[n]
                    if self.iprops[n] == 8: otemp = self.iprops[n]
                    if isopen: #self.iampl[n] != 0:
                        atemp += self.iampl[n] * self.itint[n]
                
            else:
                if (self.iampl[n] == 0): # next interval is resolvable shutting
                    if not isopen: # previous interval was shut
                        ttemp += self.itint[n]
                        if self.iprops[n] == 8: otemp = self.iprops[n]
                    else: # previous interval was open
                        rtint.append(ttemp)
                        if self.record_type == 'simulated':
                            rampl.append(atemp)
                        else:
                            rampl.append(atemp / ttemp)
                        rprops.append(otemp)
                        ttemp = self.itint[n]
                        otemp = self.iprops[n]
                        isopen = False
                else: # interval is resolvable opening
                    if not isopen:
                        rtint.append(ttemp)
                        rampl.append(0)
                        rprops.append(otemp)
                        ttemp, otemp = self.itint[n], self.iprops[n]
                        if self.record_type == 'simulated':
                            atemp = self.iampl[n]
                        else:
                            atemp = self.iampl[n] * self.itint[n]
                        isopen = True
                    else: # previous was open
                        if self.record_type == 'simulated':
                            ttemp += self.itint[n]
                            if self.iprops[n] == 8: otemp = self.iprops[n]
                        elif (math.fabs((atemp / ttemp) - self.iampl[n]) <= 1.e-5):
                            ttemp += self.itint[n]
                            atemp += self.iampl[n] * self.itint[n]
                            if self.iprops[n] == 8: otemp = self.iprops[n]
                        else:
                            rtint.append(ttemp)
                            rampl.append(atemp / ttemp)
                            rprops.append(otemp)
                            ttemp, otemp = self.itint[n], self.iprops[n]
                            atemp = self.iampl[n] * self.itint[n]
                       
            n += 1
        # end of while

        # add last interval
        if isopen:
            rtint.append(-1)
        else:
            rtint.append(ttemp)
        rprops.append(8)
        if isopen:
            if self.record_type == 'simulated':
                rampl.append(atemp)
            else:
                rampl.append(atemp / ttemp)
        else:
            rampl.append(0)
                      
        self.rint, self.ramp, self.ropt = rtint, rampl, rprops
        self.resolution_imposed = True
        self.tres = tres

    def get_periods(self):
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
        
        pint, pamp, popt = [], [], []
        if self.ramp[-1] != 0:
            self.rint.pop()
            self.ramp.pop()
            self.ropt.pop()
        if self.ramp[0] == 0:
            self.rint.pop(0)
            self.ramp.pop(0)
            self.ropt.pop(0)
        
        n = 1
        oint, oamp, oopt = self.rint[0], self.ramp[0], self.ropt[0]
        while n < len(self.rint):
            if self.ramp[n] != 0:
                oint += self.rint[n]
                oamp += self.ramp[n] * self.rint[n]
                if self.ropt[n] >= 8: oopt = 8
                
                if n == (len(self.rint) - 1):
                    pamp.append(oamp/oint)
                    pint.append(oint)
                    popt.append(oopt)
            else:
                pamp.append(oamp/oint)
                pint.append(oint)
                popt.append(oopt)
                oint, oamp, oopt = 0.0, 0.0, 0

                pamp.append(0.0)
                pint.append(self.rint[n])
                popt.append(self.ropt[n])
            n += 1

        self.pint, self.pamp, self.popt = pint, pamp, popt
        self.opint = self.pint[0::2]
        self.opamp = self.pamp[0::2]
        self.oppro = self.popt[0::2]
        self.shint = self.pint[1::2]
        self.shamp = self.pamp[1::2]
        self.shpro = self.popt[1::2]
        
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

        

        #i = 1 if self.pamp[0] == 0 else 0
        i = 0 
        
        bursts = []
        burst = []

        endburst = False
        
        while i < (len(self.pint) - 1):

            if self.pamp[i] != 0:
                burst.append(self.pint[i])
            
            else: # found gap
                if self.pint[i] < tcrit and self.popt[i] < 8:
                    burst.append(self.pint[i])
                else: # gap is longer than tcrit or bad
                    endburst = True
            
            if endburst:
                bursts.append(burst)
                endburst = False
                burst = []
            i += 1
            
        if self.pamp[i] != 0:
            burst.append(self.pint[i])
        if burst:
            bursts.append(burst)
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
            print ('{0:d} length {1:.6f} openings {2:d} :: '.
                format(ind+1, np.sum(self.bursts[ind]), (len(self.bursts[ind])+1)/2),
                self.bursts[ind])
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
        if self.filenames:
            str_repr += self.filenames[0]
        else:
            str_repr += "no file name; probably this is simulated record."
        
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
            str_repr += '\nNumber of resolved intervals = {0:d}'.format(len(self.rint))
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
            str_repr += ('\nLast shut period = {0:.9f} ms\n'.format(self.shint[-1]*1000))
        else:
            str_repr += '\nTemporal resolution not imposed...\n'

        if self.tcrit:
            str_repr += ('\nNumber of bursts = {0:d}'.format(len(self.bursts)))
            blength = self.get_burst_length_list()
            openings = self.get_openings_burst_list()
            
            if len(self.bursts) > 1:
                str_repr += ('\nAverage length = {0:.9f} ms'.
                    format(np.average(blength)*1000))
                str_repr += ('\nRange: {0:.3f}'.format(min(blength)*1000) +
                    ' to {0:.3f} millisec'.format(max(blength)*1000))
                openings = self.get_openings_burst_list()
                str_repr += ('\nAverage number of openings= {0:.9f}\n'.
                    format(np.average(openings)))
            else:
                str_repr += ('\nBurst length = {0:.9f} ms'.
                    format(blength[0] * 1000))
                str_repr += ('\nNumber of openings= {0:.9f}\n'.
                    format(openings[0]))
        else:
            str_repr += '\nBursts not separated...\n'

        return str_repr

    def printout(self, output=sys.stdout):
        output.write('%s' % self)
        