import time, math, sys
import numpy as np

from dcpyps.optimize import simplex
from dcpyps import dcio
from dcpyps import dataset
from dcpyps import mechanism

from dcprogs.likelihood import Log10Likelihood


# LOAD FLIP MECHANISM USED Burzomato et al 2004
mecfn = "./dcpyps/samples/demomec.mec"
version, meclist, max_mecnum = dcio.mec_get_list(mecfn)
mec = dcio.mec_load(mecfn, meclist[2][0])

tres = [0.000030, 0.000030, 0.000030, 0.000030]
tcrit = [0.004, -1, -0.06, -0.02]
conc = [10e-6, 30e-6, 100e-6, 1000e-6]
scnfiles = ["./dcpyps/samples/A-10.scn", "./dcpyps/samples/B-30.scn",
    "./dcpyps/samples/C-100.scn", "./dcpyps/samples/D-1000.scn"]

#ioffset, nint, calfac, header = dcio.scn_read_header(scnfiles[0])
#tint, iampl, iprops = dcio.scn_read_data(scnfiles[0], ioffset, nint, calfac)
#rec = dataset.SCRecord(scnfiles[0], header, tint, iampl, iprops)
## Impose resolution, get open/shut times and bursts.
#rec.impose_resolution(tres[0])
#rec.tcrit = tcrit[0]
#rec.print_resolved_intervals()

# LOAD DATA.
def get_bursts(sfile, tres, tcrit):
    print '\n\n Reading bursts form: ', sfile
    ioffset, nint, calfac, header = dcio.scn_read_header(sfile)
    tint, iampl, iprops = dcio.scn_read_data(sfile, ioffset, nint, calfac)
    rec = dataset.SCRecord(sfile, header, tint, iampl, iprops)
    # Impose resolution, get open/shut times and bursts.
    rec.impose_resolution(tres)
    print('\nNumber of resolved intervals = {0:d}'.format(len(rec.rtint)))

    rec.get_open_shut_periods()
    print('\nNumber of resolved periods = {0:d}'.format(len(rec.opint) + len(rec.shint)))
    print('\nNumber of open periods = {0:d}'.format(len(rec.opint)))
    print('Mean and SD of open periods = {0:.9f} +/- {1:.9f} ms'.
        format(np.average(rec.opint)*1000, np.std(rec.opint)*1000))
    print('Range of open periods from {0:.9f} ms to {1:.9f} ms'.
        format(np.min(rec.opint)*1000, np.max(rec.opint)*1000))
    print('\nNumber of shut intervals = {0:d}'.format(len(rec.shint)))
    print('Mean and SD of shut periods = {0:.9f} +/- {1:.9f} ms'.
        format(np.average(rec.shint)*1000, np.std(rec.shint)*1000))
    print('Range of shut periods from {0:.9f} ms to {1:.9f} ms'.
        format(np.min(rec.shint)*1000, np.max(rec.shint)*1000))
    print('Last shut period = {0:.9f} ms'.format(rec.shint[-1])*1000)

    rec.get_bursts(tcrit)
    print('\nNumber of bursts = {0:d}'.format(len(rec.bursts)))
    blength = rec.get_burst_length_list()
    print('Average length = {0:.9f} ms'.format(np.average(blength)*1000))
    print('Range: {0:.3f}'.format(min(blength)*1000) +
            ' to {0:.3f} millisec'.format(max(blength)*1000))
    openings = rec.get_openings_burst_list()
    print('Average number of openings= {0:.9f}'.format(np.average(openings)))
    return rec.bursts

bursts = []
for i in range(len(scnfiles)):
    bursts.append(get_bursts(scnfiles[i], tres[i], math.fabs(tcrit[i])))

#rates = [5000.0, 500.0, 2700.0, 2000.0, 800.0, 15000.0, 300.0, 0.1200E+06, 6000.0, 0.4500E+09, 1500.0, 12000.0, 4000.0, 0.9000E+09, 7500.0, 1200.0, 3000.0, 0.4500E+07, 2000.0, 0.9000E+07, 1000, 0.135000E+08]
#mec.set_rateconstants(rates)

# PREPARE RATE CONSTANTS.
# Fixed rates.
#fixed = np.array([False, False, False, False, False, False, False, True,
#    False, False, False, False, False, False])
#if fixed.size == len(mec.Rates):
for i in range(len(mec.Rates)):
    mec.Rates[i].fixed = False

# Constrained rates.
mec.Rates[21].is_constrained = True
mec.Rates[21].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[21].constrain_args = [17, 3]
mec.Rates[19].is_constrained = True
mec.Rates[19].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[19].constrain_args = [17, 2]
mec.Rates[16].is_constrained = True
mec.Rates[16].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[16].constrain_args = [20, 3]
mec.Rates[18].is_constrained = True
mec.Rates[18].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[18].constrain_args = [20, 2]
mec.Rates[8].is_constrained = True
mec.Rates[8].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[8].constrain_args = [12, 1.5]
mec.Rates[13].is_constrained = True
mec.Rates[13].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[13].constrain_args = [9, 2]

mec.Rates[7].mr=True
mec.Rates[15].mr=True
mec.update_constrains()
mec.update_mr()

# Initial guesses. Now using rate constants from numerical example.
#rates = np.log(mec.unit_rates())
#mec.set_rateconstants(np.exp(rates))
mec.printout(sys.stdout)
theta = mec.theta() #+ 1000.0 * np.random.uniform(low=-1, high=1, size=14)
print '\n\ntheta=', theta

kwargs = {'nmax': 2, 'xtol': 1e-12, 'rtol': 1e-12, 'itermax': 100, 'lower_bound': -1e6, 'upper_bound': 0}

likelihood0 = Log10Likelihood(bursts[0], mec.kA, tres[0], tcrit[0], **kwargs)
likelihood1 = Log10Likelihood(bursts[1], mec.kA, tres[1], tcrit[1], **kwargs)
likelihood2 = Log10Likelihood(bursts[2], mec.kA, tres[2], tcrit[2], **kwargs)
likelihood3 = Log10Likelihood(bursts[3], mec.kA, tres[3], tcrit[3], **kwargs)

def dcprogslik(x, args=None):
    mec.theta_unsqueeze(np.exp(x))
    mec.set_eff('c', conc[0])
    lik0 = -likelihood0(mec.Q) * math.log(10)
    mec.set_eff('c', conc[1])
    lik1 = -likelihood1(mec.Q) * math.log(10)
    mec.set_eff('c', conc[2])
    lik2 = -likelihood2(mec.Q) * math.log(10)
    mec.set_eff('c', conc[3])
    lik3 = -likelihood3(mec.Q) * math.log(10)
    return lik0+lik1+lik2+lik3, np.log(mec.theta())


lik, th = dcprogslik(np.log(theta))
print ("Starting likelihood (DCprogs)= {0:.6f}".format(-lik))

start = time.clock()
xout, fout, niter, neval = simplex(dcprogslik, np.log(theta), display=True)
print ("\nDCPROGS Fitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
        %time.localtime()[0:6])
print 'time in simplex=', time.clock() - start

print ('\n Final log-likelihood = {0:.6f}'.format(-fout))
print ('\n Number of iterations = {0:d}'.format(niter))
print ('\n Number of evaluations = {0:d}'.format(neval))
mec.theta_unsqueeze(np.exp(xout))
print "\n Final rate constants:"
mec.printout(sys.stdout)
print '\n\n'
