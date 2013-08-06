
import time, math, sys

from dcpyps import samples, dataset, dcio, optimize, scalcslib as scl, mechanism as mechanism1
from numpy import log, exp, average, abs, array, any, all

from scipy.optimize import fmin

from dcprogs import read_idealized_bursts
from dcprogs.likelihood import QMatrix, Log10Likelihood, MissedEventsG
from dcprogs.likelihood.optimization import reduce_likelihood

mechanism = samples.CH82()
tres = 0.0001
tcrit = 0.004
conc = 100e-9

filename = "./dcpyps/samples/CH82.scn"
ioffset, nint, calfac, header = dcio.scn_read_header(filename)
tint, iampl, iprops = dcio.scn_read_data(filename, ioffset, nint, calfac)
rec1 = dataset.SCRecord(filename, header, tint, iampl, iprops)
# Impose resolution, get open/shut times and bursts.
rec1.impose_resolution(tres)
rec1.get_open_shut_periods()
rec1.get_bursts(tcrit)
blength = rec1.get_burst_length_list()
openings = rec1.get_openings_burst_list()
bursts = rec1.bursts


# Prepare parameter dict for simplex (RL likelihood)
opts = {}
opts['mec'] = mechanism
opts['conc'] = conc
opts['tres'] = tres
opts['tcrit'] = tcrit
opts['isCHS'] = True
opts['data'] = bursts

# MAXIMUM LIKELIHOOD FIT.
def dcpyps(x):
  start_lik, th = scl.HJClik(log(x), opts)
  return start_lik
theta = mechanism.theta()


###################################

name   = 'CH82'
likelihood = Log10Likelihood(bursts, mechanism.kA, tres, tcrit)

def dcprogs(x):
    mechanism.theta_unsqueeze(x)
    mechanism.set_eff('c', opts['conc'])
    return -likelihood(mechanism.Q) * log(10)

x = theta #+ 10.0*random.uniform(low=-1, high=1, size=8)
print repr(x)
print "DCPYPS ", dcpyps(x)
print "DCPROGS", dcprogs(x)


#######################################################
# PREPARE RATE CONSTANTS.
# Fixed rates.
fixed = array([False, False, False, False, False,
    False, False, True, False, False])
if fixed.size == len(mechanism.Rates):
    for i in range(len(mechanism.Rates)):
        mechanism.Rates[i].fixed = fixed[i]
# Constrained rates.
mechanism.Rates[5].is_constrained = True
mechanism.Rates[5].constrain_func = mechanism1.constrain_rate_multiple
mechanism.Rates[5].constrain_args = [4, 2]
mechanism.Rates[6].is_constrained = True
mechanism.Rates[6].constrain_func = mechanism1.constrain_rate_multiple
mechanism.Rates[6].constrain_args = [8, 2]
mechanism.update_constrains()
mechanism.update_mr()
# Initial guesses. Now using rate constants from numerical example.
rates = mechanism.unit_rates()
#    rates = [100, 3000, 10000, 100, 1000, 1000, 1e+7, 5e+7, 6e+7, 10]
#    rates = [6.5, 14800, 3640, 362, 1220, 2440, 1e+7, 5e+8, 2.5e+8, 55]
mechanism.set_rateconstants(rates)
mechanism.printout(sys.stdout)
theta = mechanism.theta()
print '\ntheta=', theta


def dcprogslik(x, args=None):
    mechanism.theta_unsqueeze(exp(x))
    mechanism.set_eff('c', opts['conc'])
    return -likelihood(mechanism.Q) * log(10), log(mechanism.theta())

start = time.clock()
xout, fout, niter, neval = optimize.simplex(dcprogslik,
    log(theta), args=opts, display=True)
print ("\nDCPROGS Fitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
        %time.localtime()[0:6])
print 'time in simplex=', time.clock() - start
print ('\n Final log-likelihood = {0:.6f}'.format(-fout))
print ('\n Number of iterations = {0:d}'.format(niter))

print 'xout', xout
mechanism.theta_unsqueeze(exp(xout))
print "\n Final rate constants:"
mechanism.printout(sys.stdout)

