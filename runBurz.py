import time
import math
import sys
import yaml
import numpy as np
from scipy.optimize import minimize

from dcpyps import scalcslib as scl
from dcpyps import dcio
from dcpyps import dataset
from dcpyps import mechanism
from dcpyps.reports import FitReportHTML
from dcprogs.likelihood import Log10Likelihood

report = FitReportHTML()

# LOAD DATA.
scnfiles = [["./dcpyps/samples/glydemo/simA.scn"], ["./dcpyps/samples/glydemo/simB.scn"],
    ["./dcpyps/samples/glydemo/simC.scn"], ["./dcpyps/samples/glydemo/simD.scn"]]
tres = [0.000030, 0.000030, 0.000030, 0.000030]
tcrit = [0.004, -1, -0.06, -0.02]
chs = [True, False, False, False]
conc = [10e-6, 30e-6, 100e-6, 1000e-6]

recs = []
bursts = []
for i in range(len(scnfiles)):
    rec = dataset.SCRecord(scnfiles[i], conc[i], tres[i], tcrit[i], chs[i])
    rec.record_type = 'simulated'
    recs.append(rec)
    bursts.append(rec.bursts.intervals())
    rec.printout()

report.dataset(recs)

# LOAD FLIP MECHANISM USED Burzomato et al 2004
#mecfn = "C:/qmechdem.mec"
#version, meclist, max_mecnum = dcio.mec_get_list(mecfn)
#mec = dcio.mec_load(mecfn, meclist[3][0])

mecfn = "./dcpyps/samples/mec/demomec.mec"
version, meclist, max_mecnum = dcio.mec_get_list(mecfn)
mec = dcio.mec_load(mecfn, meclist[2][0])

#filename = 'E:/pDC/testDCpyps/Burz2004/mec-Burz-sim4fit-140124.yaml'
#stream = file(filename, 'r')
#mec = yaml.load(stream)

# PREPARE RATE CONSTANTS.
rates = mec.unit_rates()
#rates = [5000.0, 500.0, 2700.0, 2000.0, 800.0, 15000.0, 300.0, 0.1200E+06,
#    6000.0, 0.4500E+09, 1500.0, 12000.0, 4000.0, 0.9000E+09, 7500.0, 1200.0,
#    3000.0, 0.4500E+07, 2000.0, 0.9000E+07, 1000, 0.135000E+08]
mec.set_rateconstants(rates)

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
mec.update_constrains()

mec.set_mr(True, 7, 0)
mec.set_mr(True, 15, 1)

mec.printout(sys.stdout)
theta = np.log(mec.theta())

kwargs = {'nmax': 2, 'xtol': 1e-12, 'rtol': 1e-12, 'itermax': 100,
    'lower_bound': -1e6, 'upper_bound': 0}
likelihood = []

for i in range(len(recs)):
    likelihood.append(Log10Likelihood(bursts[i], mec.kA,
        recs[i].tres, recs[i].tcrit, **kwargs))

def dcprogslik(x, args=None):
    mec.theta_unsqueeze(np.exp(x))
    lik = 0
    for i in range(len(conc)):
        mec.set_eff('c', conc[i])
        lik += -likelihood[i](mec.Q) * math.log(10)
    return lik

iternum = 0
def printiter(theta):
    global iternum
    iternum += 1
    lik = dcprogslik(theta)
    print("iteration # {0:d}; log-lik = {1:.6f}".format(iternum, -lik))
    print(np.exp(theta))

def dcprogslik_all(x, args=None):
    mec.theta_unsqueeze(np.exp(x))
    lik = []
    for i in range(len(conc)):
        mec.set_eff('c', conc[i])
        lik.append(-likelihood[i](mec.Q) * math.log(10))
    return lik

lik = dcprogslik(theta)
lik_all = dcprogslik_all(theta)
report.rates(mec, lik)

#####
opts = {}
mec.set_eff('c', conc[2])
opts['mec'] = mec
opts['conc'] = conc[2]
opts['tres'] = tres[2]
opts['tcrit'] = -tcrit[2]
opts['isCHS'] = False
opts['data'] = bursts[2]

# MAXIMUM LIKELIHOOD FIT.
start_lik, th = scl.HJClik(np.log(theta), opts)
print ("Starting likelihood = {0:.6f}".format(-start_lik))
#####

print ("\nStarting likelihood (DCprogs)= {0:.6f}".format(-lik))
print ('\nSet#1: {0:.6f}; Set#2: {1:.6f}; Set#3: {2:.6f}; Set#4: {3:.6f}'.
    format(lik_all[0], lik_all[1], lik_all[2], lik_all[3]))
start = time.clock()
success = False
result = None
while not success:
    result = minimize(dcprogslik, theta, method='Nelder-Mead', callback=printiter,
        options={'xtol':1e-4, 'ftol':1e-4, 'maxiter': 5000, 'maxfev': 10000,
        'disp': True})
    if result.success:
        success = True
    else:
        theta = result.x

end = time.clock()
print ("\nDCPROGS Fitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
        %time.localtime()[0:6])
print 'time in simplex=', end - start
print '\n\nresult='
print result

print ('\n Final log-likelihood = {0:.6f}'.format(-result.fun))
print ('\n Number of iterations = {0:d}'.format(result.nit))
print ('\n Number of evaluations = {0:d}'.format(result.nfev))
mec.theta_unsqueeze(np.exp(result.x))
print "\n Final rate constants:"
mec.printout(sys.stdout)
print '\n\n'

report.fit_result(mec, start, end, result.fun, result.nit)
report.rates(mec, result.fun, True)
try:
    report.distributions(mec, recs)
except:
    print('Distribution plotting in HTML report failed')
report.finalise()