import sys
import numpy as np
from dcpyps import dcio
from dcpyps import dataset
from dcpyps import mechanism
from dcprogs.likelihood import Log10Likelihood

# LOAD DATA.
scnfiles = [["./dcpyps/samples/scn/001004S2.SCN"], ["./dcpyps/samples/scn/991119S1.DAT"],
    ["./dcpyps/samples/scn/000225S4.DAT"]]
tres = [0.000025, 0.000025, 0.000025]
tcrit = [0.002, 0.0035, -0.035]
conc = [50e-9, 100e-9, 10e-6]
badopen = [0.02, 0.02, -1]

recs = []
bursts = []
for i in range(len(scnfiles)):
    rec = dataset.SCRecord(scnfiles[i], conc[i], tres[i], tcrit[i], badopen[i])
    recs.append(rec)
    bursts.append(rec.bursts.intervals())
    rec.printout()

mecfn = "./dcpyps/samples/mec/demomec.mec"
version, meclist, max_mecnum = dcio.mec_get_list(mecfn)
mec = dcio.mec_load(mecfn, meclist[1][0])

# PREPARE RATE CONSTANTS.
#rates = mec.unit_rates()
rates = [1900.9, 50046, 3939.1, 82.36, 40099, 0.95, 10000, 0.443e+08, 80.45,
    0.281e+09, 10000, 0.443e+08, 80.45, 0.281e+09]
mec.set_rateconstants(rates)

# Fixed rates.
#fixed = np.array([False, False, False, False, False, False, False, True,
#    False, False, False, False, False, False])
#if fixed.size == len(mec.Rates):
#for i in range(len(mec.Rates)):
#    mec.Rates[i].fixed = False

# Constrained rates.
mec.Rates[6].is_constrained = True
mec.Rates[6].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[6].constrain_args = [10, 1]
mec.Rates[8].is_constrained = True
mec.Rates[8].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[8].constrain_args = [12, 1]
mec.Rates[9].is_constrained = True
mec.Rates[9].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[9].constrain_args = [13, 1]

mec.update_constrains()

mec.set_mr(True, 11, 0)

mec.printout(sys.stdout)

kwargs = {'nmax': 2, 'xtol': 1e-12, 'rtol': 1e-12, 'itermax': 100,
'lower_bound': -1e6, 'upper_bound': 0}

LL = []
print "len recs = " + str(len(recs))
for i in range(len(recs)):
    LL.append(Log10Likelihood(recs[i].bursts.intervals(), mec.kA,
    recs[i].tres, recs[i].tcrit, **kwargs))

def dcprogslik(mec, x,LL,recs):
    mec.theta_unsqueeze(x)
    mec.update_constrains()
    lik = 0
    for i in range(len(recs)):
        mec.set_eff('c', recs[i].conc)
        lik += -LL[i](mec.Q) * np.log(10)

    return lik

thetaME = np.array([2071.37316885,45826.55905894,125313.75447893,170.09825764,8294.07778271,1.13466114,70707594.17677660,16901.29463903,124.90142986,37777037.80601518])
thetaLP = np.array([2092.8, 46087, 1547.5, 7.07E-09, 12100, 1.1993, 7.16E+07, 17174, 128.26, 3.80E+07])
thetaHJC = np.array([2109.41138,46183.58203,257.16953,1.79487,12033.98145,1.21491,7.17E+07,17099.25391,130.56303,3.83E+07])

print "\ndcprogslikME = " + str(dcprogslik(mec,thetaME,LL,recs))
print "\ndcprogslikLP = " + str(dcprogslik(mec,thetaLP,LL,recs))
print "\ndcprogslikHJC = " + str(dcprogslik(mec,thetaHJC,LL,recs))
