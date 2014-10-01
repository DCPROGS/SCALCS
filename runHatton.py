import sys
import numpy as np
from dcpyps import dcio
from dcpyps import dataset
from dcpyps import mechanism
from dcpyps.fitMLL import FittingSession

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
#    rec.record_type = 'recorded'
    recs.append(rec)
    bursts.append(rec.bursts.intervals())
    rec.printout()

# LOAD FLIP MECHANISM USED Burzomato et al 2004
#mecfn = "C:/qmechdem.mec"
#version, meclist, max_mecnum = dcio.mec_get_list(mecfn)
#mec = dcio.mec_load(mecfn, meclist[3][0])

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
theta = np.log(mec.theta())

fit = FittingSession(mec, recs)
fit.run()
