import sys
import numpy as np
from dcpyps import dcio
from dcpyps import dataset
from dcpyps import mechanism
from dcpyps.fitMLL import FittingSession

# LOAD DATA.
scnfiles = [["./dcpyps/samples/glydemo/simA.scn"], ["./dcpyps/samples/glydemo/simB.scn"],
    ["./dcpyps/samples/glydemo/simC.scn"], ["./dcpyps/samples/glydemo/simD.scn"]]
tres = [0.000030, 0.000030, 0.000030, 0.000030]
tcrit = [0.004, -1, -0.06, -0.02]
conc = [10e-6, 30e-6, 100e-6, 1000e-6]

recs = []
bursts = []
for i in range(len(scnfiles)):
    rec = dataset.SCRecord(scnfiles[i], conc[i], tres[i], tcrit[i])
    rec.record_type = 'simulated'
    recs.append(rec)
    bursts.append(rec.bursts.intervals())
    rec.printout()

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
#rates = mec.unit_rates()
rates = [5000.0, 500.0, 2700.0, 2000.0, 800.0, 15000.0, 300.0, 0.1200E+06,
    6000.0, 0.4500E+09, 1500.0, 12000.0, 4000.0, 0.9000E+09, 7500.0, 1200.0,
    3000.0, 0.4500E+07, 2000.0, 0.9000E+07, 1000, 0.135000E+08]
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

fit = FittingSession(mec, recs)
fit.run()
