#! /usr/bin/python
"""
Maximum likelihood fit demo.
"""

import sys
from dcpyps import samples
from dcpyps import dataset
from dcpyps import mechanism
from dcpyps.fitMLL import FittingSession

print('\n\nTesting single channel data:')

# LOAD DATA.
filename = ["./dcpyps/samples/scn/CH82.scn"]
conc, tres, tcrit, chs = [100e-9], [0.0001], [0.004], [True]
recs = [dataset.SCRecord(filename, conc[0], tres[0], tcrit[0])]
recs[0].printout()

# LOAD DEMO MECHANISM (C&H82 numerical example).
mec = samples.CH82()
# PREPARE RATE CONSTANTS.
# Fixed rates.
mec.Rates[7].fixed = True
# Constrained rates.
mec.Rates[5].is_constrained = True
mec.Rates[5].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[5].constrain_args = [4, 2]
mec.Rates[6].is_constrained = True
mec.Rates[6].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[6].constrain_args = [8, 2]
mec.update_constrains()

# INITIAL GUESSES
rates = mec.unit_rates() # numbers from numerical example
rates = [100, 3000, 10000, 100, 1000, 1000, 1e+7, 5e+7, 6e+7, 10]
mec.set_rateconstants(rates)
mec.update_mr()
mec.printout(sys.stdout)
theta = mec.theta()

# DO THE JOB
#fitting(mec, recs, theta)

fit = FittingSession(mec, recs)
fit.run()