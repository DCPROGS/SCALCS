import time, math, sys, socket, os
import numpy as np
from pylab import*

from dcpyps import dcio
from dcpyps import dataset



# LOAD DATA.
scnfiles = [["./dcpyps/samples/scn/000225S1.DAT"]]
#"./dcpyps/samples/scn/CH82.scn"
tres = [0.000025]
tcrit = [0.01]
conc = [30e-6]
chs = [True]

recs = []
bursts = []
for i in range(len(scnfiles)):
    rec = dataset.SCRecord(scnfiles[i], conc[i], tres[i], tcrit[i], chs[i])
    rec.record_type = 'recorded'
    recs.append(rec)
    bursts.append(rec.bursts)
    rec.printout()