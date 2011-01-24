#! /usr/bin/python

# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="remis"
__date__ ="$11-Jan-2011 14:02:04$"

import time
import numpy as np

import dcpyps.qmatlib as qml
import dcpyps.scalcslib as scl
import dcpyps.mechanism as mec
import dcpyps.samples as samples


if __name__ == "__main__":

    conc = 0.0000001
    mec = samples.CH82()
    #mec = cd.demoQ()
    mec.set_eff('c', conc)

    tres = 0.0001
    open = True

    t1 = time.time()
    oproots = scl.asymptotic_roots(mec, tres, open)
    opareas = scl.asymptotic_areas(mec, tres, oproots, open)
    print time.time() - t1, "sec"

    opareas0 = opareas * np.exp(-tres * oproots)
    opareas0 = opareas0 / np.sum(opareas0)

    print ("open time constants, ms")
    for i in range(oproots.shape[0]):
        print -1000 / oproots[i], "  ", opareas[i], "  ", opareas0[i]

    open = False
    t2 = time.time()
    shroots = scl.asymptotic_roots(mec, tres, open)
    shareas = scl.asymptotic_areas(mec, tres, shroots, open)

    eigen, gama00, gama10, gama11 = scl.exact_pdf_coef(mec, tres, True)
    print "eigen=", eigen
    print "gama00=", gama00
    print "gama10=", gama10
    print "gama11=", gama11

    print "\n", time.time() - t2, "sec"

    shareas0 = shareas * np.exp(-tres * shroots)
    shareas0 = shareas0 / np.sum(shareas0)

    print ("shut time constants, ms")
    for i in range(shroots.shape[0]):
        print -1000 / shroots[i], "  ", shareas[i], "   ", shareas0[i]

    eigen, gama00, gama10, gama11 = scl.exact_pdf_coef(mec, tres, False)
    print "eigen=", eigen
    print "gama00=", gama00
    print "gama10=", gama10
    print "gama11=", gama11

