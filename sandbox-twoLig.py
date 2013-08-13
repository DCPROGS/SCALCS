#! /usr/bin/python

# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="remis"
__date__ ="$13-Aug-2013 19:19:59$"

import sys
from dcpyps import samples
from dcpyps import qmatlib as qml

if __name__ == "__main__":
    
    mec1 = samples.BCCO()
    mec1.printout(sys.stdout)
    tres = 0.0001
    tcrit = 0.004
    conc1 = 100e-9
    conc2 = 50e-9
    
    mec1.set_eff('a', conc1)
    mec1.set_eff('b', conc2)
    pinf = qml.pinf(mec1.Q)
    print '\npinf = ', pinf
    
    mec1.set_eff('a', conc1*10)
    pinf = qml.pinf(mec1.Q)
    print '\npinf = ', pinf
    
    mec1.set_eff('b', conc2*10)
    pinf = qml.pinf(mec1.Q)
    print '\npinf = ', pinf
