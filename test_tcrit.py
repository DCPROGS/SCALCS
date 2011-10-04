#! /usr/bin/python

# To change this template, choose Tools | Templates
# and open the template in the editor.

import sys

from dcpyps import scburst
from dcpyps import samples


__author__="Lab"
__date__ ="$04-Oct-2011 08:05:07$"

if __name__ == "__main__":

    output=sys.stdout
    
    mec = samples.CH82()
    mec.printout(output)
    conc = 100e-9
    mec.set_eff('c', conc)

    scburst.printout_tcrit(mec, output=sys.stdout)


