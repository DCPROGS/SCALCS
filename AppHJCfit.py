#! /usr/bin/env python
"""
An example of Qt based GUI to display single channel related plots.
"""

import sys

try:
#    from PyQt4.QtGui import *
    from PySide.QtGui import *
except:
    raise ImportError("PyQt import failed")

import dcpyps.QThjcfit as qhjc

def main(args):
    app = QApplication(sys.argv)
    form = qhjc.QhjcGUI()
    form.show()
    app.exec_()
    
if __name__ == '__main__':
    main(sys.argv[1:])


    
    

