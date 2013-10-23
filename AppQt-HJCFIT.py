#! /usr/bin/env python
"""
An example of Qt based GUI to run maximum likelihood fit of single channel 
records to postulated mechanisms.
"""
import sys
try:
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


    
    

