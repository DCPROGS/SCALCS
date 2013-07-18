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

import dcpyps.QTmain as qmg

if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = qmg.QMatGUI()
    form.show()
    app.exec_()

