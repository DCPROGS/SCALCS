#! /usr/bin/env python
"""
Launch rantest GUI.
"""

import sys

try:
    from PyQt4.QtGui import *
except:
    raise ImportError("pyqt module is missing")

import dcpyps.QTrantest as QTrantest

if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = QTrantest.rantestQT()
    form.show()
    app.exec_()