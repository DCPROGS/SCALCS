#! /usr/bin/env python
"""
Launch rantest GUI.
"""

import sys

try:
    from PyQt4.QtGui import *
except:
    raise ImportError("pyqt module is missing")

import dcpyps.rantestQT as rantestQT

if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = rantestQT.rantestQT()
    form.show()
    app.exec_()