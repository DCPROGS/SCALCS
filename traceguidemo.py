#! /usr/bin/python
"""
An example of Qt based GUI to display single channel record.
"""

import sys

try:
    from PyQt4.QtGui import *
except:
    raise ImportError("pyqt module is missing")

import dcpyps.tracegui as trg

if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = trg.TraceGUI()
    form.show()
    app.exec_()

