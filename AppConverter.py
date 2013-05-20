#! /usr/bin/env python
"""
Launch converter GUI.
"""

import sys

try:
    from PyQt4.QtGui import *
except:
    raise ImportError("pyqt module is missing")

import dcpyps.QTconverter as QTconverter

if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = QTconverter.ConverterQT()
    form.show()
    app.exec_()