#! /usr/bin/env python
"""
A Qt based GUI to display properties of ion channels as described in Colquhoun & Hawkes 1977.
"""

import sys
try:
    from PyQt5.QtWidgets import *
except:
    raise ImportError("pyqt module is missing")

from scalcs.samples import samples
from scalcs.gui.plotwindow import MatPlotWin
from scalcs.gui.plotwindow import MatPlotTools
from scalcs.gui.mechmenu import MechMenu
from scalcs.gui.savemenu import SaveMenu
from scalcs.gui.helpmenu import HelpMenu
import scalcs.gui.myqtcommon as myqtcommon

from scalcs.gui.burstmenu import BurstMenu
from scalcs.gui.jumpmenu import JumpMenu
from scalcs.gui.scalcsmenu import ScalcsMenu


class QMatGUI(QMainWindow):
    def __init__(self, parent=None):
        super(QMatGUI, self).__init__(parent)
        self.resize(1000, 700)     # wide, high in px
        self.mainFrame = QWidget()
        self.setWindowTitle("DC_PyPs: " +
            "SCALCS- calculate dwell time distributions etc from Q-matrix...")

        self.path = ""
        self.mec = samples.CH82()
        self.mec.KBlk = 0.01
        self.mec.fastblk = False
        self.conc = 100e-9    # 100 nM
        self.tres = 0.0001
        self.present_plot = None

        # Prepare menu
        self.menuBar().addMenu(MechMenu(self))
        self.menuBar().addMenu(ScalcsMenu(self))
        self.menuBar().addMenu(BurstMenu(self))
        self.menuBar().addMenu(JumpMenu(self))
        self.menuBar().addMenu(SaveMenu(self))
        self.menuBar().addMenu(HelpMenu(self))

        # Prepare text box for printout and set where printout goes
        self.textBox = QTextBrowser()
        self.log = myqtcommon.PrintLog(self.textBox) #, sys.stdout)    
        myqtcommon.startInfo(self.log)
        # Prepare text box for plot legend
        self.txtPltBox = QTextBrowser()
        # Prepare plot window
        self.canvas = MatPlotWin()
        canvastools = MatPlotTools(self.canvas, self)

        leftVBox = QVBoxLayout()
        leftVBox.addWidget(self.txtPltBox)
        leftVBox.addWidget(self.canvas)
        leftVBox.addWidget(canvastools)

        rightVBox = QVBoxLayout()
        rightVBox.addWidget(self.textBox)
        HBox = QHBoxLayout()
        HBox.addLayout(leftVBox)
        HBox.addLayout(rightVBox)
        self.mainFrame.setLayout(HBox)
        self.setCentralWidget(self.mainFrame)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = QMatGUI()
    form.show()
    app.exec_()

