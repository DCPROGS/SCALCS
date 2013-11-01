#! /usr/bin/python
"""
A simple GUI for DC_PyPs project.
Depends on pyqt and matplotlib modules.
"""

try:
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")

try:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
    from matplotlib.figure import Figure
    from matplotlib import scale as mscale
except:
    raise ImportError("matplotlib module is missing")

import samples
import scplotlib as scpl

from myqtlibs.mechmenu import MechMenu
from myqtlibs.burstmenu import BurstMenu
from myqtlibs.jumpmenu import JumpMenu
from myqtlibs.scalcsmenu import ScalcsMenu
import myqtlibs.myqtcommon as myqtcommon

class QMatGUI(QMainWindow):
    def __init__(self, parent=None):
        super(QMatGUI, self).__init__(parent)
        self.resize(1000, 700)     # wide, high in px
        self.mainFrame = QWidget()

        self.path = ""
        self.mec = samples.CH82()
        self.mec.KBlk = 0.01
        self.mec.fastblk = False
        self.conc = 100e-9    # 100 nM
        self.tres = 0.0001
        self.rec1 = None
        self.data_loaded = False
        self.my_colour = ["r", "g", "b", "m", "c", "y"]
        self.present_plot = None

        self.menuBar().addMenu(MechMenu(self))
        self.menuBar().addMenu(ScalcsMenu(self))
        self.menuBar().addMenu(BurstMenu(self))
        self.menuBar().addMenu(JumpMenu(self))

        saveMenu = self.menuBar().addMenu('&Save')
        savePrintOutAction = myqtcommon.createAction(self, "&All work to text file", self.onSavePrintOut)
        savePlotASCII = myqtcommon.createAction(self, 
            "&Save current plot to text file", self.onSavePlotASCII)
        myqtcommon.addActions(saveMenu, (savePrintOutAction, savePlotASCII,
            None))

        helpMenu = self.menuBar().addMenu('&Help')
        helpAboutAction = myqtcommon.createAction(self, "&About", self.onHelpAbout)
        myqtcommon.addActions(helpMenu, (helpAboutAction, None))

        self.dpi = 85
        self.fig = Figure((6.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.mainFrame)
        self.axes = self.fig.add_subplot(111)
        for loc, spine in self.axes.spines.iteritems():
            if loc in ['right','top']:
                spine.set_color('none') # don't draw spine
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.mplTools = NavigationToolbar(self.canvas, self.mainFrame)
        mscale.register_scale(scpl.SquareRootScale)

        self.textBox = QTextBrowser()
        # Set here if printout to TextBox only or also to file or console.
        self.log = myqtcommon.PrintLog(self.textBox) #, sys.stdout)    
        myqtcommon.startInfo(self.log)

        plotSetLayout = QVBoxLayout()
        self.txtPltBox = QTextBrowser()
        plotSetLayout.addWidget(self.txtPltBox)

        leftVBox = QVBoxLayout()
        leftVBox.addLayout(plotSetLayout)
        leftVBox.addWidget(self.canvas)
        leftVBox.addWidget(self.mplTools)
        rightVBox = QVBoxLayout()
        rightVBox.addWidget(self.textBox)
        HBox = QHBoxLayout()
        HBox.addLayout(leftVBox)
        HBox.addLayout(rightVBox)
        self.mainFrame.setLayout(HBox)
        self.setCentralWidget(self.mainFrame)
        
    def onSavePrintOut(self):
        """
        """
        printOutFilename, filt = QFileDialog.getSaveFileName(self,
                "Save as PRT file...", ".prt",
                "PRT files (*.prt)")

        self.textBox.selectAll()
        text = self.textBox.toPlainText()
        fout = open(printOutFilename,'w')
        fout.write(text)
        fout.close()

        self.txtPltBox.clear()
        self.txtPltBox.append('Saved printout file:')
        self.txtPltBox.append(printOutFilename)

    def onSavePlotASCII(self):

        savePlotTXTFilename, filt = QFileDialog.getSaveFileName(self,
                "Save as TXT file...", self.path, ".txt",
                "TXT files (*.txt)")

        fout = open(savePlotTXTFilename,'w')
        for i in range(self.present_plot.shape[1]):
            for j in range(self.present_plot.shape[0]):
                fout.write('{0:.6e}\t'.format(self.present_plot[j, i]))
            fout.write('\n')
        fout.close()

        self.txtPltBox.append('Current plot saved in text file:')
        self.txtPltBox.append(savePlotTXTFilename)

    def onHelpAbout(self):
        """
        Display About dialog.
        Called from menu Help|About.
        """
        dialog = myqtcommon.AboutDlg(self)
        if dialog.exec_():
            pass

