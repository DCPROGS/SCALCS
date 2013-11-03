#! /usr/bin/python
"""
A simple GUI for DC_PyPs  HJCFIT (maximum likelihood fit of single channel 
records to postulated mechanisms).
"""
import sys
try:
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")

from dcpyps.myqtlibs.mechmenu import MechMenu
from dcpyps.myqtlibs.datamenu import SCDataMenu
from dcpyps.myqtlibs.mecfitmenu import MecFitMenu
import dcpyps.myqtlibs.myqtcommon as myqtcommon

class QhjcGUI(QMainWindow):
    def __init__(self, parent=None):
        super(QhjcGUI, self).__init__(parent)
        self.resize(800, 600)     # wide, high in px
        self.mainFrame = QWidget()
        self.setWindowTitle("DC_PyPs: " +
            "HJCFIT- fit of model to open-shut times with missed events")
        self.setWindowIcon(QIcon("./dcpyps/samples/HJCFIT.png"))
        
        self.mec = None
        self.mecfn = None
        self.path = None
        self.scnfiles = []
        self.tres = []
        self.tcrit = []
        self.conc = []
        self.recs = []
        self.bursts = []
        self.recs_old = []
        self.data_loaded = False

        self.menuBar().addMenu(MechMenu(self))
        self.menuBar().addMenu(SCDataMenu(self))
        self.menuBar().addMenu(MecFitMenu(self))
                
        self.textBox = QTextBrowser()
        # Set here if printout to TextBox only or also to file or console.
        self.log = myqtcommon.PrintLog(self.textBox) #, sys.stdout)    
        myqtcommon.startInfo(self.log)
        rightVBox = QVBoxLayout()
        rightVBox.addWidget(self.textBox)
        self.mainFrame.setLayout(rightVBox)
        self.setCentralWidget(self.mainFrame)

def main(args):
    app = QApplication(sys.argv)
    form = QhjcGUI()
    form.show()
    app.exec_()
    
if __name__ == '__main__':
    main(sys.argv[1:])


    
    

