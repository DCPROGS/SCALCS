import time
import sys
import os
import socket
import math

try:
#    from PyQt4.QtCore import *
#    from PyQt4.QtGui import *
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")

import numpy as np

from dcpyps import dcio
from dcpyps import samples
from dcpyps import mechanism
from dcpyps import dataset
import myqtcommon


class NewSetDlg(QDialog):
    """
    """
    def __init__(self, parent=None, recs=None, log=None):
        super(NewSetDlg, self).__init__(parent)
        self.recs = recs
        self.changed = False
        self.log = log
        
        self.scnfiles = []
        self.tres = []
        self.tcrit = []
        self.conc = []

        self.layoutMain = QVBoxLayout()
        bttnAdd = QPushButton('Add Record', self)
        bttnAdd.clicked.connect(self.onAddBttn)
        self.layoutMain.addWidget(bttnAdd)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        self.layoutMain.addWidget(buttonBox)

        self.setLayout(self.layoutMain)
        self.setGeometry(100,100,750,550)
        self.setWindowTitle("Choose and add a single channel record...")
        
    def onAddBttn(self):
        filenames = None
        dialog = AddRecordDlg(self)
        if dialog.exec_():
            filenames, tres, conc, tcrit, onechan, chs, badend = dialog.return_record()
        self.scnfiles.append(filenames)
        self.tres.append(tres)
        self.conc.append(conc)
        if not chs: tcrit *= -1
        self.tcrit.append(tcrit)
        
        textBox = QTextBrowser()
        for name in filenames:
            textBox.append(name)
        textBox.append('Concentration: {0:.6f} microM'.format(conc * 1e6))
        textBox.append('Resolution: {0:.6f} microseconds'.format(tres * 1e6))
        textBox.append('Critical time: {0:.6f} milliseconds'.format(conc * 1e3))
        textBox.append('Use CHS vectors: {}'.format(chs))
        self.layoutMain.addWidget(textBox)

    def return_set(self):
        return self.scnfiles, self.conc, self.tres, self.tcrit
     
class AddRecordDlg(QDialog):
    """

    """
    def __init__(self, parent=None, filename=None):
        super(AddRecordDlg, self).__init__(parent)

        self.recfile = []
        self.tres = 20 # resolution in microsec
        self.conc = 1 # concentration in mM.
        self.tcrit = 1 # critical time in ms
        self.chs = Qt.Checked # CHS vectors: yes or no
        self.onechan = Qt.Unchecked # opening from one channel only?
        self.badend = Qt.Checked # bad shutting can terminate burst?

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Specify record: "))
        
        layoutBttnFiles = QHBoxLayout()
        bttnAdd = QPushButton('Add file', self)
        bttnAdd.clicked.connect(self.onAddBttn)
        layoutBttnFiles.addWidget(bttnAdd)
        bttnClear = QPushButton('Clear file(s)', self)
        bttnClear.clicked.connect(self.onClearBttn)
        layoutBttnFiles.addWidget(bttnClear)
        
        layoutMain.addLayout(layoutBttnFiles)
        self.textBox = QTextBrowser()
        layoutMain.addWidget(self.textBox)
        
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Concentration (mM):"))
        self.concEdit = QLineEdit(unicode(self.conc))
        self.concEdit.setMaxLength(12)
        self.connect(self.concEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.concEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Resolution (microsec):"))
        self.resEdit = QLineEdit(unicode(self.tres))
        self.resEdit.setMaxLength(12)
        self.connect(self.resEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.resEdit)
        layoutMain.addLayout(layout)
        
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Critical shut time, tcrit, (ms):"))
        self.tcritEdit = QLineEdit(unicode(self.tcrit))
        self.tcritEdit.setMaxLength(12)
        self.connect(self.tcritEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.tcritEdit)
        layoutMain.addLayout(layout)
        
        self.onechanCheck = QCheckBox('All openings from one channel', self) 
        self.onechanCheck.setCheckState(self.onechan)
        self.connect(self.onechanCheck, SIGNAL("stateChanged()"),
            self.on_par_changed)
        layoutMain.addWidget(self.onechanCheck)
        
        self.chsCheck = QCheckBox('Use CHS vectors', self) 
        self.chsCheck.setCheckState(self.chs)
        layoutMain.addWidget(self.chsCheck)
        
        self.badendCheck = QCheckBox('Bad shutting can end group', self) 
        self.badendCheck.setCheckState(self.badend)
        layoutMain.addWidget(self.badendCheck)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Enter record parameters...")
        
    def onAddBttn(self):
        filename, filt = QFileDialog.getOpenFileName(self,
            "Open SCN File...", "", "DC SCN Files (*.scn)")
        self.recfile.append(filename)
        self.textBox.append(filename)
        
    def onClearBttn(self):
        self.recfile = []
        self.textBox.clear()

    def on_par_changed(self):
        """
        """
        self.tres = float(self.resEdit.text()) * 1e-6
        self.conc = float(self.concEdit.text()) * 1e-3
        self.tcrit = float(self.tcritEdit.text()) * 1e-3
        if self.onechanCheck.checkState() > 0:
            self.onechan = True
        else:
            self.onechan = False
        if self.chsCheck.checkState() > 0:
            self.chs = True
        else:
            self.chs = False
        if self.badendCheck.checkState() > 0:
            self.badend = True
        else:
            self.badend = False
            
    def return_record(self):
        """
        Return parameter dictionary on exit.
        """
        self.on_par_changed()
        return self.recfile, self.conc, self.tres, self.tcrit, self.onechan, self.chs, self.badend


