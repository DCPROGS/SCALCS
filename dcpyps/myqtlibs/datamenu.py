
import sys
import os
try:
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")
import numpy as np

from dcpyps import dcio
from dcpyps import dataset

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
        self.recordsIn = []

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Specify records: "))

        self.layoutIn1 = QHBoxLayout()
        self.layoutIn2 = QHBoxLayout()
        for i in range(10):
            self.recordsIn.append(AddRecordDlg())
        for i in range(5):
            self.layoutIn1.addWidget(self.recordsIn[i])
            self.layoutIn2.addWidget(self.recordsIn[i+5])
        layoutMain.addLayout(self.layoutIn1)
        layoutMain.addLayout(self.layoutIn2)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setGeometry(100,100,750,550)
        self.setWindowTitle("Choose and add a single channel record...")
        
    def return_set(self):

        for i in range(len(self.recordsIn)):
            if self.recordsIn[i].recfile:
                self.scnfiles.append(self.recordsIn[i].recfile)
                self.tres.append(self.recordsIn[i].tres)
                self.tcrit.append(self.recordsIn[i].tcrit)
                self.conc.append(self.recordsIn[i].conc)

        self.conc, self.scnfiles, self.tres, self.tcrit = (
            list(t) for t in zip(*sorted(zip(
            self.conc, self.scnfiles, self.tres, self.tcrit))))
        return self.scnfiles, self.conc, self.tres, self.tcrit
     
class AddRecordDlg(QWidget):
    """

    """
    def __init__(self, parent=None, path=None):
        super(AddRecordDlg, self).__init__(parent)

        self.path = path
        self.recfile = []
        self.tres = 20 # resolution in microsec
        self.conc = 1 # concentration in mM.
        self.tcrit = 1 # critical time in ms
        self.chs = Qt.Checked # CHS vectors: yes or no
        self.onechan = Qt.Unchecked # opening from one channel only?
        self.badend = Qt.Checked # bad shutting can terminate burst?

        layoutMain = QVBoxLayout()
        
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

        self.setLayout(layoutMain)
        self.setWindowTitle("Enter record parameters...")
        
    def onAddBttn(self):
        filename, filt = QFileDialog.getOpenFileName(self,
            "Open SCN File...", self.path, "DC SCN Files (*.scn)")
        self.recfile.append(filename)
        self.textBox.append(os.path.split(str(filename))[1])
        
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


def load_data(sfile, tres, tcrit, output=sys.stdout):
    output.write('\n\n Loading '+sfile)
    ioffset, nint, calfac, header = dcio.scn_read_header(sfile)
    tint, iampl, iprops = dcio.scn_read_data(sfile, ioffset, nint, calfac)
    rec = dataset.SCRecord(sfile, header, tint, iampl, iprops)
    # Impose resolution, get open/shut times and bursts.
    rec.impose_resolution(tres)
    output.write('\nNumber of resolved intervals = {0:d}'.format(len(rec.rtint)))

    rec.get_open_shut_periods()
    output.write('\nNumber of resolved periods = {0:d}'.format(len(rec.opint) + len(rec.shint)))
    output.write('\nNumber of open periods = {0:d}'.format(len(rec.opint)))
    output.write('Mean and SD of open periods = {0:.9f} +/- {1:.9f} ms'.
        format(np.average(rec.opint)*1000, np.std(rec.opint)*1000))
    output.write('Range of open periods from {0:.9f} ms to {1:.9f} ms'.
        format(np.min(rec.opint)*1000, np.max(rec.opint)*1000))
    output.write('\nNumber of shut intervals = {0:d}'.format(len(rec.shint)))
    output.write('Mean and SD of shut periods = {0:.9f} +/- {1:.9f} ms'.
        format(np.average(rec.shint)*1000, np.std(rec.shint)*1000))
    output.write('Range of shut periods from {0:.9f} ms to {1:.9f} ms'.
        format(np.min(rec.shint)*1000, np.max(rec.shint)*1000))
    output.write('Last shut period = {0:.9f} ms'.format(rec.shint[-1]*1000))

    rec.get_bursts(tcrit)
    output.write('\nNumber of bursts = {0:d}'.format(len(rec.bursts)))
    blength = rec.get_burst_length_list()
    output.write('Average length = {0:.9f} ms'.format(np.average(blength)*1000))
    output.write('Range: {0:.3f}'.format(min(blength)*1000) +
            ' to {0:.3f} millisec'.format(max(blength)*1000))
    openings = rec.get_openings_burst_list()
    output.write('Average number of openings= {0:.9f}'.format(np.average(openings)))
    return rec