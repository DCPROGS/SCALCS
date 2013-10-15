import sys
import math
import os
import yaml

try:
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")

from dcpyps import dcio
from dcpyps import dataset
import myqtcommon

class SCDataMenu(QMenu):
    """
    """
    def __init__(self, parent):
        super(SCDataMenu, self).__init__(parent) 
        self.parent = parent
        self.setTitle('&Load SC data')
        
        openScanAction = myqtcommon.createAction(self, 
            "&Load single channel record from SCN file", self.onLoadSingleRec)
        simulateScanAction = myqtcommon.createAction(self, 
            "&Simulate single channel record", self.onSimulateData)
        loadNewSetAction = myqtcommon.createAction(self,
            "&Load new set", self.onLoadNewSet)
        loadSavedSetAction = myqtcommon.createAction(self,
            "Load saved set", self.onLoadSavedSet)
        saveSetAction = myqtcommon.createAction(self,
            "Save set", self.onSaveSet)
#        saveScanAction = myqtcommon.createAction(self, "Save single channel record (SCN file)",
#            self.onSaveDataSCN)
#        imposeResolutionAction = myqtcommon.createAction(self, "&Impose resolution",
#            self.onImposeResolution)
#        plotDataOpenAction = myqtcommon.createAction(self, "&Plot open period distribution",
#            self.onPlotDataOpen)
#        plotDataShutAction = myqtcommon.createAction(self, "&Plot shut period distribution",
#            self.onPlotDataShut)
#        plotDataBurstAction = myqtcommon.createAction(self, "&Plot burst length distribution",
#            self.onPlotDataBurst)
        self.addActions([openScanAction, simulateScanAction, loadNewSetAction, loadSavedSetAction, saveSetAction])
        
    def onLoadSingleRec(self):
        """
        """
        self.parent.recs_old = self.parent.recs
        self.parent.recs = []
        
        dialog = SCNDlg(self, path=self.parent.path, log=self.parent.log)
        if dialog.exec_():
            scnfiles, conc, tres, tcrit, chs = dialog.return_rec()
        self.parent.path = os.path.split(str(scnfiles[0]))[0]

        rec = dataset.SCRecord(scnfiles, conc, tres, tcrit, chs)
        rec.record_type = 'recorded'
        self.parent.recs.append(rec)
        self.parent.bursts.append(rec.bursts)
        self.parent.data_loaded = True
        rec.printout(self.parent.log)
        
    def onLoadNewSet(self):
        self.parent.recs_old = self.parent.recs
        self.parent.recs = []
        
        #TODO: dialog to get how many records in this set
        dialog1 = RecNumDlg()
        if dialog1.exec_():
            numrec = dialog1.return_par()
            
        for i in range(numrec):            
            dialog = SCNDlg(self, path=self.parent.path, log=self.parent.log)
            if dialog.exec_():
                scnfiles, conc, tres, tcrit, chs = dialog.return_rec()
            self.parent.path = os.path.split(str(scnfiles[0]))[0]
            rec = dataset.SCRecord(scnfiles, conc, tres, tcrit, chs)
            rec.record_type = 'recorded'
            self.parent.recs.append(rec)
            self.parent.bursts.append(rec.bursts)
            self.parent.data_loaded = True
            rec.printout(self.parent.log)
            
        self.parent.log.write("\n\nLoaded {0:d} records.".format(len(self.parent.recs)))
            
    def onLoadSavedSet(self):
        self.parent.recs_old = self.parent.recs
        self.parent.recs = []
        filename, filt = QFileDialog.getOpenFileName(self,
            "Open YAML File...", self.parent.path, "YAML Files (*.yaml)")
            
        self.parent.log.write('\n\nLoading set from YAML file:')
        self.parent.log.write(filename)
        stream = file(filename, 'r')
        recs = yaml.load(stream)
        for i in range(len(recs)):
            rec = dataset.SCRecord(recs[i]['file'], recs[i]['conc'],
                recs[i]['tres'], recs[i]['tcrit'], recs[i]['chs'])
            rec.record_type = 'recorded'
            self.parent.recs.append(rec)
            self.parent.bursts.append(rec.bursts)
            self.parent.data_loaded = True
            rec.printout(self.parent.log)
        
        self.parent.log.write("\n\nLoaded {0:d} records.".format(len(self.parent.recs)))
    
    def onSaveSet(self):
        
        reclist = []
        for i in range(len(self.parent.recs)):
            dic = {'file':self.parent.recs[i].filenames,
                'conc':self.parent.recs[i].conc,
                'tres':self.parent.recs[i].tres,
                'tcrit':self.parent.recs[i].tcrit,
                'chs':self.parent.recs[i].chs}
            reclist.append(dic)
        
        saveFilename, filt = QFileDialog.getSaveFileName(self,
                "Save as YAML file...", self.parent.path, ".yaml",
                "YAML files (*.yaml)")
        self.parent.path = os.path.split(str(saveFilename))[0]
        stream = file(saveFilename, 'w')
        yaml.dump(reclist, stream)
        self.parent.log.write('\n\nCurrent set saved in YAML file:')
        self.parent.log.write(saveFilename)

    def onSimulateData(self):
        """
        """
        self.parent.log.write('\n\n\t==== SIMULATING SINGLE CHANNEL RECORD ====')

        tres, conc, oamp, nint = 10e-6, 1e-6, 5, 5000
        dialog = SimRecDlg(self, tres, conc, oamp, nint)
        if dialog.exec_():
            tres, conc, oamp, nint = dialog.return_par()

        rec = dataset.SCRecord()
        #TODO: check if mechanism is loaded if not tell to load one.
        self.parent.mec.set_eff('c', conc)
        startstate = self.parent.mec.k - 1 # TODO: ask for this in dialog
        rec.simulate_record(self.parent.mec, tres, startstate, oamp, nint)
        self.parent.log.write("\nSimulation finished")
        self.parent.log.write('{0:d}'.format(len(rec.itint)) +
            ' intervals were simulated')
        rec.record_type = 'simulated'
        rec.get_open_shut_periods()
        #TODO: get bursts
        
        self.parent.recs.append(rec)
        scnfile = [self.saveSCNfile()]
        self.parent.scnfiles.append(scnfile)
        self.parent.data_loaded = True
        

    def saveSCNfile(self):
        saveSCNFilename, filt = QFileDialog.getSaveFileName(self,
                "Save as SCN file...", self.parent.path, ".scn",
                "SCN files (*.scn)")
        self.parent.path = os.path.split(str(saveSCNFilename))[0]

        dcio.scn_write_simulated(self.parent.recs[0].itint, 
            self.parent.recs[0].iampl, filename=saveSCNFilename)

        self.parent.log.write('Current single channel record saved in SCN file:')
        self.parent.log.write(saveSCNFilename)
        return saveSCNFilename

    def onImposeResolution(self):

        dialog = ConcResDlg(self, self.conc, self.tres)
        if dialog.exec_():
            self.conc, self.tres = dialog.return_par()
        self.rec1.impose_resolution(self.tres)
        self.textBox.append('After imposing the resolution of original {0:d}'.
            format(len(self.rec1.itint)) + ' intervals were left {0:d}'.
            format(len(self.rec1.rtint)))
        self.resolution_imposed = True
        self.rec1.get_open_shut_periods()


class SCNDlg(QDialog):
    """
    """
    def __init__(self, parent=None, path=None, rec=None, log=None):
        super(SCNDlg, self).__init__(parent)
        self.rec = rec
        self.changed = False
        self.log = log
        self.path = path
        
        if self.rec:
            self.recfiles = self.rec.filenames
            self.tres = self.rec.tres
            self.conc = self.rec.conc
            self.tcrit = self.rec.tcrit
            self.chs = self.rec.chs
            self.onechan = self.rec.onechan
            self.badend = self.rec.badend
        else:
            self.recfiles = []
            self.tres = 0.00002
            self.conc = 0.001
            self.tcrit = 0.001
            self.chs = True
            self.onechan = False
            self.badend = True
   
        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Specify SCAN file: "))

        self.layoutIn = QHBoxLayout()
        self.recIn = AddRecordDlg()
        self.layoutIn.addWidget(self.recIn)
        layoutMain.addLayout(self.layoutIn)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setGeometry(100,100,750,550)
        self.setWindowTitle("Choose and add a SC record...")
        
    def return_rec(self):

        return self.recIn.recfile, self.recIn.conc, self.recIn.tres, self.recIn.tcrit, self.recIn.chs 
    
class RecNumDlg(QDialog):
    """
    Dialog to input resolution.
    """
    def __init__(self, parent=None, numrec=1):
        super(RecNumDlg, self).__init__(parent)

        self.numrec = numrec

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Enter number of records/patches in set:"))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Number of records/patches:"))
        self.rEdit = QLineEdit(unicode(self.numrec))
        self.rEdit.setMaxLength(12)
        self.connect(self.rEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.rEdit)
        layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Number of records...")

    def on_par_changed(self):
        self.numrec = int(self.rEdit.text())

    def return_par(self):
        return self.numrec


class NewSetDlg(QDialog):
    """
    """
    def __init__(self, numfiles=6, parent=None, recs=None, log=None):
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

class SimRecDlg(QDialog):
    """

    """
    def __init__(self, parent=None, tres=0.00001, conc=100e-9, oamp=5,
        nint=5000):
        super(SimRecDlg, self).__init__(parent)

        self.tres = tres * 1e6 # resolution in microsec
        self.conc = conc * 1000 # concentration in mM.
        self.oamp = oamp # open chanel current amplitude in pA
        self.nint = nint # umber of intervals

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Simulated Single channel record:"))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Resolution (microsec):"))
        self.resEdit = QLineEdit(unicode(self.tres))
        self.resEdit.setMaxLength(12)
        self.connect(self.resEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.resEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Concentration (mM):"))
        self.concEdit = QLineEdit(unicode(self.conc))
        self.concEdit.setMaxLength(12)
        self.connect(self.concEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.concEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Open channel current amplitude (pA):"))
        self.ampEdit = QLineEdit(unicode(self.oamp))
        self.ampEdit.setMaxLength(12)
        self.connect(self.ampEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.ampEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Number of intervals to calculate:"))
        self.intEdit = QLineEdit(unicode(self.nint))
        self.intEdit.setMaxLength(12)
        self.connect(self.intEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.intEdit)
        layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Simulate single channel record...")

    def on_par_changed(self):
        """
        """
        self.tres = float(self.resEdit.text()) * 1e-6
        self.conc = float(self.concEdit.text()) * 1e-3
        self.oamp = float(self.ampEdit.text())
        self.nint = int(self.intEdit.text())

    def return_par(self):
        """
        Return parameter dictionary on exit.
        """
        return self.tres, self.conc, self.oamp, self.nint

