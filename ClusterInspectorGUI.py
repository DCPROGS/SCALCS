#! /usr/bin/env python
"""
ClusterInspector- load idealised single channel records and inspect 
clusters/bursts.
"""
import sys
from math import log10, pow
from PySide.QtGui import *
from PySide.QtCore import *
import pyqtgraph as pg
from matplotlib.widgets import MultiCursor, Cursor


from dcpyps.myqtlibs.plotwindow import MatPlotWin
from dcpyps.myqtlibs.plotwindow import MatPlotTools

import numpy as np
from dcpyps import dcio
from dcpyps import dataset
from dcpyps import scplotlib as scpl
from dcpyps.reports import ClusterReportHTML

class ClusterInspector(QMainWindow):
    def __init__(self, parent=None):
        super(ClusterInspector, self).__init__(parent)
        self.resize(900, 600)     # wide, high in px
        self.mainFrame = QWidget()
        self.setWindowTitle("DC_PyPs: ClusterInspector- load idealised " +
            "single channel records and inspect clusters/bursts.")
            
        self.path = None
        self.records = []
            
        leftVBox = QVBoxLayout()
        self.textBox = QTextBrowser()
        leftVBox.addWidget(self.textBox)
        # Add widgets into each dock
        self.loadBtn = QPushButton('Load idealised record')
        self.removeBtn = QPushButton('Remove last record')
        hbox = QHBoxLayout()
        hbox.addWidget(self.loadBtn)
        hbox.addWidget(self.removeBtn)
        leftVBox.addLayout(hbox)
        self.saveBtn = QPushButton('Save current session')
        self.clearBtn = QPushButton('Delete all records')
        hbox = QHBoxLayout()
        hbox.addWidget(self.saveBtn)
        hbox.addWidget(self.clearBtn)
        leftVBox.addLayout(hbox)
        self.loadBtn.clicked.connect(self.load)
        self.removeBtn.setEnabled(False)
        self.saveBtn.setEnabled(False)
        self.clearBtn.setEnabled(False)
        self.clearBtn.clicked.connect(self.clear)
        
        rightVBox = QVBoxLayout()
        self.plt3 = MatPlotWin((6,4))
        rightVBox.addWidget(self.plt3)
        self.plt4 = MatPlotWin((6,4))
        rightVBox.addWidget(self.plt4)
        HBox = QHBoxLayout()
        HBox.addLayout(leftVBox)
        HBox.addLayout(rightVBox)
        self.mainFrame.setLayout(HBox)
        self.update()
        self.setCentralWidget(self.mainFrame)
        
    def load(self):
        filename, filt = QFileDialog.getOpenFileName(self,
            "Open CSV file (Clampfit idealised data saved in EXCEL csv file)...",
            self.path, "CSV file  (*.csv)")
        self.textBox.append('Loaded record from file: '+filename+'\n')
        fscname = convert_clampfit_to_scn(filename)
        self.textBox.append('Converted to SCAN file: '+fscname+'\n')
        self.records.append(dataset.SCRecord([fscname]))
        
        #TODO: make popup window with open/shut time histograms with no tres
        # and tcrit imposed. Chose tres and tcrit, then separate into clusters
        # and plot cluster related plots.
        
        dialog = DwellTimeDlg(self.records[-1])
        if dialog.exec_():
             self.records[-1] = dialog.returnRec()
        
        self.records[-1].tres = 1e-5
        self.records[-1].tcrit = 0.1
        self.textBox.append('Record in ' + filename + ' contains {0:d} clusters '.
            format(self.records[-1].bursts.count()) + 'with average Popen = {0:.3f}; '.
            format(self.records[-1].bursts.get_popen_mean()) + 'tcrit = {0:.1f} ms\n'.
            format(self.records[-1].tcrit * 1000))
        for cluster in self.records[-1].bursts.all():
            self.textBox.append(str(cluster))
            
        self.update()
        self.clearBtn.setEnabled(True)
        self.saveBtn.setEnabled(True)
        self.removeBtn.setEnabled(True)
        
    def update(self):
        
#        self.plt1 =  MatPlotWin((1.5,1.5))
#        self.plt2 =  MatPlotWin((1.5,1.5))
        
        self.plt4 = MatPlotWin((6,4))
        
        if self.records:
            # Popen histogram.
            self.plt3 = MatPlotWin((6,4))
            vals = np.array(self.records[-1].bursts.get_popen_list())
            self.plt3.axes.hist(vals, bins=20, range=(0,1))
            self.plt3.draw()
            
            # Open/shut histograms

            
        #self.plt3.draw()

    def save(self):
        pass
    def remove(self):
        self.records.pop()
        self.update()
    def clear(self):
        self.records = []
        self.textBox.clear()
        self.update()
        self.clearBtn.setEnabled(False)
        self.saveBtn.setEnabled(False)
        self.removeBtn.setEnabled(False)

def convert_clampfit_to_scn(fname):
    """
    Convert
    """
    record = np.genfromtxt(fname, skip_header=1, delimiter=',')
    for i in range(len(record)):
        if np.isnan(record[i, 0]):
            record[i, 2] = 0
            record[i, 8] = record[i+1, 4] - record[i-1, 5]
    intervals = record[:, 8]
    amplitudes = record[:, 2].astype(int)
    flags = np.zeros((len(intervals)), dtype='b')
    to_filename = fname[:-3] + 'scn'
    dcio.scn_write(intervals, amplitudes, flags,
        filename=to_filename, type='Converted Clampfit ideal')
    return to_filename


class DwellTimeDlg(QDialog):
    """
    Dialog to choose mechansim and rates saved in a DC's mec file.
    """
    def __init__(self, rec, parent=None):
        super(DwellTimeDlg, self).__init__(parent)
        
        self.rec = rec
        self.tres = 1e-5
        self.tcrit = 0.1
        
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))

        self.plt1 = pg.PlotWidget()
        self.plt2 = pg.PlotWidget()
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.plt1)
        hbox1.addWidget(self.plt2)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel('tres:'))
        hbox2.addWidget(QLabel('tcrit:'))
        hbox3 = QHBoxLayout()
        
        self.spB1 = pg.SpinBox(value=self.tres, suffix='s', siPrefix=True, step=1e-6, bounds=(1e-6,1e-3))
        self.spB1.sigValueChanged.connect(self.spinBox1Changed)
        self.spB2 = pg.SpinBox(value=self.tcrit, suffix='s', siPrefix=True, step=1e-4, bounds=(1e-3,1))
        self.spB2.sigValueChanged.connect(self.spinBox2Changed)
        hbox3.addWidget(self.spB1, row=2, col=0)
        hbox3.addWidget(self.spB2, row=2, col=1)
        
        layout = QVBoxLayout()
        layout.addLayout(hbox1)
        layout.addLayout(hbox2)
        layout.addLayout(hbox3)
        layout.addWidget(buttonBox)
        
        self.update()

        self.setLayout(layout)
        self.resize(1000, 500)
        self.setWindowTitle("Impose tres and tcrit...")
        
    def update(self):
        
        self.rec.tres = self.tres
        self.rec.tcrit = self.tcrit
        self.plt1.clear()
        self.plt2.clear()
        ox, oy, dx = scpl.prepare_hist(np.array(self.rec.opint),
            self.tres)
        self.plt1.plot(ox, oy, stepMode=True, fillLevel=0,
            brush=(0, 0, 255, 80))
        self.tresLine1 = pg.InfiniteLine(angle=90, movable=True, pen='r')
        self.tresLine1.setValue(log10(self.rec.tres))
        self.tresLine1.sigPositionChangeFinished.connect(self.tresLine1Changed)
        self.plt1.addItem(self.tresLine1)
        self.plt1.setLogMode(x=True, y=False)
        self.plt1.setLabel('bottom', "Open periods", units='s')
        
        sx, sy, dx = scpl.prepare_hist(np.array(self.rec.shint), 
            self.rec.tres)
        self.plt2.plot(sx, sy, stepMode=True, fillLevel=0,
            brush=(0, 0, 255, 80))
        self.tresLine2 = pg.InfiniteLine(angle=90, movable=True, pen='r')
        self.tresLine2.setValue(log10(self.rec.tres))
        self.tcritLine = pg.InfiniteLine(angle=90, movable=True, pen='y')
        self.tcritLine.setValue(log10(self.rec.tcrit))
        self.tresLine2.sigPositionChangeFinished.connect(self.tresLine2Changed)
        self.tcritLine.sigPositionChangeFinished.connect(self.tcritLineChanged)
        self.plt2.addItem(self.tresLine2)
        self.plt2.addItem(self.tcritLine)
        self.plt2.setLogMode(x=True, y=False)
        self.plt2.setLabel('bottom', "Shut periods") #, units='ms')
        self.spB1.setValue(self.tres)
        self.spB2.setValue(self.tcrit)

    def tresLine1Changed(self):
        val = self.tresLine1.value()
        #self.tresLine2.setValue(val)
        self.tres = pow(10, val)
        #self.spB1.setValue(self.tres)
        self.update()
    def tresLine2Changed(self):
        val = self.tresLine2.value()
        #self.tresLine1.setValue(val)
        self.tres = pow(10, val)
        #self.rec.tres = self.tres
        #self.spB1.setValue(self.tres)
        self.update()
    def tcritLineChanged(self):
        val = self.tcritLine.value()
        self.tcrit = pow(10, val)
        #self.rec.tcrit = self.tcrit
        #self.spB2.setValue(self.tcrit)
        self.update()
        
    def spinBox1Changed(self):
        val = self.spB1.value()
        #self.tresLine1.setValue(log10(val))
        #self.tresLine2.setValue(log10(val))
        self.tres = val
        #self.rec.tres = self.tres
        self.update()
    def spinBox2Changed(self):
        val = self.spB2.value()
        #self.tcritLine.setValue(log10(val))
        self.tcrit = val
        #self.rec.tcrit = self.tcrit
        self.update()
        
    def returnRec(self):
        """
        Return rates on exit.
        """
        return self.rec

        
if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    form = ClusterInspector()
    form.show()
    app.exec_()