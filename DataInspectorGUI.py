#! /usr/bin/env python
"""
ClusterInspector- load idealised single channel records and inspect 
clusters/bursts.
"""
import sys
import os
from math import log10, pow
from PySide.QtGui import *
from PySide.QtCore import *
import pyqtgraph as pg
from pyqtgraph.dockarea import *
import numpy as np

from dcpyps import dcio
from dcpyps import dataset
from dcpyps import scplotlib as scpl
from dcpyps.reports import ClusterReportHTML

class DataInspector(QMainWindow):
    def __init__(self, parent=None):
        super(DataInspector, self).__init__(parent)
        self.resize(900, 600)     # wide, high in px
        self.mainFrame = QWidget()
        self.setWindowTitle("DC_PyPs: ClusterInspector- load idealised " +
            "single channel records and inspect clusters/bursts.")
        area = DockArea()
        self.setCentralWidget(area)

        self.path = None
        self.recs = []
        self.curr_rec = -1
        
        d1 = PatchInspector(self)
        area.addDock(d1, 'left')
        self.d2 = BurstPlots(self)
        area.addDock(self.d2, 'right')

    def update(self):
        self.d2.update()

class BurstPlots(Dock):
    def __init__(self, parent):
        super(BurstPlots, self).__init__(parent)
        self.parent = parent
        self.ma_period = 2
        self.min_op = 2
        self.curr_cluster = 0

        # Right side
        
        self.plt1 = pg.PlotWidget()
        self.plt2 = pg.PlotWidget()
        self.plt3 = pg.PlotWidget()
        self.plt4 = pg.PlotWidget()
        self.plt5 = pg.PlotWidget()
        self.spB1 = pg.SpinBox(value=self.min_op, step=1, bounds=(2,100))
        self.spB1.sigValueChanged.connect(self.spinBox1Changed)
        self.spB2 = pg.SpinBox(value=self.ma_period, step=1, bounds=(1,100))
        self.spB2.sigValueChanged.connect(self.spinBox2Changed)
        self.spB3 = pg.SpinBox(value=self.curr_cluster+1, step=1) # , bounds=(1,100))
        self.spB3.sigValueChanged.connect(self.spinBox3Changed)

        w2 = pg.LayoutWidget()
        
        w2.addWidget(self.plt1, row=1, col=0, colspan=2)
        w2.addWidget(self.plt2, row=1, col=2, colspan=2)
        w2.addWidget(QLabel('Use clusters with number of openings more than:'), row=2, col=0, colspan=3)
        w2.addWidget(self.spB1, row=2, col=3)
        w2.addWidget(self.plt3, row=3, col=0, colspan=2)
        w2.addWidget(self.plt4, row=3, col=2, colspan=2)
        w2.addWidget(self.plt5, row=4, col=0, colspan=4)
        w2.addWidget(QLabel('Moving average period for cluster:'), row=5, col=0)
        w2.addWidget(self.spB2, row=5, col=1)
        w2.addWidget(QLabel('Cluster #:'), row=5, col=2)
        w2.addWidget(self.spB3, row=5, col=3)
        self.addWidget(w2)

    def update(self):

        self.plt1.clear()
        self.plt2.clear()
        self.plt3.clear()
        self.plt4.clear()
        self.plt5.clear()

        all_popen = []
        all_mean_ampl = []
        opav = []
        all_op_lists = []
        all_sh_lists = []
        cluster_num = 0
        for record in self.parent.recs:
            clusters = record.bursts.get_long(self.min_op)
            cluster_num += clusters.count()
            all_popen.extend(clusters.get_popen_list())
            all_mean_ampl.extend(clusters.get_mean_ampl_list())
            opav.extend(clusters.get_opening_length_mean_list())
            all_op_lists.extend(clusters.get_op_lists())
            all_sh_lists.extend(clusters.get_sh_lists())
        y,x = np.histogram(np.array(all_popen)) #, bins=np.linspace(-3, 8, 40))
        hist = pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 80))
        self.plt1.addItem(hist)
        self.plt1.setXRange(0, 1) #, padding=None, update=True)
        self.plt1.setLabel('bottom', "Popen")
        self.plt1.setLabel('left', "Count #")
        
        y,x = np.histogram(np.array(opav)) #, bins=np.linspace(-3, 8, 40))
        hist = pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 80))
        self.plt2.addItem(hist)
        self.plt2.setLabel('bottom', "Opening mean length", units='s')
        self.plt2.setLabel('left', "Count #")
        
        self.plt3.plot(np.array(all_popen), np.array(opav)*1000,  pen=None, symbol='o', symbolPen='b', symbolSize=5, symbolBrush='b')
        self.plt3.setLabel('left', "Opening mean length", units='ms')
        self.plt3.setLabel('bottom', "Popen")
        self.plt3.setXRange(0, 1)

        #self.plt9.plot(np.array(all_popen), np.array(all_mean_ampl),  pen=None, symbol='o', symbolPen='b', symbolSize=5, symbolBrush='b')
        self.plt4.setLabel('left', "Mean amplitude", units='pA')
        self.plt4.setLabel('bottom', "Popen")
        self.plt4.setXRange(0, 1)

        copma = moving_average(all_op_lists[self.curr_cluster][:-1], self.ma_period)
        cshma = moving_average(all_sh_lists[self.curr_cluster], self.ma_period)
        cpoma = copma / (copma + cshma)
        self.plt5.plot(cpoma, stepMode=True,pen='g')
        self.plt5.setLabel('left', "Popen")
        self.plt5.setLabel('bottom', "Interval #")
        self.plt5.setYRange(0, 1)

        self.spB1.setValue(self.min_op)
        self.spB2.setValue(self.ma_period)
        self.spB3.setValue(self.curr_cluster+1)
        self.spB3.setMaximum(cluster_num)
        self.spB3.setMinimum(1)

    def spinBox1Changed(self):
        val = self.spB1.value()
        self.min_op = val
        self.update()
    def spinBox2Changed(self):
        val = self.spB2.value()
        self.ma_period = val
        self.update()
    def spinBox3Changed(self):
        val = self.spB3.value()
        self.curr_cluster = int(val-1)
        self.update()

class PatchInspector(Dock):
    def __init__(self, parent):
        super(PatchInspector, self).__init__(parent)
        self.parent = parent
        #self.title = "Patch Inspector"
        #self.resize=(1, 1)
        
        self.ma_period = 10
        self.tres_all = Qt.Unchecked
        self.tcrit_all = Qt.Unchecked

        self.textBox = QTextBrowser()

        self.loadBtn = QPushButton('Load idealised record(s)')
        self.removeBtn = QPushButton('Remove current record')
        self.saveBtn = QPushButton('Save current session')
        self.clearBtn = QPushButton('Delete all records')
        self.loadBtn.clicked.connect(self.load)
        self.removeBtn.clicked.connect(self.remove)
        self.removeBtn.setEnabled(False)
        self.saveBtn.setEnabled(False)
        self.clearBtn.setEnabled(False)
        self.clearBtn.clicked.connect(self.clear)

        self.spB1 = pg.SpinBox(suffix='s', siPrefix=True, step=1e-6, bounds=(1e-6,1e-3))
        self.spB1.sigValueChanged.connect(self.spinBox1Changed)
        self.ckB1 = QCheckBox('Impose to all loaded patches?', self)
        self.ckB1.setCheckState(self.tres_all)
        self.ckB1.stateChanged.connect(self.checkBox1Changed)

        self.spB2 = pg.SpinBox(suffix='s', siPrefix=True, step=1e-4, bounds=(1e-3,1))
        self.spB2.sigValueChanged.connect(self.spinBox2Changed)
        self.ckB2 = QCheckBox('Impose to all loaded patches?', self)
        self.ckB2.setCheckState(self.tcrit_all)
        self.ckB2.stateChanged.connect(self.checkBox2Changed)

        self.plt1 = pg.PlotWidget()
        self.plt2 = pg.PlotWidget()
        self.plt3 = pg.PlotWidget()
        self.plt4 = pg.PlotWidget()
        self.plt5 = pg.PlotWidget()
        self.spB3 = pg.SpinBox(value=self.ma_period, step=1, bounds=(1,100))
        self.spB3.sigValueChanged.connect(self.spinBox3Changed)
        self.spB7 = pg.SpinBox(value=parent.curr_rec+1, step=1)
        self.spB7.sigValueChanged.connect(self.spinBox7Changed)

        w1 = pg.LayoutWidget()

        w1.addWidget(self.loadBtn, row=0, col=0)
        w1.addWidget(self.removeBtn, row=0, col=1)
        w1.addWidget(self.saveBtn, row=0, col=2)
        w1.addWidget(self.clearBtn, row=0, col=3)

        w1.addWidget(self.textBox, row=1, col=0, colspan=4)

        w1.addWidget(QLabel('Displaying patch '), row=2, col=0)
        w1.addWidget(self.spB7, row=2, col=1)
        self.label1 = QLabel(' out of {0:d}'.format(len(parent.recs)))
        w1.addWidget(self.label1, row=2, col=2)
        self.spB7.setMaximum(len(parent.recs))
        self.spB7.setMinimum(0)

        w1.addWidget(self.plt1, row=3, col=0, colspan=2)
        w1.addWidget(self.plt2, row=3, col=2, colspan=2)
        w1.addWidget(QLabel('tres:'), row=4, col=0)
        w1.addWidget(self.spB1, row=4, col=1)
        w1.addWidget(self.ckB1, row=4, col=2, colspan=2)


        w1.addWidget(QLabel('tcrit:'), row=5, col=0)
        w1.addWidget(self.spB2, row=5, col=1)
        w1.addWidget(self.ckB2, row=5, col=2, colspan=2)

        w1.addWidget(self.plt3, row=6, col=0, colspan=4)
        w1.addWidget(self.plt4, row=7, col=0, colspan=4)
        w1.addWidget(self.plt5, row=8, col=0, colspan=4)
        w1.addWidget(QLabel('Moving average period:'), row=9, col=0, colspan=2)
        w1.addWidget(self.spB3, row=9, col=2)
        self.addWidget(w1)

    def update(self):

        self.plt1.clear()
        self.plt2.clear()
        self.plt3.clear()
        self.plt4.clear()
        self.plt5.clear()

        self.label1.setText(' out of {0:d}'.format(len(self.parent.recs)))
        self.spB7.setMaximum(len(self.parent.recs))
        self.spB7.setValue(self.parent.curr_rec+1)
        if self.parent.recs:
            self.spB7.setMinimum(1)

        ox, oy, dx = scpl.prepare_hist(np.array(self.parent.recs[self.parent.curr_rec].opint),
            self.parent.recs[self.parent.curr_rec].tres)
        self.plt1.plot(ox, oy, stepMode=True, fillLevel=0,
            brush=(0, 0, 255, 80))
        self.tresLine1 = pg.InfiniteLine(angle=90, movable=True, pen='r')
        self.tresLine1.setValue(log10(self.parent.recs[self.parent.curr_rec].tres))
        self.tresLine1.sigPositionChangeFinished.connect(self.tresLine1Changed)
        self.plt1.addItem(self.tresLine1)
        self.plt1.setLogMode(x=True, y=False)
        self.plt1.setLabel('bottom', "Open periods", units='s')
        self.plt1.setLabel('left', "Count #")

        sx, sy, dx = scpl.prepare_hist(np.array(self.parent.recs[self.parent.curr_rec].shint),
            self.parent.recs[self.parent.curr_rec].tres)
        self.plt2.plot(sx, sy, stepMode=True, fillLevel=0,
            brush=(0, 0, 255, 80))
        self.tresLine2 = pg.InfiniteLine(angle=90, movable=True, pen='r')
        self.tresLine2.setValue(log10(self.parent.recs[self.parent.curr_rec].tres))
        self.tcritLine = pg.InfiniteLine(angle=90, movable=True, pen='y')
        self.tcritLine.setValue(log10(self.parent.recs[self.parent.curr_rec].tcrit))
        self.tresLine2.sigPositionChangeFinished.connect(self.tresLine2Changed)
        self.tcritLine.sigPositionChangeFinished.connect(self.tcritLineChanged)
        self.plt2.addItem(self.tresLine2)
        self.plt2.addItem(self.tcritLine)
        self.plt2.setLogMode(x=True, y=False)
        self.plt2.setLabel('bottom', "Shut periods",  units='s') #, units='ms')
        self.plt2.setLabel('left', "Count #")
        self.spB1.setValue(self.parent.recs[self.parent.curr_rec].tres)
        self.spB2.setValue(self.parent.recs[self.parent.curr_rec].tcrit)
        self.spB3.setValue(self.ma_period)

        opma = moving_average(self.parent.recs[self.parent.curr_rec].opint, self.ma_period)
        shma = moving_average(self.parent.recs[self.parent.curr_rec].shint, self.ma_period)
        poma = opma / (opma + shma)
        self.plt3.plot(opma, stepMode=True,pen='r')
        self.plt3.plot(shma, stepMode=True,pen='b')
        self.plt3.plot(poma, stepMode=True,pen='g')
        
        #self.plt3.setLabel('left', "Open periods", units='s')
        self.plt3.setLabel('bottom', "Interval #")
        self.plt3.setLogMode(x=False, y=True)
#        self.plt4.plot(shma, stepMode=True,pen='b')
#        self.plt4.setLabel('left', "Shut periods", units='s')
#        self.plt4.setLabel('bottom', "Interval #")
#        self.plt5.plot(poma, stepMode=True,pen='g')
#        self.plt5.setLabel('left', "Popen")
#        self.plt5.setLabel('bottom', "Interval #")
#        self.plt5.setYRange(0, 1)
        
        self.parent.update()

    def update_tres(self, tres):
        if self.tres_all:
            for rec in self.parent.recs:
                rec.tres = tres
        else:
            self.parent.recs[self.parent.curr_rec].tres = tres
        self.update()
    def update_tcrit(self, tcrit):
        if self.tcrit_all:
            for rec in self.parent.recs:
                rec.tcrit = tcrit
        else:
            self.parent.recs[self.parent.curr_rec].tcrit = tcrit
        self.update()

    def tresLine1Changed(self):
        val = self.tresLine1.value()
        self.update_tres(pow(10, val))
    def tresLine2Changed(self):
        val = self.tresLine2.value()
        self.update_tres(pow(10, val))
    def tcritLineChanged(self):
        val = self.tcritLine.value()
        self.update_tcrit(pow(10, val))

    def spinBox1Changed(self):
        tres = self.spB1.value()
        self.update_tres(tres)
    def spinBox2Changed(self):
        tcrit = self.spB2.value()
        self.update_tcrit(tcrit)
    def spinBox3Changed(self):
        val = self.spB3.value()
        self.ma_period = val
        self.update()
    def spinBox7Changed(self):
        val = self.spB7.value()
        self.parent.curr_rec = int(val-1)
        self.update()

    def checkBox1Changed(self):
        if self.ckB1.checkState() > 0:
            self.tres_all = True
        else:
            self.tres_all = False
        self.update_tres(self.spB1.value())
    def checkBox2Changed(self):
        if self.ckB2.checkState() > 0:
            self.tcrit_all = True
        else:
            self.tcrit_all = False
        self.update_tcrit(self.spB2.value())

    def load(self):
        filelist, filt = QFileDialog.getOpenFileNames(self.parent,
            "Open CSV file(s) (Clampfit idealised data saved in EXCEL csv file)...",
            self.parent.path, "CSV file  (*.csv)")
        for filename in filelist:
            self.parent.path, fname = os.path.split(filename)
            #self.textBox.append('Loaded record from file: '+filename+'\n')
            fscname = convert_clampfit_to_scn(filename)
            self.textBox.append('Converted to SCAN file: '+fscname) #+'\n')
            self.parent.recs.append(dataset.SCRecord([fscname]))
        
        #self.textBox.append('Record in ' + filename + ' contains {0:d} clusters '.
        #    format(self.recs[-1].bursts.count()) + 'with average Popen = {0:.3f}; '.
        #    format(self.recs[-1].bursts.get_popen_mean()) + 'tcrit = {0:.1f} ms\n'.
        #    format(self.recs[-1].tcrit * 1000))
        #for cluster in self.recs[-1].bursts.all():
        #    self.textBox.append(str(cluster))

        self.parent.curr_rec = len(self.parent.recs) - 1
        self.update()
        self.parent.update()
        self.clearBtn.setEnabled(True)
        self.saveBtn.setEnabled(True)
        self.removeBtn.setEnabled(True)
        
    def save(self):
        pass
    def remove(self):
        self.parent.recs.pop(self.parent.curr_rec)
        self.parent.curr_rec = len(self.parent.recs) - 1
        self.update()
    def clear(self):
        self.plt1.clear()
        self.plt2.clear()
        self.plt3.clear()
        self.plt4.clear()
        self.plt5.clear()
        self.plt6.clear()
        self.plt7.clear()
        self.plt8.clear()
        self.plt9.clear()
        self.plt10.clear()
        self.parent.recs = []
        self.textBox.clear()
        #self.update()
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

def moving_average(x, n):
    """
    Compute an n period moving average.
    """
    x = np.asarray(x)
    weights = np.ones(n)
    weights /= weights.sum()
    a =  np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a


if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    form = DataInspector()
    form.show()
    app.exec_()