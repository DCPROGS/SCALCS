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

class ClusterInspector(QMainWindow):
    def __init__(self, parent=None):
        super(ClusterInspector, self).__init__(parent)
        self.resize(900, 600)     # wide, high in px
        self.mainFrame = QWidget()
        self.setWindowTitle("DC_PyPs: ClusterInspector- load idealised " +
            "single channel records and inspect clusters/bursts.")
        area = DockArea()
        self.setCentralWidget(area)

        self.path = None
        self.recs = []
        self.tres = 1e-5
        self.tcrit = 0.1
        self.ma_period1 = 10
        self.ma_period2 = 2
        self.min_op = 10
        self.curr_cluster = 0
        
        # Add widgets to left side
        self.loadBtn = QPushButton('Load idealised record')
        self.removeBtn = QPushButton('Remove last record')
        self.saveBtn = QPushButton('Save current session')
        self.clearBtn = QPushButton('Delete all records')
        self.loadBtn.clicked.connect(self.load)
        self.removeBtn.setEnabled(False)
        self.saveBtn.setEnabled(False)
        self.clearBtn.setEnabled(False)
        self.clearBtn.clicked.connect(self.clear)
        self.spB1 = pg.SpinBox(value=self.tres, suffix='s', siPrefix=True, step=1e-6, bounds=(1e-6,1e-3))
        self.spB1.sigValueChanged.connect(self.spinBox1Changed)
        self.spB2 = pg.SpinBox(value=self.tcrit, suffix='s', siPrefix=True, step=1e-4, bounds=(1e-3,1))
        self.spB2.sigValueChanged.connect(self.spinBox2Changed)
        self.plt1 = pg.PlotWidget()
        self.plt2 = pg.PlotWidget()
        self.plt3 = pg.PlotWidget()
        self.plt4 = pg.PlotWidget()
        self.plt5 = pg.PlotWidget()
        self.spB3 = pg.SpinBox(value=self.ma_period1, step=1, bounds=(1,100))
        self.spB3.sigValueChanged.connect(self.spinBox3Changed)




        d1 = Dock("Dock1", size=(1, 1))
        area.addDock(d1, 'left')
        w1 = pg.LayoutWidget()
        w1.addWidget(self.loadBtn, row=0, col=0)
        w1.addWidget(self.removeBtn, row=0, col=1)
        w1.addWidget(self.saveBtn, row=0, col=2)
        w1.addWidget(self.clearBtn, row=0, col=3)
        w1.addWidget(self.plt1, row=1, col=0, colspan=2)
        w1.addWidget(self.plt2, row=1, col=2, colspan=2)
        w1.addWidget(QLabel('tres:'), row=2, col=0)
        w1.addWidget(QLabel('tcrit:'), row=2, col=2)
        w1.addWidget(self.spB1, row=2, col=1)
        w1.addWidget(self.spB2, row=2, col=3)
        w1.addWidget(self.plt3, row=3, col=0, colspan=4)
        w1.addWidget(self.plt4, row=4, col=0, colspan=4)
        w1.addWidget(self.plt5, row=5, col=0, colspan=4)
        w1.addWidget(QLabel('Moving average period:'), row=6, col=0, colspan=2)
        w1.addWidget(self.spB3, row=6, col=2)
        d1.addWidget(w1)

        # Right side
        self.textBox = QTextBrowser()
        self.plt6 = pg.PlotWidget()
        self.plt7 = pg.PlotWidget()
        self.plt8 = pg.PlotWidget()
        self.plt9 = pg.PlotWidget()
        self.plt10 = pg.PlotWidget()
        self.spB4 = pg.SpinBox(value=self.min_op, step=1, bounds=(1,100))
        self.spB4.sigValueChanged.connect(self.spinBox4Changed)
        self.spB5 = pg.SpinBox(value=self.ma_period2, step=1, bounds=(1,100))
        self.spB5.sigValueChanged.connect(self.spinBox5Changed)
        self.spB6 = pg.SpinBox(value=self.curr_cluster+1, step=1) # , bounds=(1,100))
        self.spB6.sigValueChanged.connect(self.spinBox6Changed)

        d2 = Dock("Dock2", size=(1, 1))
        area.addDock(d2, 'right')
        w2 = pg.LayoutWidget()
        w2.addWidget(self.textBox, row=0, col=0, colspan=4)
        w2.addWidget(self.plt6, row=1, col=0, colspan=2)
        w2.addWidget(self.plt7, row=1, col=2, colspan=2)
        w2.addWidget(QLabel('Use clusters with number of openings more than:'), row=2, col=0, colspan=3)
        w2.addWidget(self.spB4, row=2, col=3)
        w2.addWidget(self.plt8, row=3, col=0, colspan=2)
        w2.addWidget(self.plt9, row=3, col=2, colspan=2)
        w2.addWidget(self.plt10, row=4, col=0, colspan=4)
        w2.addWidget(QLabel('Moving average period for cluster:'), row=5, col=0)
        w2.addWidget(self.spB5, row=5, col=1)
        w2.addWidget(QLabel('Cluster #:'), row=5, col=2)
        w2.addWidget(self.spB6, row=5, col=3)

        d2.addWidget(w2)

    def update(self):

        self.recs[-1].tres = self.tres
        self.recs[-1].tcrit = self.tcrit
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

        ox, oy, dx = scpl.prepare_hist(np.array(self.recs[-1].opint),
            self.tres)
        self.plt1.plot(ox, oy, stepMode=True, fillLevel=0,
            brush=(0, 0, 255, 80))
        self.tresLine1 = pg.InfiniteLine(angle=90, movable=True, pen='r')
        self.tresLine1.setValue(log10(self.recs[-1].tres))
        self.tresLine1.sigPositionChangeFinished.connect(self.tresLine1Changed)
        self.plt1.addItem(self.tresLine1)
        self.plt1.setLogMode(x=True, y=False)
        self.plt1.setLabel('bottom', "Open periods", units='s')
        self.plt1.setLabel('left', "Count #")

        sx, sy, dx = scpl.prepare_hist(np.array(self.recs[-1].shint),
            self.recs[-1].tres)
        self.plt2.plot(sx, sy, stepMode=True, fillLevel=0,
            brush=(0, 0, 255, 80))
        self.tresLine2 = pg.InfiniteLine(angle=90, movable=True, pen='r')
        self.tresLine2.setValue(log10(self.recs[-1].tres))
        self.tcritLine = pg.InfiniteLine(angle=90, movable=True, pen='y')
        self.tcritLine.setValue(log10(self.recs[-1].tcrit))
        self.tresLine2.sigPositionChangeFinished.connect(self.tresLine2Changed)
        self.tcritLine.sigPositionChangeFinished.connect(self.tcritLineChanged)
        self.plt2.addItem(self.tresLine2)
        self.plt2.addItem(self.tcritLine)
        self.plt2.setLogMode(x=True, y=False)
        self.plt2.setLabel('bottom', "Shut periods",  units='s') #, units='ms')
        self.plt2.setLabel('left', "Count #")
        self.spB1.setValue(self.tres)
        self.spB2.setValue(self.tcrit)
        self.spB3.setValue(self.ma_period1)



        opma = moving_average(self.recs[-1].opint, self.ma_period1)
        shma = moving_average(self.recs[-1].shint, self.ma_period1)
        poma = opma / (opma + shma)
        self.plt3.plot(opma, stepMode=True,pen='r')
        self.plt3.setLabel('left', "Open periods", units='s')
        self.plt3.setLabel('bottom', "Interval #")
        self.plt4.plot(shma, stepMode=True,pen='b')
        self.plt4.setLabel('left', "Shut periods", units='s')
        self.plt4.setLabel('bottom', "Interval #")
        self.plt5.plot(poma, stepMode=True,pen='g')
        self.plt5.setLabel('left', "Popen")
        self.plt5.setLabel('bottom', "Interval #")
        self.plt5.setYRange(0, 1)

        all_popen = []
        all_mean_ampl = []
        opav = []
        all_op_lists = []
        all_sh_lists = []
        cluster_num = 0
        for record in self.recs:
            clusters = record.bursts.get_long(self.min_op)
            cluster_num += clusters.count()
            all_popen.extend(clusters.get_popen_list())
            all_mean_ampl.extend(clusters.get_mean_ampl_list())
            opav.extend(clusters.get_opening_length_mean_list())
            all_op_lists.extend(clusters.get_op_lists())
            all_sh_lists.extend(clusters.get_sh_lists())
        y,x = np.histogram(np.array(all_popen)) #, bins=np.linspace(-3, 8, 40))
        hist = pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 80))
        self.plt6.addItem(hist)
        self.plt6.setXRange(0, 1) #, padding=None, update=True)
        self.plt6.setLabel('bottom', "Popen")
        self.plt6.setLabel('left', "Count #")
        
        y,x = np.histogram(np.array(opav)) #, bins=np.linspace(-3, 8, 40))
        hist = pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 80))
        self.plt7.addItem(hist)
        self.plt7.setLabel('bottom', "Opening mean length", units='s')
        self.plt7.setLabel('left', "Count #")
        
        self.plt8.plot(np.array(all_popen), np.array(opav)*1000,  pen=None, symbol='o', symbolPen='b', symbolSize=5, symbolBrush='b')
        self.plt8.setLabel('left', "Opening mean length", units='ms')
        self.plt8.setLabel('bottom', "Popen")
        self.plt8.setXRange(0, 1)

        #self.plt9.plot(np.array(all_popen), np.array(all_mean_ampl),  pen=None, symbol='o', symbolPen='b', symbolSize=5, symbolBrush='b')
        self.plt9.setLabel('left', "Mean amplitude", units='pA')
        self.plt9.setLabel('bottom', "Popen")
        self.plt9.setXRange(0, 1)

        copma = moving_average(all_op_lists[self.curr_cluster][:-1], self.ma_period2)
        cshma = moving_average(all_sh_lists[self.curr_cluster], self.ma_period2)
        cpoma = copma / (copma + cshma)
        self.plt10.plot(cpoma, stepMode=True,pen='g')
        self.plt10.setLabel('left', "Popen")
        self.plt10.setLabel('bottom', "Interval #")
        self.plt10.setYRange(0, 1)

        self.spB4.setValue(self.min_op)
        self.spB5.setValue(self.ma_period2)
        self.spB6.setValue(self.curr_cluster+1)
        self.spB6.setMaximum(cluster_num)
        self.spB6.setMinimum(1)

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
    def spinBox3Changed(self):
        val = self.spB3.value()
        self.ma_period1 = val
        self.update()
    def spinBox4Changed(self):
        val = self.spB4.value()
        self.min_op = val
        self.update()
    def spinBox5Changed(self):
        val = self.spB5.value()
        self.ma_period2 = val
        self.update()
    def spinBox6Changed(self):
        val = self.spB6.value()
        self.curr_cluster = int(val-1)
        self.update()
        
    def load(self):
        filename, filt = QFileDialog.getOpenFileName(self,
            "Open CSV file (Clampfit idealised data saved in EXCEL csv file)...",
            self.path, "CSV file  (*.csv)")
        self.path, fname = os.path.split(filename)
        #self.textBox.append('Loaded record from file: '+filename+'\n')
        fscname = convert_clampfit_to_scn(filename)
        self.textBox.append('Converted to SCAN file: '+fscname+'\n')
        self.recs.append(dataset.SCRecord([fscname]))
        
        #self.textBox.append('Record in ' + filename + ' contains {0:d} clusters '.
        #    format(self.recs[-1].bursts.count()) + 'with average Popen = {0:.3f}; '.
        #    format(self.recs[-1].bursts.get_popen_mean()) + 'tcrit = {0:.1f} ms\n'.
        #    format(self.recs[-1].tcrit * 1000))
        #for cluster in self.recs[-1].bursts.all():
        #    self.textBox.append(str(cluster))
            
        self.update()
        self.clearBtn.setEnabled(True)
        self.saveBtn.setEnabled(True)
        self.removeBtn.setEnabled(True)
        
    def save(self):
        pass
    def remove(self):
        self.recs.pop()
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
        self.recs = []
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
    form = ClusterInspector()
    form.show()
    app.exec_()