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

try:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
    from matplotlib.figure import Figure
    from matplotlib import scale as mscale
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter, FuncFormatter
    import matplotlib.pyplot as plt
#    from matplotlib import transforms as mtransforms
#    from matplotlib import ticker
except:
    raise ImportError("matplotlib module is missing")

import numpy as np

from scipy.optimize import curve_fit
from scipy.optimize import leastsq

import scalcslib as scl
import cjumps
import scburst
import popen
import dcio
import samples
import scplotlib as scpl
import mechanism

import optimize
import dataset


def startInfo():
    """
    Get date, time, machine info, etc.
    """
    str1 = "DC_PyPs: HJCFIT, Q matrix calculations, etc."
    str2 = ("Date and time of analysis: %4d/%02d/%02d %02d:%02d:%02d"
        %time.localtime()[0:6])
    machine = socket.gethostname()
    system = sys.platform
    str3 = "Machine: %s; System: %s" %(machine, system)
    return str1, str2, str3


def createAction(self, text, slot=None, shortcut=None, icon=None,
        tip=None, checkable=False, signal="triggered()"):
    """
    Create menu actions.
    """
    action = QAction(text, self)
    if icon is not None:
        action.setIcon(QIcon(":/%s.png" % icon))
    if shortcut is not None:
        action.setShortcut(shortcut)
    if tip is not None:
        action.setToolTip(tip)
        action.setStatusTip(tip)
    if slot is not None:
        self.connect(action, SIGNAL(signal), slot)
    if checkable:
        action.setCheckable(True)
    return action

def addActions(target, actions):
    """
    Add actions to menu.
    """
    for action in actions:
        if action is None:
            target.addSeparator()
        else:
            target.addAction(action)


class PrintLog:
    """
    Write stdout to a QTextEdit.
    out1 = QTextEdit, QTextBrowser, etc.
    out2 = sys.stdout, file, etc.
    """
    def __init__(self, out1, out2=None):
        self.out1 = out1
        self.out2 = out2
    def write(self, text):
        self.out1.append(text.rstrip('\n'))
        if self.out2:
            self.out2.write(text)
            
def addMechMenuElements(self):
    loadMechMenu = self.menuBar().addMenu('&Load Mec')
    loadDemo1Action = createAction(self, "&Load demo: CH82", self.onLoadDemo_CH82,
        None, "loaddemo", "Load Demo mec")
    loadDemo2Action = createAction(self, "&Load demo: dC-K", self.onLoadDemo_dCK,
        None, "loaddemo", "Load Demo mec")
    loadFromMecFileAction = createAction(self, "&Load from DCprogs MEC File...",
        self.onLoadMecFile,
        None, "loadfrommecfile", "Load from Mec file")
    loadFromPrtFileAction = createAction(self, "&Load from DCprogs PRT File...",
        self.onLoadPrtFile,
        None, "loadfromprtfile", "Load from Prt file")
    loadFromModFileAction = createAction(self, "&Load from ChannelLab MOD File...",
        self.onLoadModFile,
        None, "loadfrommodfile", "Load from ChannelLab Mod file")
    modifyMecAction = createAction(self, "&Modify loaded mec rates", self.onModifyMec,
        None, "modifymec", "Modify mec rates")
    modifyStatesAction = createAction(self, "&Modify loaded mec states", self.onModifyStates,
        None, "modifystates", "Modify mec states")
    quitAction = createAction(self, "&Quit", self.close,
        "Ctrl+Q", "appquit", "Close the application")
    addActions(loadMechMenu, (loadDemo1Action, loadDemo2Action,
        loadFromMecFileAction, loadFromPrtFileAction, loadFromModFileAction,
        modifyMecAction, modifyStatesAction, quitAction))
    return loadMechMenu


class RateTableDlg(QDialog):
    """
    """
    def __init__(self, parent=None, mec=None, log=None):
        super(RateTableDlg, self).__init__(parent)
        self.mec = mec
        self.changed = False
        self.log = log

        layoutMain = QVBoxLayout()
        self.table = RateTable(self.mec)
        self.table.display_data()
        self.connect(self.table,
                SIGNAL("cellChanged(int, int)"),
                self.tableItemChanged)
        layoutMain.addWidget(self.table)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setGeometry(100,100,750,550)
        self.setWindowTitle("View and modify rate constants...")

    def tableItemChanged(self, row, column):
        # TODO: update mechanism if anything changed in table

        if column == 3:
            newratecon = float(self.table.item(row, column).text())
            self.mec.Rates[row].rateconstants = newratecon

        if column == 5:
            value = False
            if self.table.item(row, column).checkState() > 0:
                value = True
            self.mec.Rates[row].fixed = value
            print 'fixed value=', value

        if column == 6:
            value = False
            if self.table.item(row, column).checkState() > 0:
                value = True
            self.mec.Rates[row].mr = value
            print 'mr value=', value

        if column == 7 or column == 8 or column == 9:
            #TODO: handle exceptions in this block
            value = False
            if self.table.item(row, 7).checkState() > 0:
                value = True
            self.mec.Rates[row].is_constrained = value
            self.mec.Rates[row].constrain_func = mechanism.constrain_rate_multiple
#            print 'is_constrained =', value
            factor = float(self.table.item(row, 8).text())
#            print 'factor=', factor
            torate = int(self.table.item(row, 9).text()) - 1
#            print 'to rate=', torate
            self.mec.Rates[row].constrain_args = [torate, factor]

        if column == 10 or column == 11:
            newlimits = [float(self.table.item(row, 10).text()),
                float(self.table.item(row, 11).text())]
            self.mec.Rates[row].limits = newlimits

        self.changed = True

    def return_mec(self):
        if self.changed:
            self.mec.update_constrains()
            self.mec.update_mr()
            self.log.write("\n\nMechanism modified:\n")
            self.mec.printout(self.log)
        return self.mec

class RateTable(QTableWidget):
    """ Creates a custom table widget """
    def __init__(self, mec=None, *args):
        QTableWidget.__init__(self, *args)
        self.setSelectionMode(self.ContiguousSelection)
        self.setGeometry(0,0,700,400)
        self.setShowGrid(False)
        self.mec = mec

    def display_data(self):
        """ Reads in data as a 2D list and formats and displays it in
            the table """

        header = ['From State', 'To State', 'Rate name', 'Rate value',
            'Conc depend', 'Fixed', 'MR', 'Constr.', 'Factor', 'To rate',
            'Lower limit', 'Higher limit']

        nrows = len(self.mec.Rates)
        ncols = len(header)
        self.setRowCount(nrows)
        self.setColumnCount(ncols)
        self.setHorizontalHeaderLabels(header)

        for i in xrange(nrows):
            cell = QTableWidgetItem(self.mec.Rates[i].State1.name)
            self.setItem(i, 0, cell)
            cell = QTableWidgetItem(self.mec.Rates[i].State2.name)
            self.setItem(i, 1, cell)
            cell = QTableWidgetItem(self.mec.Rates[i].name)
            self.setItem(i, 2, cell)
            cell = QTableWidgetItem(str(self.mec.Rates[i].unit_rate()))
            self.setItem(i, 3, cell)

            if self.mec.Rates[i].effectors[0] is None:
                eff = ''
            else:
                eff = self.mec.Rates[i].effectors[0]
            cell = QTableWidgetItem(eff)
            self.setItem(i, 4, cell)

            check = QTableWidgetItem()
            value = Qt.Unchecked
            if self.mec.Rates[i].fixed > 0:
                value = Qt.Checked
            check.setCheckState(value)
            self.setItem(i, 5, check)

            check = QTableWidgetItem()
            value = Qt.Unchecked
            if self.mec.Rates[i].mr:
                value = Qt.Checked
            check.setCheckState(value)
            self.setItem(i, 6, check)

            check = QTableWidgetItem()
            value = Qt.Unchecked
            if self.mec.Rates[i].is_constrained:
                value = Qt.Checked
            check.setCheckState(value)
            self.setItem(i, 7, check)
            factor = ''
            torate = ''
            if self.mec.Rates[i].constrain_args:
                factor = self.mec.Rates[i].constrain_args[1]
                torate = self.mec.Rates[i].constrain_args[0] + 1
            cell = QTableWidgetItem(str(factor))
            self.setItem(i, 8, cell)
            cell = QTableWidgetItem(str(torate))
            self.setItem(i, 9, cell)

            if len(self.mec.Rates[i].limits) == 0:
                if eff == '':
                    limits = [[1e-3,1e+7]]
                else:
                    limits = [[1e-3,1e+10]]
            else:
                limits = self.mec.Rates[i].limits
            cell = QTableWidgetItem(str(limits[0][0]))
            self.setItem(i, 10, cell)
            cell = QTableWidgetItem(str(limits[0][1]))
            self.setItem(i, 11, cell)


        self.resizeColumnsToContents()
        self.resizeRowsToContents()

class StateTableDlg(QDialog):
    """
    """
    def __init__(self, parent=None, mec=None, log=None):
        super(StateTableDlg, self).__init__(parent)
        self.mec = mec
        self.changed = False
        self.log = log

        layoutMain = QVBoxLayout()
        self.table = StateTable(self.mec)
        self.table.display_data()
        self.connect(self.table,
                SIGNAL("cellChanged(int, int)"),
                self.tableItemChanged)
        layoutMain.addWidget(self.table)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setGeometry(100,100,450,350)
        self.setWindowTitle("View and modify states...")

    def tableItemChanged(self, row, column):
        # TODO: update mechanism if anything changed in table

#        if column == 0:
#            newname = self.table.item(row, column).text()
#            self.mec.States[row].name = newname

        if column == 1:
            newtype = self.table.item(row, column).text().upper()
            self.mec.States[row].statetype = newtype

        if column == 2:
            newgamma = float(self.table.item(row, column).text())
            self.mec.Rates[row].conductance = newgamma

        self.changed = True

    def return_mec(self):
        if self.changed:
            self.mec.update_states()
            self.log.write("\n\nMechanism states modified:\n")
            self.mec.printout(self.log)
        return self.mec

class StateTable(QTableWidget):
    """ Creates a custom table widget """
    def __init__(self, mec=None, *args):
        QTableWidget.__init__(self, *args)
        self.setSelectionMode(self.ContiguousSelection)
        self.setGeometry(0,0,700,400)
        self.setShowGrid(False)
        self.mec = mec

    def display_data(self):
        """ Reads in data as a 2D list and formats and displays it in
            the table """

        header = ['State name', 'State Class', 'Conductance',
            'Number of connections']

        nrows = len(self.mec.States)
        ncols = len(header)
        self.setRowCount(nrows)
        self.setColumnCount(ncols)
        self.setHorizontalHeaderLabels(header)

        for i in xrange(nrows):
            cell = QTableWidgetItem(self.mec.States[i].name)
            self.setItem(i, 0, cell)
            cell = QTableWidgetItem(self.mec.States[i].statetype)
            self.setItem(i, 1, cell)
            cell = QTableWidgetItem(str(self.mec.States[i].conductance))
            self.setItem(i, 2, cell)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()
