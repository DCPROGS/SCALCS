import sys
import socket
import datetime

try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
except:
    raise ImportError("pyqt module is missing")

try:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    from matplotlib import scale as mscale
except:
    raise ImportError("matplotlib module is missing")

def startInfo(log):
    """
    Get date, time, machine info, etc.
    """
    log.write("DC_PyPs: HJCFIT, Q matrix calculations, etc.")
    log.write("Date and time of analysis: " + str(datetime.datetime.now())[:19])
    machine = socket.gethostname()
    system = sys.platform
    log.write("Machine: {0};   System: {1}".format(machine, system))

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
        action.triggered.connect(slot)
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

def ok_cancel_button(parent):
    buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
    buttonBox.button(QDialogButtonBox.Ok).setDefault(True)
    buttonBox.accepted.connect(parent.accept)
    buttonBox.rejected.connect(parent.reject)
    return buttonBox


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
        self.out1.append(text.strip('\n'))
        if self.out2:
            self.out2.write(text)
                
class ConcDlg(QDialog):
    """
    Dialog to input concentration.
    """
    def __init__(self, parent=None, conc=100e-9):
        super(ConcDlg, self).__init__(parent)

        self.conc = conc * 1e6 # in microM

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Enter concentration:"))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Concentration (microM):"))
        self.cEdit = QLineEdit(str(self.conc))
        self.cEdit.setMaxLength(12)
        self.cEdit.editingFinished.connect(self.on_par_changed)
        layout.addWidget(self.cEdit)
        layoutMain.addLayout(layout)

        layoutMain.addWidget(ok_cancel_button(self))
        self.setLayout(layoutMain)
        self.setWindowTitle("Concentration...")

    def on_par_changed(self):
        self.conc = float(self.cEdit.text())

    def return_par(self):
        return self.conc * 1e-6
    
class ResDlg(QDialog):
    """
    Dialog to input resolution.
    """
    def __init__(self, parent=None, tres=25e-6):
        super(ResDlg, self).__init__(parent)

        self.tres = tres * 1e6 # in microsec
        self.KB = 1.0
        self.fastBl = False

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Enter resolution:"))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Resolution (microsec):"))
        self.rEdit = QLineEdit(str(self.tres))
        self.rEdit.setMaxLength(12)
        self.rEdit.editingFinished.connect(self.on_par_changed)
        layout.addWidget(self.rEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Correct for fast block "))
        self.fbCheck = QCheckBox()
        self.fbCheck.setCheckState(Qt.Unchecked)
        self.fbCheck.stateChanged.connect(self.on_par_changed)
        layout.addWidget(self.fbCheck)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Fast block equilibrium constant KB (mM):"))
        self.fbEdit = QLineEdit(str(self.KB))
        self.fbEdit.setMaxLength(12)
        self.fbEdit.editingFinished.connect(self.on_par_changed)
        layout.addWidget(self.fbEdit)
        layoutMain.addLayout(layout)

        layoutMain.addWidget(ok_cancel_button(self))
        self.setLayout(layoutMain)
        self.setWindowTitle("Resolution...")

    def on_par_changed(self):
        self.tres = float(self.rEdit.text())
        self.fastBl = self.fbCheck.isChecked()
        self.KB = float(self.fbEdit.text())

    def return_par(self):
        # Return tcrit in sec, KB in M
        return self.tres * 1e-6, self.fastBl, self.KB * 1e-3

class ConcRangeDlg(QDialog):
    """
    Dialog to get concentration range.
    """
    def __init__(self, parent=None, cmin=1e-6, cmax=0.001):
        super(ConcRangeDlg, self).__init__(parent)

        self.cmin = cmin * 1000 # in mM.
        self.cmax = cmax * 1000 # in mM.

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Enter concentrations:"))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Start concentration (mM):"))
        self.conc1Edit = QLineEdit(str(self.cmin))
        self.conc1Edit.setMaxLength(12)
        self.conc1Edit.editingFinished.connect(self.on_par_changed)
        layout.addWidget(self.conc1Edit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("End concentration (mM):"))
        self.conc2Edit = QLineEdit(str(self.cmax))
        self.conc2Edit.setMaxLength(12)
        self.conc2Edit.editingFinished.connect(self.on_par_changed)
        layout.addWidget(self.conc2Edit)
        layoutMain.addLayout(layout)

        layoutMain.addWidget(ok_cancel_button(self))
        self.setLayout(layoutMain)
        self.setWindowTitle("Concentration range...")

    def on_par_changed(self):
        """
        """
        self.cmin = float(self.conc1Edit.text()) * 0.001
        self.cmax = float(self.conc2Edit.text()) * 0.001

    def return_par(self):
        """
        Return parameter dictionary on exit.
        """
        return self.cmin, self.cmax
    
class ShutRangeDlg(QDialog):
    """
    Dialog to input shut time range.
    """
    def __init__(self, parent=None):
        super(ShutRangeDlg, self).__init__(parent)

        self.u1 = 0.001 # 1 ms
        self.u2 = 0.01 # 10 ms

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Shut time range:"))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("From shut time (ms):"))
        self.u1Edit = QLineEdit(str(self.u1))
        self.u1Edit.setMaxLength(10)
        self.u1Edit.editingFinished.connect(self.on_par_changed)
        layout.addWidget(self.u1Edit)
        
        layout.addWidget(QLabel("To shut time (ms):"))
        self.u2Edit = QLineEdit(str(self.u2))
        self.u2Edit.setMaxLength(10)
        self.u2Edit.editingFinished.connect(self.on_par_changed)
        layout.addWidget(self.u2Edit)
        layoutMain.addLayout(layout)

        layoutMain.addWidget(ok_cancel_button(self))
        self.setLayout(layoutMain)
        self.setWindowTitle("Shut time range...")

    def on_par_changed(self):
        self.u1 = float(self.u1Edit.text())
        self.u2 = float(self.u2Edit.text())

    def return_par(self):
        return self.u1 * 0.001, self.u2 * 0.001 # Return tcrit in sec

class ConcResDlg(QDialog):
    """
    Dialog to input concentration and resolution.
    """
    def __init__(self, parent=None, conc=100e-9, tres=25e-6):
        super(ConcResDlg, self).__init__(parent)

        self.conc = conc * 1e6 # in microM
        self.tres = tres * 1e6 # in microsec

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Enter concentration and resolution:"))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Concentration (microM):"))
        self.cEdit = QLineEdit(str(self.conc))
        self.cEdit.setMaxLength(12)
        self.cEdit.editingFinished.connect(self.on_par_changed)
        layout.addWidget(self.cEdit)
        
        layout.addWidget(QLabel("Resolution (microsec):"))
        self.rEdit = QLineEdit(str(self.tres))
        self.rEdit.setMaxLength(12)
        self.rEdit.editingFinished.connect(self.on_par_changed)
        layout.addWidget(self.rEdit)
        layoutMain.addLayout(layout)
        layoutMain.addWidget(ok_cancel_button(self))

        self.setLayout(layoutMain)
        self.setWindowTitle("Concentration and resolution...")

    def on_par_changed(self):
        self.conc = float(self.cEdit.text())
        self.tres = float(self.rEdit.text())

    def return_par(self):
        return self.conc * 1e-6, self.tres * 1e-6 # Return tcrit in sec

class MatPlotWin(FigureCanvas):
    """
    """
    def __init__(self, size=(6.0, 4.0), fsize=8):
        # Prepare matplotlib plot window
        self.fig = Figure(size, dpi=85)
        self.axes = self.fig.add_subplot(111)
        self.axes.autoscale_view(True,True,True)
        self.fontsize = fsize
        for loc, spine in self.axes.spines.items(): #iteritems():
            if loc in ['right','top']:
                spine.set_color('none') # don't draw spine
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        for label in self.axes.xaxis.get_ticklabels():
            label.set_fontsize(self.fontsize)
        for label in self.axes.yaxis.get_ticklabels():
            label.set_fontsize(self.fontsize)
#        self.mplTools = NavigationToolbar(self.canvas, self.parent)
        FigureCanvas.__init__(self, self.fig) 
        
class MatPlotTools(NavigationToolbar):
    """
    """
    def __init__(self, canvas, parent):
        NavigationToolbar.__init__(self, canvas, parent)