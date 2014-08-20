import sys
import socket
import datetime

try:
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")

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
        self.cEdit = QLineEdit(unicode(self.conc))
        self.cEdit.setMaxLength(12)
        self.connect(self.cEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.cEdit)
        layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

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
        self.rEdit = QLineEdit(unicode(self.tres))
        self.rEdit.setMaxLength(12)
        self.connect(self.rEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.rEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Correct for fast block "))
        self.fbCheck = QCheckBox()
        self.fbCheck.setCheckState(Qt.Unchecked)
        self.connect(self.fbCheck, SIGNAL("stateChanged()"),
            self.on_par_changed)
        layout.addWidget(self.fbCheck)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Fast block equilibrium constant KB (mM):"))
        self.fbEdit = QLineEdit(unicode(self.KB))
        self.fbEdit.setMaxLength(12)
        self.connect(self.fbEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.fbEdit)
        layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

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
        self.conc1Edit = QLineEdit(unicode(self.cmin))
        self.conc1Edit.setMaxLength(12)
        self.connect(self.conc1Edit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.conc1Edit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("End concentration (mM):"))
        self.conc2Edit = QLineEdit(unicode(self.cmax))
        self.conc2Edit.setMaxLength(12)
        self.connect(self.conc2Edit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.conc2Edit)
        layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

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
        self.u1Edit = QLineEdit(unicode(self.u1))
        self.u1Edit.setMaxLength(10)
        self.connect(self.u1Edit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.u1Edit)
        
        layout.addWidget(QLabel("To shut time (ms):"))
        self.u2Edit = QLineEdit(unicode(self.u2))
        self.u2Edit.setMaxLength(10)
        self.connect(self.u2Edit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.u2Edit)
        layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

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
        self.cEdit = QLineEdit(unicode(self.conc))
        self.cEdit.setMaxLength(12)
        self.connect(self.cEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.cEdit)
        
        layout.addWidget(QLabel("Resolution (microsec):"))
        self.rEdit = QLineEdit(unicode(self.tres))
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
        self.setWindowTitle("Concentration and resolution...")

    def on_par_changed(self):
        self.conc = float(self.cEdit.text())
        self.tres = float(self.rEdit.text())

    def return_par(self):
        return self.conc * 1e-6, self.tres * 1e-6 # Return tcrit in sec

