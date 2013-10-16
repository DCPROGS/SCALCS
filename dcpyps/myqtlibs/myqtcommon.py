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
            

class AboutDlg(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self)

        okButton = QPushButton("&OK")
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(okButton)
        button_layout.addStretch()

        movie_screen = self.movie_screen()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<p align=center><b>Welcome to DC_PyPs: "
        "Q matrix calculations!</b></p>"))
        layout.addWidget(movie_screen)
        layout.addLayout(button_layout)

        self.connect(okButton, SIGNAL("clicked()"),
        self, SLOT("accept()"))
        self.setStyleSheet("QWidget { background-color: %s }"% "white")
        self.setWindowTitle("About DC_PyPs: Q matrix calculations")

    def movie_screen(self):
        """
        Set up the gif movie screen.
        """
        movie_screen = QLabel()
        # Expand and center the label
        movie_screen.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        movie_screen.setAlignment(Qt.AlignCenter)
        movie = QMovie("dca2.gif", QByteArray(), self)
        movie.setCacheMode(QMovie.CacheAll)
        movie.setSpeed(100)
        movie_screen.setMovie(movie)
        movie.start()
        return movie_screen
