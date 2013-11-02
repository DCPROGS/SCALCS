import numpy as np
import matplotlib.pyplot as plt
try:
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")

from dcpyps import scplotlib as scpl
from dcpyps import scalcslib as scl
from dcpyps import popen
import myqtcommon

class HelpMenu(QMenu):
    """
    """
    def __init__(self, parent):
        super(HelpMenu, self).__init__(parent) 
        self.parent = parent
        self.setTitle('&Help')
        
        helpAboutAction = myqtcommon.createAction(self,
            "&About", self.onHelpAbout)
        self.addActions([helpAboutAction])
        
    def onHelpAbout(self):
        """
        Display About dialog.
        Called from menu Help|About.
        """
        dialog = AboutDlg(self)
        if dialog.exec_():
            pass

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
