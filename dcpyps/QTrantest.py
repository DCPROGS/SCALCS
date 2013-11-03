#! /usr/bin/python
import os
from PySide.QtCore import *
from PySide.QtGui import *

import dcstatslib as dcstats
import myqtlibs.myqtcommon as myqtcommon

__author__="remis"
__date__ ="$03-Jan-2010 15:26:00$"

class rantestQT(QDialog):
    def __init__(self, parent=None):
        super(rantestQT, self).__init__(parent)
        tab_widget = QTabWidget()
        tab1 = QWidget()
        tab1.setStyleSheet("QWidget { background-color: %s }"% "white")
        tab2 = QWidget()
        tab3 = QWidget()
        tab4 = QWidget()
        tab_widget.addTab(tab1, "Wellcome!")
        tab_widget.addTab(tab2, "Rantest: continuous")
        tab_widget.addTab(tab3, "Rantest: binary")
        tab_widget.addTab(tab4, "Fieller")

        self.nran = 5000
        self.paired = 0
        self.rancon_data = []
        self.rancon_data_source = ''
        self.path = ""

        ####### Tabs ##########
        movie_screen = self.movie_screen()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.addWidget(QLabel("<p align=center><b>Welcome to DC_PyPs: "
        "Statistics!</b></p>"))
        tab1_layout.addWidget(movie_screen)
        tab1_layout.addWidget(QLabel("<p align=center><b>To continue select a "
        "statistical test from visible tabs.</b></p>"))

        tab2_layout = QVBoxLayout(tab2)
        tab2_layout = self.rancont_layout(tab2_layout)

        tab3_layout = QVBoxLayout(tab3)
        tab3_layout = self.ranbin_layout(tab3_layout)

        tab4_layout = QVBoxLayout(tab4)
        tab4_layout = self.fieller_layout(tab4_layout)

        ##### Finalise main window ######
        vbox = QVBoxLayout()
        vbox.addWidget(tab_widget)
        quitButton = QPushButton("&QUIT")
        self.connect(quitButton, SIGNAL("clicked()"), self, SLOT("close()"))
        b_layout = QHBoxLayout()
        b_layout.addStretch()
        b_layout.addWidget(quitButton)
        b_layout.addStretch()
        vbox.addLayout(b_layout)
        self.setLayout(vbox)
        self.setWindowTitle("DC_PyPs: Statistics")

#######   TAB 4: FIELLER. START  #############
    def fieller_layout(self, tab_layout):
        'Prepare layout for Tab 4. Fieller theorema.'
        
        tab_layout.addWidget(QLabel(dcstats.intro_fieller))

        grid = QGridLayout()
        grid.addWidget(QLabel("Nominator:"), 0, 0)
        grid.addWidget(QLabel("SD of Nominator:"), 1, 0)
        grid.addWidget(QLabel("Denominator:"), 2, 0)
        grid.addWidget(QLabel("SD of Denominator:"), 3, 0)
        grid.addWidget(QLabel("Correlation coefficient (nom,denom):"), 4, 0)
        grid.addWidget(QLabel("Student's t value:"), 5, 0)
        self.tb4e1 = QLineEdit("14")
        grid.addWidget(self.tb4e1, 0, 1)
        self.tb4e2 = QLineEdit("3")
        grid.addWidget(self.tb4e2, 1, 1)
        self.tb4e3 = QLineEdit("7")
        grid.addWidget(self.tb4e3, 2, 1)
        self.tb4e4 = QLineEdit("2")
        grid.addWidget(self.tb4e4, 3, 1)
        self.tb4e5 = QLineEdit("0")
        grid.addWidget(self.tb4e5, 4, 1)
        self.tb4e6 = QLineEdit("2")
        grid.addWidget(self.tb4e6, 5, 1)
        tab_layout.addLayout(grid)

        self.tb4b1 = QPushButton("Calculate")
        b_layout = QHBoxLayout()
        b_layout.addStretch()
        b_layout.addWidget(self.tb4b1)
        b_layout.addStretch()
        tab_layout.addLayout(b_layout)

        self.tb4txt = QTextBrowser()
        self.tb4txt.append("RESULT WILL BE DISPLAYED HERE")
        tab_layout.addWidget(self.tb4txt)
        self.connect(self.tb4b1, SIGNAL("clicked()"), self.callback2)
        
        return tab_layout

    def callback2(self):
        'Called by CALCULATE button in Tab4.'
        a = float(self.tb4e1.text())
        b = float(self.tb4e3.text())
        sa = float(self.tb4e2.text())
        sb = float(self.tb4e4.text())
        r = float(self.tb4e5.text())
        tval = float(self.tb4e6.text())
        self.tb4txt.clear()
        log = myqtcommon.PrintLog(self.tb4txt) #, sys.stdout)
        #Call Fieller to calculate statistics.
        dcstats.fieller_printout(a,b, sa, sb, r, tval, output=log)
#######   TAB 4: FIELLER. END  #############

#######   TAB 3: RANTEST FOR BINARY DATA. START  #############
    def ranbin_layout(self, tab_layout):
        """ """
        tab_layout.addWidget(QLabel(dcstats.intro_randomisation))
        tab_layout.addWidget(QLabel("Sample 1"))
        layout1 = QHBoxLayout()
        layout1.addWidget(QLabel("Successes:"))
        self.tb3e1 = QLineEdit("3")
        layout1.addWidget(self.tb3e1)
        layout1.addWidget(QLabel("Failures:"))
        self.tb3e2 = QLineEdit("4")
        layout1.addWidget(self.tb3e2)
        layout1.addStretch()
        tab_layout.addLayout(layout1)

        tab_layout.addWidget(QLabel("Sample 2"))
        layout2 = QHBoxLayout()
        layout2.addWidget(QLabel("Successes:"))
        self.tb3e3 = QLineEdit("4")
        layout2.addWidget(self.tb3e3)
        layout2.addWidget(QLabel("Failures:"))
        self.tb3e4 = QLineEdit("5")
        layout2.addWidget(self.tb3e4)
        layout2.addStretch()
        tab_layout.addLayout(layout2)

        layout3 = QHBoxLayout()
        layout3.addWidget(QLabel("Number of randomisations:"))
        self.tb3e5 = QLineEdit("5000")
        layout3.addWidget(self.tb3e5)
        layout3.addStretch()
        tab_layout.addLayout(layout3)
        
        self.tb3b1 = QPushButton("Calculate")
        b_layout = QHBoxLayout()
        b_layout.addStretch()
        b_layout.addWidget(self.tb3b1)
        b_layout.addStretch()
        tab_layout.addLayout(b_layout)

        self.tb3txt = QTextBrowser()
        self.tb3txt.append("RESULT WILL BE DISPLAYED HERE")
        tab_layout.addWidget(self.tb3txt)
        self.connect(self.tb3b1, SIGNAL("clicked()"), self.callback3)

        return tab_layout

    def callback3(self):
        """Called by button CALCULATE."""
        ir1 = int(self.tb3e1.text())
        if1 = int(self.tb3e2.text())
        ir2 = int(self.tb3e3.text())
        if2 = int(self.tb3e4.text())
        nran = int(self.tb3e5.text())
        self.tb3txt.clear()
        log = PrintLog(self.tb3txt)
        dcstats.stats_binomial_printout(ir1, if1, ir2, if2, output=log)
        dcstats.rantest_binomial_printout(ir1, if1, ir2, if2,
            nran, output=log)
#######   TAB 3: RANTEST FOR BINARY DATA. END  #############

#######   TAB 2: RANTEST FOR CONTINUOSLY VARIABLY DATA. START  #############
    def rancont_layout(self, tab_layout):
        """Create Tab2 layout."""
        tab_layout.addWidget(QLabel(dcstats.intro_randomisation))

        self.tb2b1 = QPushButton("Get data")
        b_layout = QHBoxLayout()
        b_layout.addStretch()
        b_layout.addWidget(self.tb2b1)
        b_layout.addStretch()
        tab_layout.addLayout(b_layout)

        layout1 = QHBoxLayout()
        layout1.addWidget(QLabel("Number of randomisations:"))
        self.tb2e5 = QLineEdit(unicode(self.nran))
        layout1.addWidget(self.tb2e5)
        self.tb2c1 = QCheckBox("&Paired test?")
        layout1.addWidget(self.tb2c1)
        tab_layout.addLayout(layout1)

        self.tb2b2 = QPushButton("Run test")
        b_layout = QHBoxLayout()
        b_layout.addStretch()
        b_layout.addWidget(self.tb2b2)
        b_layout.addStretch()
        tab_layout.addLayout(b_layout)

        self.tb2txt = QTextBrowser()
        self.tb2txt.append("RESULT WILL BE DISPLAYED HERE")
        tab_layout.addWidget(self.tb2txt)
        #self.tb2b3 = QPushButton("Plot distribution")
        #tab_layout.addWidget(self.tb2b3)
        self.connect(self.tb2e5, SIGNAL("editingFinished()"), self.ran_changed)
        self.connect(self.tb2c1, SIGNAL("stateChanged(int)"), self.ran_changed)
        self.connect(self.tb2b1, SIGNAL("clicked()"), self.callback1)
        self.connect(self.tb2b2, SIGNAL("clicked()"), self.callback4)
        return tab_layout

    def callback1(self):
        """Called by TAKE DATA FROM FILE button in Tab2"""
        filename, filt = QFileDialog.getOpenFileName(self,
            "Open Data File...", self.path, "Text Data Files (*.txt)")
        self.path = os.path.split(str(filename))[0]
        self.X, self.Y = dcstats.data_from_txt_file(filename)
        self.tb2txt.clear()
        self.tb2txt.append('Data loaded from a text file: ' + filename + '\n')
        log = myqtcommon.PrintLog(self.tb2txt)
        dcstats.stats_continuous_printout(self.X, self.Y,
            self.paired, output=log)
        
    def callback4(self):
        """Called by RUN TEST button in Tab2."""
        log = myqtcommon.PrintLog(self.tb2txt)
        dcstats.rantest_continuous_printout(self.X, self.Y,
            self.paired, self.nran, output=log)

    def ran_changed(self):
        if self.tb2c1.isChecked():
            self.paired = 1
        else:
            self.paired = 0
        self.nran = int(self.tb2e5.text())
#######   TAB 2: RANTEST FOR CONTINUOSLY VARIABLY DATA. START  #############

#######   TAB 1: WELCOME!  START   ############
    def movie_screen(self):
        """Set up the gif movie screen."""
        movie_screen = QLabel()
        # expand and center the label
        movie_screen.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        movie_screen.setAlignment(Qt.AlignCenter)
        movie = QMovie("dca2.gif", QByteArray(), self)
        movie.setCacheMode(QMovie.CacheAll)
        movie.setSpeed(100)
        movie_screen.setMovie(movie)
        movie.start()
        return movie_screen
#######   TAB 1: WELCOME!  END   ############