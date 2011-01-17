#! /usr/bin/python
"""
A simple GUI for DC_PyPs project.
Depends on pyqt and matplotlib modules.
"""
import time
import sys
import os
import socket

try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
except:
    raise ImportError("pyqt module is missing")

try:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
    from matplotlib.figure import Figure
except:
    raise ImportError("matplotlib module is missing")

import scalcslib as scl
import readmecfile as readmec
import samples

class QMatGUI(QMainWindow):
    def __init__(self, parent=None):
        super(QMatGUI, self).__init__(parent)
        self.resize(1000, 700)     # wide, high in px
        self.mainFrame = QWidget()

        self.mec = samples.CH82()
        self.mec.KB = 0.01
        self.mec.fastBlk = False
        self.conc = 100e-9    # 100 nM
        self.tres = 0.00004
        self.tmin = 10e-6
        self.tmax = 0.1
        self.cmin = 10e-9
        self.cmax = 0.1

        loadMenu = self.menuBar().addMenu('&Load')
        loadDemoAction = self.createAction("&Demo", self.onLoadDemo,
            None, "loaddemo", "Load Demo mec")
        loadFromMecFileAction = self.createAction("&From Mec File...",
            self.onLoadFromFile,
            None, "loadfrommecfile", "Load from Mec file")
        self.addActions(loadMenu, (loadDemoAction,
            loadFromMecFileAction))

        plotMenu = self.menuBar().addMenu('&Plot')
        plotPopenAction = self.createAction("&Popen curve", self.onPlotPopen)
        plotOpenTimePDFAction = self.createAction(
            "&Open time pdf", self.onPlotOpenTimePDF)
        plotShutTimePDFAction = self.createAction(
            "&Shut time pdf", self.onPlotShutTimePDF)
        plotBurstLenPDFAction = self.createAction(
            "&Burst length pdf", self.onPlotBrstLenPDF)
        plotBurstOpeningDistrAction = self.createAction(
            "&Burst openings distribution", self.onPlotBrstOpDistr)
        plotBurstLenVConcAction = self.createAction(
            "&Burst length vs concentration", self.onPlotBrstLenConc)
        self.addActions(plotMenu, (plotPopenAction, plotOpenTimePDFAction,
            plotShutTimePDFAction, plotBurstLenPDFAction,
            plotBurstOpeningDistrAction, plotBurstLenVConcAction))

        helpMenu = self.menuBar().addMenu('&Help')
        helpAboutAction = self.createAction("&About", self.onHelpAbout)
        self.addActions(helpMenu, (helpAboutAction, None))

        self.dpi = 85
        self.fig = Figure((6.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.mainFrame)
        self.axes = self.fig.add_subplot(111)
        for loc, spine in self.axes.spines.iteritems():
            if loc in ['right','top']:
                spine.set_color('none') # don't draw spine
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.mplTools = NavigationToolbar(self.canvas, self.mainFrame)

        self.textBox = QTextBrowser()
        str1, str2, str3 = self.startInfo()
        self.textBox.append(str1)
        self.textBox.append(str2)
        self.textBox.append(str3)
        self.setWindowTitle(str1)

        plotSetLayout = QVBoxLayout()
        mainLabel = QLabel("Set parameters for plots:")
        plotSetLayout.addWidget(mainLabel)

        tresBox = QHBoxLayout()
        tresBox.addWidget(QLabel("Temporal resolution ="))
        self.tresEdit = QLineEdit(unicode(self.tres * 1000000))
        self.tresEdit.setMaxLength(8)
        self.connect(self.tresEdit, SIGNAL("editingFinished()"),
            self.on_settings_changed)
        tresBox.addWidget(self.tresEdit)
        tresBox.addWidget(QLabel("mikrosec"))
        tresBox.addStretch()
        plotSetLayout.addLayout(tresBox)

        trangeBox = QHBoxLayout()
        trangeBox.addWidget(QLabel("Time range: min "))
        self.trangeEdit1 = QLineEdit(unicode(self.tmin * 1000))
        self.trangeEdit1.setMaxLength(8)
        self.connect(self.trangeEdit1, SIGNAL("editingFinished()"),
            self.on_settings_changed)
        trangeBox.addWidget(self.trangeEdit1)
        trangeBox.addWidget(QLabel("millisec,  max "))
        self.trangeEdit2 = QLineEdit(unicode(self.tmax * 1000))
        self.trangeEdit2.setMaxLength(8)
        self.connect(self.trangeEdit2, SIGNAL("editingFinished()"),
            self.on_settings_changed)
        trangeBox.addWidget(self.trangeEdit2)
        trangeBox.addWidget(QLabel("millisec"))
        trangeBox.addStretch()
        plotSetLayout.addLayout(trangeBox)

        concBox = QHBoxLayout()
        concBox.addWidget(QLabel("Concentration = "))
        self.concEdit = QLineEdit(unicode(self.conc * 1000000))
        self.concEdit.setMaxLength(8)
        self.connect(self.concEdit, SIGNAL("editingFinished()"),
            self.on_settings_changed)
        concBox.addWidget(self.concEdit)
        concBox.addWidget(QLabel("mikroM"))
        concBox.addStretch()
        plotSetLayout.addLayout(concBox)

        crangeBox = QHBoxLayout()
        crangeBox.addWidget(QLabel("Concentration range: min "))
        self.crangeEdit1 = QLineEdit(unicode(self.cmin * 1000000))
        self.crangeEdit1.setMaxLength(8)
        self.connect(self.crangeEdit1, SIGNAL("editingFinished()"),
            self.on_settings_changed)
        crangeBox.addWidget(self.crangeEdit1)
        crangeBox.addWidget(QLabel("mikroM,  max "))
        self.crangeEdit2 = QLineEdit(unicode(self.cmax * 1000000))
        self.crangeEdit2.setMaxLength(8)
        self.connect(self.crangeEdit2, SIGNAL("editingFinished()"),
            self.on_settings_changed)
        crangeBox.addWidget(self.crangeEdit2)
        crangeBox.addWidget(QLabel("mikroM"))
        crangeBox.addStretch()
        plotSetLayout.addLayout(crangeBox)

        fastBlkBox = QHBoxLayout()
        self.fastBlkCheckBox = QCheckBox("&Fast block?")
        self.fastBlkCheckBox.setChecked(self.mec.fastBlk)
        self.connect(self.fastBlkCheckBox, SIGNAL("stateChanged(int)"),
            self.on_settings_changed)
        fastBlkBox.addWidget(self.fastBlkCheckBox)
        fastBlkBox.addWidget(QLabel("KB ="))
        self.KBEdit = QLineEdit(unicode(self.mec.KB * 1000))
        self.KBEdit.setMaxLength(8)
        self.connect(self.KBEdit, SIGNAL("editingFinished()"),
            self.on_settings_changed)
        fastBlkBox.addWidget(self.KBEdit)
        fastBlkBox.addWidget(QLabel("mM"))
        fastBlkBox.addStretch()
        plotSetLayout.addLayout(fastBlkBox)

        leftVBox = QVBoxLayout()
        leftVBox.addLayout(plotSetLayout)
        leftVBox.addWidget(self.canvas)
        leftVBox.addWidget(self.mplTools)
        rightVBox = QVBoxLayout()
        rightVBox.addWidget(self.textBox)
        HBox = QHBoxLayout()
        HBox.addLayout(leftVBox)
        HBox.addLayout(rightVBox)
        self.mainFrame.setLayout(HBox)
        self.setCentralWidget(self.mainFrame)
        
    def on_settings_changed(self):
        """
        Get setting values.
        """
        self.mec.KB = float(self.KBEdit.text()) / 1000.0
        self.tres = float(self.tresEdit.text()) / 1000000.0
        self.tmin = float(self.trangeEdit1.text()) / 1000.0
        self.tmax = float(self.trangeEdit2.text()) / 1000.0
        self.conc = float(self.concEdit.text()) / 1000000.0
        self.cmin = float(self.crangeEdit1.text()) / 1000000.0
        self.cmax = float(self.crangeEdit2.text()) / 1000000.0
        self.mec.fastBlk = self.fastBlkCheckBox.isChecked()

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

    def addActions(self, target, actions):
        """
        Add actions to menu.
        """
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def startInfo(self):
        """
        Get date, time, machine info, etc.
        """
        str1 = "DC_PyPs: Q matrix calculations."
        str2 = ("Date and time of analysis: %4d/%02d/%02d %02d:%02d:%02d"
            %time.localtime()[0:6])
        machine = socket.gethostname()
        system = sys.platform
        str3 = "Machine: %s; System: %s" %(machine, system)
        return str1, str2, str3

    def onPlotPopen(self):
        """
        Display Popen curve.
        """
        self.textBox.append('\nCalculating Popen curve parameters:')
        self.textBox.append('Temporal resolution = {0:.3f} mikrosec'.
            format(self.tres * 1000000))
        text1, text2, c, pe, pi = scl.get_Popen_plot(self.mec, self.tres,
            self.cmin, self.cmax)
        self.textBox.append(text1)
        self.textBox.append(text2)
        self.axes.clear()
        self.axes.semilogx(c, pe, 'b-', c , pi, 'r--')
        self.axes.set_ylim(0, 1)
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotOpenTimePDF(self):
        """
        Display open time probability density function.
        """
        self.textBox.append('\nCalculating open time pdf:')
        self.textBox.append('Agonist concentration = %e M' %self.conc)
        t, br = scl.get_opentime_pdf(self.mec, self.conc,
            self.tmin, self.tmax)
        #self.textBox.append(text1)
        self.axes.clear()
        self.axes.semilogx(t, br, 'b-')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotShutTimePDF(self):
        """
        Display shut time probability density function.
        """
        self.textBox.append('\nCalculating shut time pdf:')
        self.textBox.append('Agonist concentration = %e M' %self.conc)
        t, br = scl.get_shuttime_pdf(self.mec, self.conc,
            self.tmin, self.tmax)
        #self.textBox.append(text1)
        self.axes.clear()
        self.axes.semilogx(t, br, 'b-')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotBrstLenPDF(self):
        """
        Display the burst length distribution.
        """
        self.textBox.append('\nCalculating burst parameters:')
        self.textBox.append('Agonist concentration = %e M' %self.conc)
        text1, t, br = scl.get_burstlen_pdf(self.mec, self.conc,
            self.tmin, self.tmax)
        self.textBox.append(text1)
        self.axes.clear()
        self.axes.semilogx(t, br, 'b-')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotBrstOpDistr(self):
        """
        Display the distribution of number of openings per burst.
        """
        self.textBox.append('\nCalculating burst parameters:')
        self.textBox.append('Agonist concentration = %e M' %self.conc)
        text1, r, Pr = scl.get_burstopenings_distr(self.mec, self.conc)
        self.textBox.append(text1)
        self.axes.clear()
        self.axes.plot(r, Pr,'ro')
        self.axes.set_xlim(0, 11)
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotBrstLenConc(self):
        """
        Display mean burst length versus concentration plot.
        """
        self.axes.clear()

        if self.mec.fastBlk:
            c, br, brblk = scl.get_burstlen_conc_fblk_plot(self.mec, self.cmin,
                self.cmax)
            self.axes.plot(c, br,'b-', c, brblk, 'g-')
        else:
            c, br = scl.get_burstlen_conc_plot(self.mec, self.cmin, self.cmax)
            self.axes.plot(c, br,'b-')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()


    def onHelpAbout(self):
        """
        Display About dialog.
        Called from menu Help|About.
        """
        dialog = AboutDlg(self)
        if dialog.exec_():
            pass

    def onLoadDemo(self):
        """
        Load demo mechanism (C&H82 numerical example).
        Called from menu Load|Demo.
        """
        self.mec = demo.demoQ()
        self.textBox.append("\nLoaded Demo.")
        
    def onLoadFromFile(self):
        """
        Load a mechanism and rates from DC's mec file.
        Called from menu Load|From Mec File...
        """
        filename = QFileDialog.getOpenFileName(self,
            "Open Mec File...", "", "DC Mec Files (*.mec)")
        self.textBox.append("\nFile to read: " + os.path.split(str(filename))[1])

        version, meclist, max_mecnum = readmec.get_mec_list(filename)
        self.textBox.append("Mec file version: %d; contains %d mechanisms."
            %(version, max_mecnum))

        dialog = MecListDlg(meclist, max_mecnum, self)
        if dialog.exec_():
            nrate = dialog.returnRates()

        self.mec = readmec.load_mec(filename, meclist[nrate][0])

        self.textBox.append("Loaded mec: " + meclist[nrate][2])
        self.textBox.append("Loaded rates: " + meclist[nrate][3] + "\n")

class MecListDlg(QDialog):
    """
    Dialog to choose mechansim and rates saved in a DC's mec file.
    """
    def __init__(self, meclist, max_mecnum, parent=None):
        super(MecListDlg, self).__init__(parent)

        self.nmec = 1
        self.nrate = 0
        self.meclist = meclist

        self.mList = QListWidget()
        for i in range(1, (max_mecnum + 1)):
                present = False
                id = 0
                for j in range(len(self.meclist)):
                    if i == self.meclist[j][1]:
                        present = True
                        id = j
                if present:
                    self.mList.addItem(str(i) + " "+ self.meclist[id][2])
        self.connect(self.mList,
            SIGNAL("itemSelectionChanged()"),
            self.mecSelected)

        self.rList = QListWidget()
        self.connect(self.rList,
            SIGNAL("itemSelectionChanged()"),
            self.rateSelected)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))

        layout1 = QHBoxLayout()
        layout1.addWidget(self.mList)
        layout1.addWidget(self.rList)
        layout2 = QVBoxLayout()
        layout2.addLayout(layout1)
        layout2.addWidget(buttonBox)

        self.setLayout(layout2)
        self.resize(1000, 500)
        self.setWindowTitle("Choose mechanisms and rates...")

    def mecSelected(self):
        """
        Populate rate list when a mechanism selected.
        """
        self.nmec = self.mList.currentRow() + 1
        self.rList.clear()
        for i in range(len(self.meclist)):
           if self.meclist[i][1] == self.nmec:
               self.rList.addItem(str(i+1) + " "+ self.meclist[i][3])
               self.nrate = i + 1

    def rateSelected(self):
        """
        Get selected rates.
        """
        str1 = self.rList.currentItem().text()
        self.nrate = int(str1.split(" ")[0]) - 1

    def returnRates(self):
        """
        Return rates on exit.
        """
        return self.nrate

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

if __name__ == "__main__":

    app = QApplication(sys.argv)
    form = QMatGUI()
    form.show()
    app.exec_()
