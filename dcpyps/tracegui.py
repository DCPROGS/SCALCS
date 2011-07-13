#! /usr/bin/python
"""
A simple GUI to display and process single channel record.
Depends on pyqt and matplotlib modules.
"""

try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
except:
    raise ImportError("pyqt module is missing")

import io

class TraceGUI(QMainWindow):
    def __init__(self, parent=None):
        super(TraceGUI, self).__init__(parent)
        self.resize(1000, 700)     # width, height in px
        self.mainFrame = QWidget()
        self.setWindowTitle('pyPlotsamp')
        self.setBackgroundRole(QPalette.Base)
        self.setAutoFillBackground(True)

        self.painter =  QPainter()

        self.loaded = False

        self.line_length = 5 # seconds
        self.page_lines = 5
        self.point_every = 50
        self.line_separ = 10 # pA
        self.pages = 1
        self.page = 1


        fileMenu = self.menuBar().addMenu('&File')
        fileSSDOpenAction = self.createAction("&Open SSD file", self.onSSDFileOpen,
            None, "ssdfileopen", "File Open")
        fileABFOpenAction = self.createAction("&Open ABF file", self.onABFFileOpen,
            None, "abffileopen", "File Open")
        fileSaveAsAction = self.createAction("&Save As...", self.onFileSaveAs,
            None, "filesaveas", "File Save As")
        self.addActions(fileMenu, (fileSSDOpenAction, fileABFOpenAction,
            fileSaveAsAction))

        plotMenu = self.menuBar().addMenu('&Plot')
        nextPageAction = self.createAction("&Next page", self.onNextPage)
        prevPageAction = self.createAction("&Previous page", self.onPrevPage)
        printPageAction = self.createAction("&Print page", self.onPrint)
        plotOptionsAction = self.createAction("&Plot options", self.onPlotOptions)
        self.addActions(plotMenu, (nextPageAction,
            prevPageAction, printPageAction, plotOptionsAction))

        helpMenu = self.menuBar().addMenu('&Help')
        helpAboutAction = self.createAction("&About", self.onHelpAbout)
        self.addActions(helpMenu, (helpAboutAction, None))

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

    def onSSDFileOpen(self):
        """
        """

        self.filename = QFileDialog.getOpenFileName(self,
            "Open Data File...", "", "Consam files (*.ssd *.SSD *.dat *.DAT)")
        self.h = io.ssd_read_header (self.filename)
        self.trace = io.ssd_read_data(self.filename, self.h)

        self.calfac = self.h['calfac']
        self.sample = 1 / self.h['srate']
        self.points_total = self.h['ilen'] / 2
        self.ffilter = self.h['filt']

        self.file_type = 'ssd'
        self.loaded = True
        self.page = 1
        self.update()

    def onABFFileOpen(self):
        """
        """

        self.filename = QFileDialog.getOpenFileName(self,
            "Open Data File...", "", "Axon files (*.abf)")
        self.h = io.abf_read_header(self.filename)
        self.trace = io.abf_read_data(self.filename, self.h)

        self.points_total = self.h['IActualAcqLength'] / self.h['nADCNumChannels']
        self.sample = self.h['fADCSampleInterval'] * self.h['nADCNumChannels'] / 1e6
        self.calfac = (1 /
            (6553.6 * self.h['fInstrumentScaleFactor'][self.h['nADCSamplingSeq'][0]]))
        self.ffilter = float(self.h['fSignalLowpassFilter'][self.h['nADCSamplingSeq'][0]])

        self.file_type = 'abf'
        self.loaded = True
        self.page = 1
        self.update()

    def onFileSaveAs(self):
        """
        """

        self.out_filename = QFileDialog.getSaveFileName(self,
            "Save File As...", "",
            "Consam file (*.ssd)")

        if self.file_type == 'ssd':
            io.ssd_save(self.out_filename, self.h, self.trace)
        elif self.file_type == 'abf':
            h_conv = io.abf2ssd(self.h)
            io.ssd_save(self.out_filename, h_conv, self.trace)

    def onPlotOptions(self):
        """
        """

        dialog = PlotPageDlg(self)
        if dialog.exec_():
            self.line_length, self.page_lines, self.point_every, self.line_separ = dialog.return_par()
        self.update()
        
    def onNextPage(self):
        """
        """
       
        if self.page < self.pages:
            self.page += 1
            self.update()

    def onPrevPage(self):
        """
        """
        
        if self.page > 1:
            self.page -= 1
            self.update()

    def onPrint(self):
        """
        """

        printer=QPrinter(QPrinter.HighResolution)
        printer.setOrientation(QPrinter.Landscape)
        printDialog=QPrintDialog(printer)
        if (printDialog.exec_() == QDialog.Accepted):
            self.painter.begin(printer)
            self.drawSCTrace(self.painter)
            self.painter.end()

    def onHelpAbout(self):
        """
        """

        pass

    def paintEvent(self, event):
        """
        """

        if self.loaded:
            self.painter.begin(self)
            self.drawSCTrace(self.painter)
            self.painter.end()
        
    def drawSCTrace(self, event):
        """
        """

        line_points = int(self.line_length / self.sample + 1)
        page_points = line_points * self.page_lines
        line_points_draw = int(line_points / self.point_every)
        self.pages = self.points_total / page_points

        page_str = (self.filename + "; Page " + str(self.page) + " of " +
            str(self.pages))
        point_str = ("Points " + str(page_points * (self.page - 1) + 1) +
            " to " + str(page_points * self.page) + " every " +
            str(self.point_every) + " point(s); seconds/line: " +
            str(self.line_length) + "; line separation (pA): " + str(self.line_separ))
        self.painter.drawText(100, 50, page_str)
        self.painter.drawText(100, 650, point_str)

        xMinPix = int(self.width() * 5 / 100)
        xMaxPix = int(self.width() * 90 / 100)
        yMaxPix = int(self.height() * 10 / 100)
        yMinPix = int(self.height() * 90 / 100)

        xMinDbl = float(0)
        xMaxDbl = float(self.line_length)
        yMinDbl = float(0)
        yMaxDbl = float(self.page_lines + 1) * self.line_separ
        yStartDbl = float(self.page_lines * self.line_separ)

        xScaleDbl = float(xMaxPix - xMinPix) / float(xMaxDbl - xMinDbl)
        yScaleDbl = float(yMaxPix - yMinPix) / float(yMaxDbl - yMinDbl)

        xPix1 = xMinPix + int((xMinDbl) * xScaleDbl)
        yPix1 = yMinPix + int((yMinDbl) * yScaleDbl)
        xPix2 = xMinPix + int((xMaxDbl) * xScaleDbl)
        yPix2 = yMinPix + int((yMaxDbl) * yScaleDbl)

        for j in range(self.page_lines):
            xDbl1 = 0
            yDbl1 = self.trace[0 + page_points*(self.page-1) + line_points * j] * self.calfac + yStartDbl - (j+1)*self.line_separ
            for i in range (line_points_draw):

                xDbl2 = float((i+1) * self.sample * self.point_every)
                yDbl2 = float(self.trace[0 + page_points*(self.page-1) + line_points * j + (i+1)*self.point_every] * self.calfac + yStartDbl - (j+1)*self.line_separ)
                xPix1 = xMinPix + int((xDbl1 - xMinDbl) * xScaleDbl)
                yPix1 = yMinPix + int((yDbl1 - yMinDbl) * yScaleDbl)
                xPix2 = xMinPix + int((xDbl2 - xMinDbl) * xScaleDbl)
                yPix2 = yMinPix + int((yDbl2 - yMinDbl) * yScaleDbl)
                self.painter.drawLine(xPix1, yPix1, xPix2, yPix2)
                xDbl1 = xDbl2
                yDbl1 = yDbl2


class PlotPageDlg(QDialog):
    """
    Dialog to input page plotting parameters.
    """
    def __init__(self, parent=None):
        super(PlotPageDlg, self).__init__(parent)

        self.line_length = 5 # seconds
        self.page_lines = 5
        self.point_every = 50
        self.line_separ = 10 # pA

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel('Plot layout options'))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Seconds per line:"))
        self.lengthEdit = QLineEdit(unicode(self.line_length))
        self.lengthEdit.setMaxLength(10)
        self.connect(self.lengthEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.lengthEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Number of lines per page:"))
        self.linesEdit = QLineEdit(unicode(self.page_lines))
        self.linesEdit.setMaxLength(10)
        self.connect(self.linesEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.linesEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Draw every nth point:"))
        self.everyEdit = QLineEdit(unicode(self.point_every))
        self.everyEdit.setMaxLength(10)
        self.connect(self.everyEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.everyEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("pA between lines:"))
        self.separEdit = QLineEdit(unicode(self.line_separ))
        self.separEdit.setMaxLength(10)
        self.connect(self.separEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.separEdit)
        layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Plot layout options...")

    def on_par_changed(self):
        """
        """

        self.line_length = int(self.lengthEdit.text())
        self.page_lines = int(self.linesEdit.text())
        self.point_every = int(self.everyEdit.text())
        self.line_separ = int(self.separEdit.text())

    def return_par(self):
        """
        Return parameters on exit.
        """
        return self.line_length, self.page_lines, self.point_every, self.line_separ
    
class FilterOptsDlg(QDialog):
    """
    Dialog to input filter options.
    """
    def __init__(self, parent=None):
        super(FilterOptsDlg, self).__init__(parent)

        self.filter = 1000 # Hz
        self.factor = 1

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel('Filter options:'))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Filter with Gaussian filter to have final fc (Hz):"))
        self.filterEdit = QLineEdit(unicode(self.filter))
        self.filterEdit.setMaxLength(10)
        self.connect(self.filterEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.filterEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Multiply signal by factor (negative to invert):"))
        self.factorEdit = QLineEdit(unicode(self.factor))
        self.factorEdit.setMaxLength(10)
        self.connect(self.factorEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.factorEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Trace will be resampled to keep ratio filter/sampling ~ 10."))
#        self.everyEdit = QLineEdit(unicode(50))
#        self.everyEdit.setMaxLength(10)
#        self.connect(self.everyEdit, SIGNAL("editingFinished()"),
#            self.on_par_changed)
#        layout.addWidget(self.everyEdit)
        layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Filter options...")

    def on_par_changed(self):
        """
        """

        self.filter = int(self.filterEdit.text())
        self.factor = int(self.factorEdit.text())

    def return_par(self):
        """
        Return parameters on exit.
        """
        return self.filter, self.factor