#! /usr/bin/python
"""
A simple GUI for DC_PyPs project.
Depends on pyqt and matplotlib modules.
"""
import time
import sys
import os
import socket
import math

#import numpy as np
try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
except:
    raise ImportError("pyqt module is missing")

try:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
    from matplotlib.figure import Figure
    from matplotlib import scale as mscale
#    from matplotlib import transforms as mtransforms
#    from matplotlib import ticker
except:
    raise ImportError("matplotlib module is missing")

import numpy as np

import scalcslib as scl
#import rcj
import cjumps
import scburst
import popen
import dcio
import samples
import scplotlib as scpl
import mechanism

import optimize
import dataset

class QMatGUI(QMainWindow):
    def __init__(self, parent=None):
        super(QMatGUI, self).__init__(parent)
        self.resize(1000, 700)     # wide, high in px
        self.mainFrame = QWidget()

        self.path = ""
        self.mec = samples.CH82()
        self.mec.KBlk = 0.01
        self.mec.fastblk = False
        self.conc = 100e-9    # 100 nM
        self.tres = 0.0001
        self.rec1 = None
        self.my_colour = ["r", "g", "b", "m", "c", "y"]
        self.present_plot = None

        self.cjprofile = 'rcj'
        self.cjlen = 0.05
        self.cjstep = 5e-6
        self.cjwidth = 0.00001
        self.cjfunc = None
        self.cjargs = None

        loadMenu = self.menuBar().addMenu('&Mechanims')
        loadDemo1Action = self.createAction("&Load demo: CH82", self.onLoadDemo_CH82,
            None, "loaddemo", "Load Demo mec")
        loadDemo2Action = self.createAction("&Load demo: dC-K", self.onLoadDemo_dCK,
            None, "loaddemo", "Load Demo mec")
        loadFromMecFileAction = self.createAction("&Load from DCprogs MEC File...",
            self.onLoadMecFile,
            None, "loadfrommecfile", "Load from Mec file")
        loadFromModFileAction = self.createAction("&Load from ChannelLab MOD File...",
            self.onLoadModFile,
            None, "loadfrommodfile", "Load from ChannelLab Mod file")
        modifyMecAction = self.createAction("&Modify loaded mec", self.onModifyMec,
            None, "modifymec", "Modify mec")
        quitAction = self.createAction("&Quit", self.close,
            "Ctrl+Q", "appquit", "Close the application")
        self.addActions(loadMenu, (loadDemo1Action, loadDemo2Action,
            loadFromMecFileAction, loadFromModFileAction,
            modifyMecAction, quitAction))

        plotMenu = self.menuBar().addMenu('&Plot')
        plotPopenAction = self.createAction("&Popen curve", self.onPlotPopen)
        plotOpenTimePDFAction = self.createAction(
            "&Open time pdf", self.onPlotOpenTimePDF)
        plotShutTimePDFAction = self.createAction(
            "&Shut time pdf", self.onPlotShutTimePDF)
        plotSubsetTimePDFAction = self.createAction(
            "&Subset time pdf", self.onPlotSubsetTimePDF)
        plotBurstLenPDFAction = self.createAction(
            "&Burst length pdf", self.onPlotBrstLenPDF)
        plotBurstLenPDFActionCond = self.createAction(
            "&Conditional burst length pdf", self.onPlotBrstLenPDFCond)
        plotBurstOpeningDistrAction = self.createAction(
            "&Burst openings distribution", self.onPlotBrstOpDistr)
        plotBurstOpeningDistrActionCond = self.createAction(
            "&Conditional burst openings distribution", self.onPlotBrstOpDistrCond)
        plotBurstLenVConcAction = self.createAction(
            "&Burst length vs concentration", self.onPlotBrstLenConc)
        plotJumpPopenAction = self.createAction(
            "&Concentration jump: Popen", self.onPlotCJumpPopen)
        plotJumpOccupanciesAction = self.createAction(
            "&Concentration jump: occupancies",
            self.onPlotCJumpOccupancies)
        plotJumpOnOffTauConc = self.createAction(
            "&Concentration jump: weighted on/off tau versus concentration",
            self.onPlotCJumpRiseVConc)
        plotCorrOpenShutAction = self.createAction(
            "&Correlations", self.onPlotOpShCorr)
        plotAdjacentOpenShutAction = self.createAction(
            "&Open time adjacent to shut time range pdf", self.onPlotOpAdjShacent)
#        plotJump2PopenAction = self.createAction(
#            "&Instant rise and exponential decay concentration jump: Popen", self.onPlotCJump2Popen)
        plotSaveASCII = self.createAction(
            "&Save current plot as ASCII file", self.onPlotSaveASCII)
        self.addActions(plotMenu, (plotOpenTimePDFAction, plotShutTimePDFAction,
            plotAdjacentOpenShutAction, plotCorrOpenShutAction,
            # setDisabled(False) to activate plotting the subset time distributions
            plotSubsetTimePDFAction.setDisabled(True),
            plotBurstLenPDFAction, plotBurstLenPDFActionCond,
            plotBurstOpeningDistrAction, plotBurstOpeningDistrActionCond,
            plotBurstLenVConcAction,
            plotJumpPopenAction, plotJumpOccupanciesAction, plotJumpOnOffTauConc,
#            plotJump2PopenAction,            
            plotPopenAction,
            plotSaveASCII))
        plotMenu.insertSeparator(plotJumpPopenAction)
        plotMenu.insertSeparator(plotPopenAction)
        plotMenu.insertSeparator(plotSaveASCII)

# UNCOMMENT NEXT LINES TO ENABLE DATA DISTRIBUTION PLOTTING
        dataMenu = self.menuBar().addMenu('&Data')
        openScanAction = self.createAction("&Load SC record", self.onLoadData)
        imposeResolutionAction = self.createAction("&Impose resolution",
            self.onImposeResolution)
        plotDataOpenAction = self.createAction("&Plot open period distribution",
            self.onPlotDataOpen)
        plotDataShutAction = self.createAction("&Plot shut period distribution",
            self.onPlotDataShut)
        plotDataBurstAction = self.createAction("&Plot burst length distribution",
            self.onPlotDataBurst)
        likelihoodAction = self.createAction("&HJCfit",
            self.onCalculateLikelihood)
        self.addActions(dataMenu, (openScanAction, imposeResolutionAction,
            plotDataOpenAction, plotDataShutAction, plotDataBurstAction,
            likelihoodAction))
# LINES RELATED TO DATA HISTOGRAMMS PLOTTING

        printOutMenu = self.menuBar().addMenu('&Printout')
        printOutSaveAction = self.createAction("&Save", self.onPrintOutSave)
        self.addActions(printOutMenu, (printOutSaveAction,
            None))

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
        mscale.register_scale(scpl.SquareRootScale)

        self.textBox = QTextBrowser()
        # Set here if printout to TextBox only or also to file or console.
        self.log = PrintLog(self.textBox) #, sys.stdout)
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
        tresBox.addWidget(QLabel("microsec"))
        tresBox.addStretch()
        plotSetLayout.addLayout(tresBox)

        concBox = QHBoxLayout()
        concBox.addWidget(QLabel("Concentration = "))
        self.concEdit = QLineEdit(unicode(self.conc * 1000000))
        self.concEdit.setMaxLength(8)
        self.connect(self.concEdit, SIGNAL("editingFinished()"),
            self.on_settings_changed)
        concBox.addWidget(self.concEdit)
        concBox.addWidget(QLabel("microM"))
        concBox.addStretch()
        plotSetLayout.addLayout(concBox)

        fastBlkBox = QHBoxLayout()
        self.fastBlkCheckBox = QCheckBox("&Fast block?")
        self.fastBlkCheckBox.setChecked(self.mec.fastblk)
        self.connect(self.fastBlkCheckBox, SIGNAL("stateChanged(int)"),
            self.on_settings_changed)
        fastBlkBox.addWidget(self.fastBlkCheckBox)
        fastBlkBox.addWidget(QLabel("KB ="))
        self.KBEdit = QLineEdit(unicode(self.mec.KBlk * 1000))
        self.KBEdit.setMaxLength(8)
        self.connect(self.KBEdit, SIGNAL("editingFinished()"),
            self.on_settings_changed)
        fastBlkBox.addWidget(self.KBEdit)
        fastBlkBox.addWidget(QLabel("mM"))
        fastBlkBox.addStretch()
        plotSetLayout.addLayout(fastBlkBox)

        self.txtPltBox = QTextBrowser()
        plotSetLayout.addWidget(self.txtPltBox)

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
        self.mec.KBlk = float(self.KBEdit.text()) / 1000.0
        self.tres = float(self.tresEdit.text()) / 1000000.0
        self.conc = float(self.concEdit.text()) / 1000000.0
        self.mec.fastblk = self.fastBlkCheckBox.isChecked()

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

# UNCOMMENT NEXT FUNCTIONS TO ENABLE DATA DISTRIBUTION PLOTTING
    def onLoadData(self):
        """
        """

        self.textBox.append('\n\n\t===== LOADING DATA FILE =====')
        filename = QFileDialog.getOpenFileName(self,
            "Open SCN File...", "", "DC SCN Files (*.scn)")
        ioffset, nint, calfac, header = dcio.scn_read_header(filename)
        tint, iampl, iprops = dcio.scn_read_data(filename, ioffset, nint, calfac)
        self.rec1 = dataset.TimeSeries(filename, header, tint, iampl, iprops)
        self.textBox.append("\nLoaded record from file: " +
            os.path.split(str(filename))[1])

    def onImposeResolution(self):

        self.rec1.impose_resolution(self.tres)
        falsrate = dataset.false_events(self.tres,
            self.rec1.header['ffilt'] * 1000, self.rec1.header['rms'],
            self.rec1.header['avamp'] * self.rec1.calfac)
        trise = 0.3321 / (self.rec1.header['ffilt'] * 1000)
        zo = self.tres / trise
        aamaxo = dataset.erf(0.88604 * zo)
        self.textBox.append('\nAt resolution {0:.0f} microsec false event rate '.
            format(self.tres * 1000000) +
            '(per sec) for openings and shuttings is {0:.3e}'.format(falsrate)+
            ' ( {0:.2f} risetimes, A/Amax = {1:.2f})'.format(zo, aamaxo))
        self.textBox.append('After imposing the resolution of original {0:d}'.
            format(len(self.rec1.itint)) + ' intervals were left {0:d}'.
            format(len(self.rec1.rampl)))
        self.rec1.get_open_shut_periods()

    def onCalculateLikelihood(self):
        """
        """

        opts = {}
        opts['mec'] = self.mec
        opts['conc'] = self.conc
        opts['tres'] = self.tres
        opts['tcrit'] = self.tcrit
        opts['isCHS'] = True
        opts['data'] = self.rec1.bursts
        theta = self.mec.theta()
        start_lik, th = scl.HJClik(np.log(theta), opts)
        self.textBox.append('\nLog Likelihood = {0:.3f}'.
            format(-start_lik))
        self.mec.printout(self.log)
        self.textBox.append("\nFitting started: {0}\n".format(time.asctime()))
        xout, fout, niter, neval = optimize.simplex(scl.HJClik,
            np.log(theta), args=opts, display=True) #, outdev=self.log)
        self.textBox.append("\nFitting finished: {0}\n".format(time.asctime()))
        # Display results.
        self.mec.theta_unsqueeze(np.exp(xout))
        self.textBox.append("\n Final rate constants:")
        self.mec.printout(self.log)
        self.textBox.append('\n Final log-likelihood = {0:.6f}'.format(-fout))
        self.textBox.append('\n Number of evaluations = {0:d}'.format(neval))
        self.textBox.append('\n Number of iterations = {0:d}\n\n'.format(niter))

    def onPlotDataBurst(self):
        """
        """
        self.textBox.append('\n\n\t===== PLOTTING DATA: BURST LENGTH =====')
        self.textBox.append("\nFirst burst starts only after gap > tcrit is found.")
        self.textBox.append("Unusable shut time treated as a valid end of burst.")

        dialog = BurstPlotDlg(self)
        if dialog.exec_():
            self.tcrit = dialog.return_par()

        self.textBox.append('\nCritical gap length = {0:.3f} millisec'.
            format(self.tcrit * 1000))
        self.rec1.get_bursts(self.tcrit)
        self.textBox.append('\nNumber of bursts = {0:d}'.
            format(len(self.rec1.bursts)))
        blength = self.rec1.get_burst_length_list()
        self.textBox.append('Average = {0:.3f} millisec'.
            format(np.average(blength)))
        self.textBox.append('Range: {0:.3f}'.format(min(blength)) +
            ' to {0:.3f} millisec'.format(max(blength)))

        x, y, dx = dataset.prepare_hist(blength, self.tres)

        self.axes.clear()
        self.axes.semilogx(x, y, 'b-')
        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotDataOpen(self):
        """
        """
        self.txtPltBox.clear()
        self.txtPltBox.append('\t===== HISTOGRAM AND DISTRIBUTION OF OPEN PERIODS =====')
        self.txtPltBox.append('Agonist concentration = {0:.5g} mikroM'.
            format(self.conc * 1000000))
        self.txtPltBox.append('Resolution = {0:.5g} mikrosec'.
            format(self.tres * 1000000))
        self.txtPltBox.append('Ideal pdf- red dashed line.')
        self.txtPltBox.append('Exact pdf- blue solid line.')
        self.txtPltBox.append('Data histogram- black line.')

        self.textBox.append('\n\n\t===== PLOTTING DATA: OPEN PERIODS =====')
        self.textBox.append('\nNumber of open periods = {0:d}'.
            format(len(self.rec1.opint)))
        self.textBox.append('Average = {0:.3f} millisec'.
            format(np.average(self.rec1.opint)))
        self.textBox.append('Range: {0:.3f}'.format(min(self.rec1.opint)) +
            ' to {0:.3f} millisec'.format(max(self.rec1.opint)))
        x, y, dx = dataset.prepare_hist(self.rec1.opint, self.tres)

        self.mec.set_eff('c', self.conc)
#        scl.printout_occupancies(self.mec, self.tres, output=self.log)
#        scl.printout_distributions(self.mec, self.tres, output=self.log)
        t, ipdf, epdf, apdf = scpl.open_time_pdf(self.mec, self.tres)
        sipdf = scpl.scaled_pdf(t, ipdf, math.log10(dx), len(self.rec1.opint))
        sepdf = scpl.scaled_pdf(t, epdf, math.log10(dx), len(self.rec1.opint))

#        self.present_plot = np.vstack((t, ipdf, epdf, apdf))

        self.axes.clear()
        self.axes.semilogx(x, y, 'k-',
            t, sipdf, 'r--', t, sepdf, 'b-')
        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotDataShut(self):
        """
        """

        self.txtPltBox.clear()
        self.txtPltBox.append('\t===== HISTOGRAM AND DISTRIBUTION OF SHUT TIMES =====')
        self.txtPltBox.append('Agonist concentration = {0:.5g} mikroM'.
            format(self.conc * 1000000))
        self.txtPltBox.append('Resolution = {0:.5g} mikrosec'.
            format(self.tres * 1000000))
        self.txtPltBox.append('Ideal pdf- red dashed line.')
        self.txtPltBox.append('Exact pdf- blue solid line.')
        self.txtPltBox.append('Data histogram- black line.')


        self.textBox.append('\n\n\t===== PLOTTING DATA: OPEN PERIODS =====')
        self.textBox.append('\nNumber of shut periods = {0:d}'.
            format(len(self.rec1.shint)))
        self.textBox.append('Average = {0:.3f} millisec'.
            format(np.average(self.rec1.shint)))
        self.textBox.append('Range: {0:.3f}'.format(min(self.rec1.shint)) +
            ' to {0:.3f} millisec'.format(max(self.rec1.shint)))
        x, y, dx = dataset.prepare_hist(self.rec1.shint, self.tres)

        self.mec.set_eff('c', self.conc)
#        scl.printout_occupancies(self.mec, self.tres, output=self.log)
#        scl.printout_distributions(self.mec, self.tres, output=self.log)
        t, ipdf, epdf, apdf = scpl.shut_time_pdf(self.mec, self.tres)
        sipdf = scpl.scaled_pdf(t, ipdf, math.log10(dx), len(self.rec1.opint))
        sepdf = scpl.scaled_pdf(t, epdf, math.log10(dx), len(self.rec1.opint))
#        self.present_plot = np.vstack((t, ipdf, epdf, apdf))

        self.axes.clear()
        self.axes.semilogx(x, y, 'k-',
            t, sipdf, 'r--', t, sepdf, 'b-')
        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()
# LINES RELATED TO DATA HISTOGRAMMS PLOTTING

    def onPlotCJumpPopen(self):
        """
        Display concentration jump.
        """

        dialog1 = ConcProfileDlg(self)
        if dialog1.exec_():
            self.cjprofile = dialog1.return_par()
        dialog = CJumpParDlg(self, self.cjprofile, self.cjlen, self.cjstep, self.cjargs)
        if dialog.exec_():
            self.cjlen, self.cjstep, self.cjfunc, self.cjargs = dialog.return_par()

        self.txtPltBox.clear()
        self.txtPltBox.append('===== CONCENTRATION JUMP =====')
        self.txtPltBox.append('Concentration profile- green solid line.')
        self.txtPltBox.append('Relaxation- blue solid line.')
        self.txtPltBox.append('\nConcentration pulse profile:')
        self.txtPltBox.append('Peak concentration = {0:.5g} mM'
            .format(self.cjargs[0] * 1000))
        self.txtPltBox.append('Background concentration = {0:.5g} mM'
            .format(self.cjargs[1] * 1000))
        if self.cjprofile == 'rcj':
            self.txtPltBox.append('10- 90% rise time = {0:.5g} microsec'
                .format(self.cjargs[4] * 1e+6))
            self.txtPltBox.append('90- 10% decay time = {0:.5g} microsec'
                .format(self.cjargs[5] * 1e+6))
            self.txtPltBox.append('Pulse width = {0:.5g} millisec'
                .format(self.cjargs[3] * 1000))
        elif self.cjprofile == 'instexp':
            self.txtPltBox.append('Decay time constant = {0:.5g} millisec'
                .format(self.cjargs[3] * 1000))
        elif self.cjprofile == 'square':
            self.txtPltBox.append('Pulse width = {0:.5g} millisec'
                .format(self.cjargs[3] * 1000))
        elif self.cjprofile == 'square2':
            self.txtPltBox.append('Pulse width = {0:.5g} millisec'
                .format(self.cjargs[3] * 1000))
            self.txtPltBox.append('Interpulse interval = {0:.5g} millisec'
                .format(self.cjargs[4] * 1000))
        self.txtPltBox.append("---\n")

        t, c, Popen, P  = cjumps.solve_jump(self.mec, self.cjlen, self.cjstep,
            self.cjfunc, self.cjargs)
        maxP = max(Popen)
        maxC = max(c)
        c1 = (c / maxC) * 0.2 * maxP + 1.02 * maxP

        self.axes.clear()
        self.axes.plot(t * 1000, Popen,'b-', t * 1000, c1, 'g-')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

        if self.cjprofile == 'instexp':
            self.textBox.append('\n\nCalculated response to an instantan jump to {0:.5g} mM '.
                format(self.cjargs[0] * 1000) +
                'concentration with an exponential decay tau of {0:.5g} ms: '.
                format(self.cjargs[3] * 1000) +
                'maximal Popen- {0:.5g}'.format(maxP))
        elif ((self.cjprofile == 'rcj') or (self.cjprofile == 'square')):
            cjumps.printout(self.mec, self.cjargs[0], self.cjargs[3], output=self.log)
        self.present_plot = np.vstack((t, Popen, c, P))

    def onPlotCJumpOccupancies(self):
        """
        Display realistic concentration jump.
        """

        dialog1 = ConcProfileDlg(self)
        if dialog1.exec_():
            self.cjprofile = dialog1.return_par()
        dialog = CJumpParDlg(self, self.cjprofile, self.cjlen, self.cjstep, self.cjargs)
        if dialog.exec_():
            self.cjlen, self.cjstep, self.cjfunc, self.cjargs = dialog.return_par()

        self.txtPltBox.clear()
        self.txtPltBox.append('===== REALISTIC CONCENTRATION JUMP =====')
        self.txtPltBox.append('Concentration profile- black solid line on top.')
        self.txtPltBox.append('Popen relaxation- black solid line.')
        self.txtPltBox.append('Occupancies of open states- red dashed lines.')
        self.txtPltBox.append('Occupancies of shortlived shut states- green dashed lines.')
        self.txtPltBox.append('Occupancies of longlived shut states- blue dashed lines.')

        self.txtPltBox.append('\nConcentration pulse profile:')
        self.txtPltBox.append('Peak concentration = {0:.5g} mM'
            .format(self.cjargs[0] * 1000))
        self.txtPltBox.append('Background concentration = {0:.5g} mM'
            .format(self.cjargs[1] * 1000))
        if self.cjprofile == 'rcj':
            self.txtPltBox.append('10- 90% rise time = {0:.5g} microsec'
                .format(self.cjargs[4] * 1e+6))
            self.txtPltBox.append('90- 10% decay time = {0:.5g} microsec'
                .format(self.cjargs[5] * 1e+6))
            self.txtPltBox.append('Pulse width = {0:.5g} millisec'
                .format(self.cjargs[3] * 1000))
        elif self.cjprofile == 'instexp':
            self.txtPltBox.append('Decay time constant = {0:.5g} millisec'
                .format(self.cjargs[3] * 1000))
        elif self.cjprofile == 'square':
            self.txtPltBox.append('Pulse width = {0:.5g} millisec'
                .format(self.cjargs[3] * 1000))
        self.txtPltBox.append("---\n")

        t, c, Popen, P = cjumps.solve_jump(self.mec, self.cjlen, self.cjstep,
            self.cjfunc, self.cjargs)
        maxP = max(Popen)
        maxC = max(c)
        c1 = (c / maxC) * 0.2 * maxP + 1.02 * maxP

        self.axes.clear()
        self.axes.plot(t * 1000, c1, 'k-')
        self.axes.plot(t * 1000, Popen, 'k-')
        for i in range (0, self.mec.kA):
            self.axes.plot(t * 1000, P[i], 'r--')
        for i in range (self.mec.kA, self.mec.kA + self.mec.kB):
            self.axes.plot(t * 1000, P[i], 'g--')
        for i in range (self.mec.kA + self.mec.kB, self.mec.k):
            self.axes.plot(t * 1000, P[i], 'b--')
#        self.axes.set_ylim(0.01, maxC * 1.01)
        #self.axes.set_xlim(5, 30)

        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

        self.present_plot = np.vstack((t, Popen, c, P))
#        rcj.printout(self.mec, jpar, output=self.log)

    def onPlotCJumpRiseVConc(self):
        """
        Display plot of weighted on-tau versus concentration. Square pulse only.
        """
        
        self.cjprofile == 'square'

        dialog = CJumpParDlg2(self)
        if dialog.exec_():
            cmin, cmax, self.width = dialog.return_par()

        self.txtPltBox.clear()
        self.txtPltBox.append('===== WEIGHTED ON/OFF TAU VERSUS CONCENTRATION =====')
        self.txtPltBox.append('Pulse width = {0:.5g} millisec'
            .format(self.width * 1000))
        self.txtPltBox.append('Tau ON - blue solid line.')
        self.txtPltBox.append('Tau OFF - green solid line.')
        self.txtPltBox.append('X axis in mM; Y axis in ms.')
        self.txtPltBox.append("---\n")

        c, ton, toff  = scpl.conc_jump_on_off_taus_versus_conc_plot(self.mec,
            cmin, cmax, self.width)

        self.axes.clear()
        self.axes.semilogx(c, ton,'b-', c, toff, 'g-')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotPopen(self):
        """
        Display Popen curve.
        """

        self.txtPltBox.clear()
        self.txtPltBox.append('\t===== Popen PLOT =====')
        self.txtPltBox.append('Resolution = {0:.5g} mikrosec'.
            format(self.tres * 1000000))
        self.txtPltBox.append('Ideal curve- red dashed line.')
        self.txtPltBox.append('HJC curve- blue solid line.')

        popen.printout(self.mec, self.tres, output=self.log)
        c, pe, pi = scpl.Popen(self.mec, self.tres)
        self.present_plot = np.vstack((c, pe, pi))

        self.axes.clear()
        self.axes.semilogx(c, pe, 'b-', c , pi, 'r--')
        self.axes.set_ylim(0, 1)
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotOpShCorr(self):
        """
        Display open, shut and open-shut time correlations.
        """

        self.txtPltBox.clear()
        self.txtPltBox.append('\t===== OPEN, SHUT, OPEN/SHUT CORRELATION PLOTS =====')
        self.txtPltBox.append('Agonist concentration = {0:.5g} mikroM'.
            format(self.conc * 1000000))
        self.txtPltBox.append('Shut time correlation - red circles.')
        self.txtPltBox.append('Open time correlation - green circles')
        self.txtPltBox.append('Open-shut time correlation - blue circles')

        self.mec.set_eff('c', self.conc)
        scl.printout_correlations(self.mec, output=self.log)
        # TODO: need dialog to enter lag value. 
        lag = 5
        n, roA, roF, roAF = scpl.corr_open_shut(self.mec, lag)
        self.present_plot = np.vstack((n, roA, roF, roAF))

        self.axes.clear()
        self.axes.plot(n, roA,'go', n, roF, 'ro', n, roAF, 'bo')
        self.axes.axhline(y=0, xmin=0, xmax=1, color='k')
        self.axes.set_xlim(0, 6)
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()
        
    def onPlotOpAdjShacent(self):
        """
        Display open time adjacent to shut time range pdf.
        """
        self.txtPltBox.clear()
        self.txtPltBox.append('\t===== OPEN TIME ADJACENT TO SHUT TIME RANGE PDF =====')
        self.txtPltBox.append('Agonist concentration = {0:.5g} mikroM'.
            format(self.conc * 1000000))
        self.txtPltBox.append('Ideal open time pdf- red dashed line.')
        self.txtPltBox.append('Open times adjacent to shut time range pdf- blue solid line.')

        self.mec.set_eff('c', self.conc)
        # TODO: need dialog to enter lag value. 
        dialog = ShutRangeDlg(self)
        if dialog.exec_():
            u1, u2 = dialog.return_par()
        scl.printout_adjacent(self.mec, u1, u2, output=self.log)
        
        t, ipdf, ajpdf = scpl.adjacent_open_time_pdf(self.mec, self.tres, u1, u2)
        self.present_plot = np.vstack((t, ipdf, ajpdf))

        self.axes.clear()
        self.axes.semilogx(t, ipdf, 'r--', t, ajpdf, 'b-')
        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()
        
    def onPlotOpenTimePDF(self):
        """
        Display open time probability density function.
        """
        self.txtPltBox.clear()
        self.txtPltBox.append('\t===== OPEN TIME PDF =====')
        self.txtPltBox.append('Agonist concentration = {0:.5g} mikroM'.
            format(self.conc * 1000000))
        self.txtPltBox.append('Resolution = {0:.5g} mikrosec'.
            format(self.tres * 1000000))
        self.txtPltBox.append('Ideal pdf- red dashed line.')
        self.txtPltBox.append('Exact pdf- blue solid line.')
        self.txtPltBox.append('Asymptotic pdf- green solid line.')

        self.mec.set_eff('c', self.conc)

        scl.printout_occupancies(self.mec, self.tres, output=self.log)
        scl.printout_distributions(self.mec, self.tres, output=self.log)
        
        t, ipdf, epdf, apdf = scpl.open_time_pdf(self.mec, self.tres)
        self.present_plot = np.vstack((t, ipdf, epdf, apdf))

        self.axes.clear()
        self.axes.semilogx(t, ipdf, 'r--', t, epdf, 'b-', t, apdf, 'g-')
        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotSubsetTimePDF(self):
        """
        """

        self.txtPltBox.clear()
        self.txtPltBox.append('\t===== SUBSET TIME PDF =====')
        self.txtPltBox.append('Agonist concentration = {0:.5g} mikroM'.
            format(self.conc * 1000000))
        self.txtPltBox.append('Resolution = {0:.5g} mikrosec'.
            format(self.tres * 1000000))
        self.txtPltBox.append('Ideal pdf- red dashed line.')
        self.txtPltBox.append('Subset life time pdf- blue solid line.')

        self.mec.set_eff('c', self.conc)
        # TODO: need dialog to enter state1 and state2
        state1 = 8
        state2 = 10

        t, ipdf, spdf = scpl.subset_time_pdf(self.mec, self.tres, state1, state2)
        self.present_plot = np.vstack((t, ipdf, s))

        self.axes.clear()
        self.axes.semilogx(t, spdf, 'b-', t, ipdf, 'r--')
        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotShutTimePDF(self):
        """
        Display shut time probability density function.
        """

        self.txtPltBox.clear()
        self.txtPltBox.append('\t===== SHUT TIME PDF =====')
        self.txtPltBox.append('Agonist concentration = {0:.5g} mikroM'.
            format(self.conc * 1000000))
        self.txtPltBox.append('Resolution = {0:.5g} mikrosec'.
            format(self.tres * 1000000))
        self.txtPltBox.append('Ideal pdf- red dashed line.')
        self.txtPltBox.append('Exact pdf- blue solid line.')
        self.txtPltBox.append('Asymptotic pdf- green solid line.')

        self.mec.set_eff('c', self.conc)
        scl.printout_tcrit(self.mec, output=self.log)
        t, ipdf, epdf, apdf = scpl.shut_time_pdf(self.mec, self.tres)
        self.present_plot = np.vstack((t, ipdf, epdf, apdf))

        self.axes.clear()
        self.axes.semilogx(t, ipdf, 'r--', t, epdf, 'b-', t, apdf, 'g-')
        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotBrstLenPDF(self):
        """
        Display the burst length distribution.
        """
        self.txtPltBox.clear()
        self.txtPltBox.append('\t===== BURST LENGTH PDF =====')
        self.txtPltBox.append('Agonist concentration = {0:.5g} microM'.
            format(self.conc * 1000000))
        self.txtPltBox.append('Ideal pdf- blue solid line.')
        self.txtPltBox.append('Individual components- blue dashed lines.')

        self.mec.set_eff('c', self.conc)
        scburst.printout_pdfs(self.mec, output=self.log)
        t, fbrst, mfbrst = scpl.burst_length_pdf(self.mec, multicomp=True)
        self.present_plot = np.vstack((t, fbrst, mfbrst))
        
        self.axes.clear()
        self.axes.semilogx(t, fbrst, 'b-')
        for i in range(self.mec.kE):
            self.axes.semilogx(t, mfbrst[i], 'b--')
        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotBrstLenPDFCond(self):
        """
        Display the conditional burst length distribution.
        """
        self.txtPltBox.clear()
        self.txtPltBox.append('===== BURST LENGTH PDF ' +
            '\nCONDITIONAL ON STARTING STATE =====')
        self.txtPltBox.append('Agonist concentration = {0:.5g} microM'.
            format(self.conc * 1000000))
        self.txtPltBox.append('Ideal pdf- blue solid line.')

        self.mec.set_eff('c', self.conc)

        t, fbst, cfbst = scpl.burst_length_pdf(self.mec, conditional=True)
        self.present_plot = np.vstack((t, fbst, cfbst))
        self.axes.clear()

        # TODO: only 6 colours are available now.        
        for i in range(self.mec.kA):
            self.axes.semilogx(t, cfbst[i], self.my_colour[i]+'-',
                label="State {0:d}".format(i+1))
        self.axes.semilogx(t, fbst, 'k-', label="Not conditional")
        handles, labels = self.axes.get_legend_handles_labels()
        self.axes.legend(handles, labels, frameon=False)

        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotBrstOpDistr(self):
        """
        Display the distribution of number of openings per burst.
        """

        self.txtPltBox.clear()
        self.txtPltBox.append('===== DISTRIBUTION OF NUMBER OF OPENINGS PER BURST =====')

        self.mec.set_eff('c', self.conc)
        # TODO: need dialog to enter n
        n = 10
        r, Pr = scpl.burst_openings_pdf(self.mec, n)
        self.present_plot = np.vstack((r, Pr))

        self.axes.clear()
        self.axes.plot(r, Pr,'ro')
        self.axes.set_xlim(0, 11)
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotBrstOpDistrCond(self):
        """
        Display the conditional distribution of number of openings per burst.
        """

        self.txtPltBox.clear()
        self.txtPltBox.append('===== DISTRIBUTION OF NUMBER OF OPENINGS PER BURST' +
        '\nCONDITIONAL ON STARTING STATE=====')

        self.mec.set_eff('c', self.conc)
        n = 10
        r, Pr, cPr = scpl.burst_openings_pdf(self.mec, n, conditional=True)
        self.present_plot = np.vstack((r, Pr, cPr))

        self.axes.clear()
        # TODO: only 6 colours are available now.
        for i in range(self.mec.kA):
            self.axes.plot(r, cPr[i], self.my_colour[i]+'o',
                label="State {0:d}".format(i+1))
        self.axes.plot(r, Pr,'ko', label="Not conditional")
        handles, labels = self.axes.get_legend_handles_labels()
        self.axes.legend(handles, labels, frameon=False)
        self.axes.set_xlim(0, n+1)
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotBrstLenConc(self):
        """
        Display mean burst length versus concentration plot.
        """
        self.txtPltBox.clear()
        self.txtPltBox.append('===== MEAN BURST LENGTH VERSUS CONCENTRATION =====')
        self.txtPltBox.append('Dashed line: corrected for fast block.')

        # TODO: need dialog to enter concentration range.
        cmin = 10e-9
        cmax = 0.1
        c, br, brblk = scpl.burst_length_versus_conc_plot(self.mec, cmin, cmax)
        self.present_plot = np.vstack((c, br, brblk))

        self.axes.clear()
        self.axes.plot(c, br,'r-', c, brblk, 'r--')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPrintOutSave(self):
        """
        """
        printOutFilename = QFileDialog.getSaveFileName(self,
                "Save as PRT file...", ".prt",
                "PRT files (*.prt)")

        self.textBox.selectAll()
        text = self.textBox.toPlainText()
        fout = open(printOutFilename,'w')
        fout.write(text)
        fout.close()

        self.txtPltBox.clear()
        self.txtPltBox.append('Saved printout file:')
        self.txtPltBox.append(printOutFilename)

    def onPlotSaveASCII(self):

        savePlotTXTFilename = QFileDialog.getSaveFileName(self,
                "Save as TXT file...", ".txt",
                "TXT files (*.txt)")

        fout = open(savePlotTXTFilename,'w')
        for i in range(self.present_plot.shape[1]):
            for j in range(self.present_plot.shape[0]):
                fout.write('{0:.6e}\t'.format(self.present_plot[j, i]))
            fout.write('\n')
        fout.close()

    def onHelpAbout(self):
        """
        Display About dialog.
        Called from menu Help|About.
        """
        dialog = AboutDlg(self)
        if dialog.exec_():
            pass

    def onLoadDemo_CH82(self):
        """
        Load demo mechanism (C&H82 numerical example).
        Called from menu Load|Demo.
        """

        self.mec = samples.CH82()
        self.textBox.append("\nLoaded Colquhoun&Hawkes 82 numerical example.\n")
        self.mec.printout(self.log)

    def onLoadDemo_dCK(self):
        """
        Load del Castillo - Katz mechanism.
        Called from menu Load|Demo.
        """

        self.mec = samples.CCO()
        self.textBox.append("\nLoaded del Castillo-Katz mechanism.\n")
        self.mec.printout(self.log)

    def onLoadMecFile(self):
        """
        Load a mechanism and rates from DC's mec file.
        Called from menu Load|From DCPROGS .MEC File...
        """
        filename = QFileDialog.getOpenFileName(self,
            "Open Mec File...", self.path, "DC Mec Files (*.mec *.MEC)")
        self.path = os.path.split(str(filename))[0]
        self.textBox.append("\nFile to read: " + os.path.split(str(filename))[1])

        version, meclist, max_mecnum = dcio.mec_get_list(filename)
        self.textBox.append("Mec file version: %d; contains %d mechanisms."
            %(version, max_mecnum))

        dialog = MecListDlg(meclist, max_mecnum, self)
        if dialog.exec_():
            nrate = dialog.returnRates()

        self.mec = dcio.mec_load(filename, meclist[nrate][0])

        self.textBox.append("Loaded mec: " + meclist[nrate][2])
        self.textBox.append("Loaded rates: " + meclist[nrate][3] + "\n")
        self.mec.printout(self.log)

    def onLoadModFile(self):
        """
        Load a mechanism and rates from Channel Lab .mod file.
        Called from menu Load|From Channel Lab .MOD File...
        """
        filename = QFileDialog.getOpenFileName(self,
            "Open MOD File...", self.path, "Channel Lab MOD Files (*.mod *.MOD)")
        self.path = os.path.split(str(filename))[0]
        self.textBox.append("\nFile to read: " + os.path.split(str(filename))[1])

        self.mec, title = dcio.mod_load(filename)
        self.textBox.append("\n" + title + "\n")
        self.mec.printout(self.log)

    def onModifyMec(self):
        """
        """
        table = RateTableDlg(self, self.mec, self.log)
        if table.exec_():
            self.mec = table.return_mec()

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
        self.out1.insertPlainText(text)
        if self.out2:
            self.out2.write(text)

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
            value = 0
            if self.mec.Rates[i].fixed > 0:
                value = 2
            check.setCheckState(value)
            self.setItem(i, 5, check)

            check = QTableWidgetItem()
            value = 0
            if self.mec.Rates[i].mr:
                value = 2
            check.setCheckState(value)
            self.setItem(i, 6, check)

            check = QTableWidgetItem()
            value = 0
            if self.mec.Rates[i].is_constrained:
                value = 2
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

class CJumpParDlg(QDialog):
    """
    Dialog to input realistic concentration pulse parameters.
    """
    def __init__(self, parent=None, profile='rcj', cjlen=0.05, cjstep=5e-6, cjargs=None):
        super(CJumpParDlg, self).__init__(parent)

        self.profile = profile

        self.reclength = cjlen * 1000 # Record length in ms.
        self.step = cjstep * 1e6 # The sample step in microsec.
        if cjargs:
            self.cmax = cjargs[0] * 1000 # in mM.
            self.cb = cjargs[1] * 1000 # Background concentration in mM.
            if self.profile == 'rcj':
                # 'rcj' profile.
                self.centre = cjargs[2] * 1000 # Pulse centre in ms.
                self.rise = cjargs[4] * 1e6 # 10-90% rise time for error function in microsec.
                self.decay = cjargs[5] * 1e6 # 90-10% decay time for error function in microsec.
                self.width = cjargs[3] * 1000 # Pulse halfwidth in ms.
            elif self.profile == 'instexp':
                self.prepulse = cjargs[2] * 1000 # Time before pulse starts (ms)
                self.tdec = cjargs[3] * 1000 # Decay time constant (ms)
            elif self.profile == 'square':
                self.prepulse = cjargs[2] * 1000 # Time before pulse starts (ms)
                self.width = cjargs[3] * 1000 # Pulse halfwidth in ms.    
            elif self.profile == 'square2':
                self.prepulse = cjargs[2] * 1000 # Time before pulse starts (ms)
                self.width = cjargs[3] * 1000 # Pulse halfwidth in ms. 
                self.inter = cjargs[4] * 1000 # Interpulse interval in ms. 
        else:
            # Default values
            self.cmax = 1 # in mM.
            self.cb = 0.0 # Background concentration in mM.
            # 'rcj' profile.
            self.centre = 10 # Pulse centre in ms.
            self.rise = 200 # 10-90% rise time for error function in microsec.
            self.decay = 200 # 90-10% decay time for error function in microsec.
            self.width = 10 # Pulse halfwidth in ms.
            # 'instexp' profile
            self.prepulse = self.reclength / 10.0 # Time before pulse starts (ms)
            self.tdec = 2.5 # Decay time constant (ms)
            self.inter = 10 # Interpulse interval in ms. 

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Concentration pulse profile:"))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Record length (millisec):"))
        self.reclengthEdit = QLineEdit(unicode(self.reclength))
        self.reclengthEdit.setMaxLength(12)
        self.connect(self.reclengthEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.reclengthEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Sampling interval (microsec):"))
        self.stepEdit = QLineEdit(unicode(self.step))
        self.stepEdit.setMaxLength(12)
        self.connect(self.stepEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.stepEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Pulse concentration (mM):"))
        self.concEdit = QLineEdit(unicode(self.cmax))
        self.concEdit.setMaxLength(12)
        self.connect(self.concEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.concEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Background concentration (mM):"))
        self.bckgrconcEdit = QLineEdit(unicode(self.cb))
        self.bckgrconcEdit.setMaxLength(12)
        self.connect(self.bckgrconcEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.bckgrconcEdit)
        layoutMain.addLayout(layout)

        if self.profile == 'rcj':

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Pulse centre position (millisec):"))
            self.centreEdit = QLineEdit(unicode(self.centre))
            self.centreEdit.setMaxLength(12)
            self.connect(self.centreEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.centreEdit)
            layoutMain.addLayout(layout)

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Pulse 10-90% rise time (microsec):"))
            self.riseEdit = QLineEdit(unicode(self.rise))
            self.riseEdit.setMaxLength(12)
            self.connect(self.riseEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.riseEdit)
            layoutMain.addLayout(layout)

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Pulse 90-10% decay time (microsec):"))
            self.decayEdit = QLineEdit(unicode(self.decay))
            self.decayEdit.setMaxLength(12)
            self.connect(self.decayEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.decayEdit)
            layoutMain.addLayout(layout)

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Concentration pulse width (millisec):"))
            self.widthEdit = QLineEdit(unicode(self.width))
            self.widthEdit.setMaxLength(12)
            self.connect(self.widthEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.widthEdit)
            layoutMain.addLayout(layout)

        elif self.profile == 'instexp':

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Time before pulse (ms):"))
            self.prepulseEdit = QLineEdit(unicode(self.prepulse))
            self.prepulseEdit.setMaxLength(12)
            self.connect(self.prepulseEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.prepulseEdit)
            layoutMain.addLayout(layout)

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Decay time constant (ms):"))
            self.decayEdit = QLineEdit(unicode(self.tdec))
            self.decayEdit.setMaxLength(12)
            self.connect(self.decayEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.decayEdit)
            layoutMain.addLayout(layout)

        elif self.profile == 'square':

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Time before pulse (ms):"))
            self.prepulseEdit = QLineEdit(unicode(self.prepulse))
            self.prepulseEdit.setMaxLength(12)
            self.connect(self.prepulseEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.prepulseEdit)
            layoutMain.addLayout(layout)

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Concentration pulse width (millisec):"))
            self.widthEdit = QLineEdit(unicode(self.width))
            self.widthEdit.setMaxLength(12)
            self.connect(self.widthEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.widthEdit)
            layoutMain.addLayout(layout)
        
        elif self.profile == 'square2':

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Time before pulse (ms):"))
            self.prepulseEdit = QLineEdit(unicode(self.prepulse))
            self.prepulseEdit.setMaxLength(12)
            self.connect(self.prepulseEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.prepulseEdit)
            layoutMain.addLayout(layout)

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Concentration pulse width (millisec):"))
            self.widthEdit = QLineEdit(unicode(self.width))
            self.widthEdit.setMaxLength(12)
            self.connect(self.widthEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.widthEdit)
            layoutMain.addLayout(layout)
            
            layout = QHBoxLayout()
            layout.addWidget(QLabel("Interval between pulses (millisec):"))
            self.interEdit = QLineEdit(unicode(self.inter))
            self.interEdit.setMaxLength(12)
            self.connect(self.interEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.interEdit)
            layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Design concentration pulse...")

    def on_par_changed(self):
        """
        """

        self.step = float(self.stepEdit.text()) * 1e-6
        self.reclength = float(self.reclengthEdit.text()) * 0.001
        self.cmax = float(self.concEdit.text()) * 0.001
        self.cb = float(self.bckgrconcEdit.text()) * 0.001

        if self.profile == 'rcj':
            self.centre = float(self.centreEdit.text()) * 0.001
            self.rise = float(self.riseEdit.text()) * 1e-6
            self.decay = float(self.decayEdit.text()) * 1e-6
            self.width = float(self.widthEdit.text()) * 0.001
        elif self.profile == 'instexp':
            self.prepulse = float(self.prepulseEdit.text()) * 0.001
            self.tdec = float(self.decayEdit.text()) * 0.001
        elif self.profile == 'square':
            self.prepulse = float(self.prepulseEdit.text()) * 0.001
            self.width = float(self.widthEdit.text()) * 0.001
        elif self.profile == 'square2':
            self.prepulse = float(self.prepulseEdit.text()) * 0.001
            self.width = float(self.widthEdit.text()) * 0.001
            self.inter = float(self.interEdit.text()) * 0.001

    def return_par(self):
        """
        Return parameter dictionary on exit.
        """

        if self.profile == 'rcj':
            cargs = (self.cmax, self.cb, self.centre, self.width,
                self.rise, self.decay)
            cfunc = cjumps.pulse_erf
        elif self.profile == 'instexp':
            cargs = (self.cmax, self.cb, self.prepulse, self.tdec)
            cfunc = cjumps.pulse_instexp
        elif self.profile == 'square':
            cargs = (self.cmax, self.cb, self.prepulse, self.width)
            cfunc = cjumps.pulse_square
        elif self.profile == 'square2':
            cargs = (self.cmax, self.cb, self.prepulse, self.width, self.inter)
            cfunc = cjumps.pulse_square_paired
        return self.reclength, self.step, cfunc, cargs

class CJumpParDlg2(QDialog):
    """
    Dialog to squatre concentration pulse width and concentration range.
    """
    def __init__(self, parent=None, width = 0.01, cmin=1e-6, cmax=0.001):
        super(CJumpParDlg2, self).__init__(parent)

        self.cmin = cmin * 1000 # in mM.
        self.cmax = cmax * 1000 # in mM.
        self.width = width * 1000 # Pulse halfwidth in ms.

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Square concentration pulse:"))

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

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Concentration pulse width (millisec):"))
        self.widthEdit = QLineEdit(unicode(self.width))
        self.widthEdit.setMaxLength(12)
        self.connect(self.widthEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.widthEdit)
        layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Square concentration pulse pulse...")

    def on_par_changed(self):
        """
        """
        self.cmin = float(self.conc1Edit.text()) * 0.001
        self.cmax = float(self.conc2Edit.text()) * 0.001
        self.width = float(self.widthEdit.text()) * 0.001

    def return_par(self):
        """
        Return parameter dictionary on exit.
        """
        return self.cmin, self.cmax, self.width

class ConcProfileDlg(QDialog):
    """
    """
    def __init__(self, parent=None):
        super(ConcProfileDlg, self).__init__(parent)

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Concentration pulse profile:"))

        self.squareRB = QRadioButton("&Square pulse")
        self.square2RB = QRadioButton("&Paired square pulses")
        self.realisticRB = QRadioButton("&Realistic pulse")
        self.realisticRB.setChecked(True)
        self.instexpRB = QRadioButton("&Instantaneous rise and exponential decay")
        
        layoutMain.addWidget(self.squareRB)
        layoutMain.addWidget(self.square2RB)
        layoutMain.addWidget(self.realisticRB)
        layoutMain.addWidget(self.instexpRB)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Choose concentration pulse...")

    def return_par(self):
        if self.instexpRB.isChecked():
            profile = 'instexp'
        elif self.realisticRB.isChecked():
            profile = 'rcj'
        elif self.squareRB.isChecked():
            profile = 'square'
        elif self.square2RB.isChecked():
            profile = 'square2'
        return profile

class BurstPlotDlg(QDialog):
    """
    Dialog to input burst separation and plotting parameters.
    """
    def __init__(self, parent=None):
        super(BurstPlotDlg, self).__init__(parent)

        self.tcrit = 4 # Critical time interval.

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Define burst:"))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Critical time interval (millisec):"))
        self.tcritEdit = QLineEdit(unicode(4))
        self.tcritEdit.setMaxLength(10)
        self.connect(self.tcritEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.tcritEdit)
        layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Define burst...")

    def on_par_changed(self):
        self.tcrit = float(self.tcritEdit.text())

    def return_par(self):
        return self.tcrit * 0.001 # Return tcrit in sec
    
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
