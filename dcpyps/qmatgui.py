#! /usr/bin/python
"""
A simple GUI for DC_PyPs project.
Depends on pyqt and matplotlib modules.
"""
import time
import sys
import os
import socket

import numpy as np
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
    from matplotlib import transforms as mtransforms
    from matplotlib import ticker
except:
    raise ImportError("matplotlib module is missing")

import scalcslib as scl
import cjumpslib as cjl
import usefulib as ufl
import dataset
import io
import samples

class QMatGUI(QMainWindow):
    def __init__(self, parent=None):
        super(QMatGUI, self).__init__(parent)
        self.resize(1000, 700)     # wide, high in px
        self.mainFrame = QWidget()

        self.mec = samples.CH82()
        self.mec.KBlk = 0.01
        self.mec.fastblk = False
        self.conc = 100e-9    # 100 nM
        self.tres = 0.0001
        self.rec1 = None

        loadMenu = self.menuBar().addMenu('&Load')
        loadDemoAction = self.createAction("&Demo", self.onLoadDemo,
            None, "loaddemo", "Load Demo mec")
        loadFromMecFileAction = self.createAction("&From Mec File...",
            self.onLoadMecFile,
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
        plotJumpAction = self.createAction(
            "&Realistic concentration jump", self.onPlotCJump)
        self.addActions(plotMenu, (plotPopenAction, plotOpenTimePDFAction,
            plotShutTimePDFAction, plotBurstLenPDFAction,
            plotBurstOpeningDistrAction, plotBurstLenVConcAction,
            plotJumpAction))

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
        likelihoodAction = self.createAction("&Calculate likelihood",
            self.onCalculateLikelihood)

        self.addActions(dataMenu, (openScanAction, imposeResolutionAction,
            plotDataOpenAction, plotDataShutAction, plotDataBurstAction,
            likelihoodAction))

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
        mscale.register_scale(SquareRootScale)

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

    def onLoadData(self):
        """
        """

        self.textBox.append('\n\n\t===== LOADING DATA FILE =====')
        filename = QFileDialog.getOpenFileName(self,
            "Open SCN File...", "", "DC SCN Files (*.scn)")
        ioffset, nint, calfac, header = io.scn_read_header(filename)
        tint, iampl, iprops = io.scn_read_data(filename, ioffset, nint, calfac)
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

        #self.mec.set_eff('c', self.conc)
        opts = {}
        opts['mec'] = self.mec
        opts['conc'] = self.conc
        opts['tres'] = self.tres
        opts['tcrit'] = self.tcrit
        opts['isCHS'] = True

        rates = self.mec.get_rates()

        #loglik = scl.HJClik(rates, self.rec1.bursts, opts)
        #self.rec1.bursts, self.mec, self.tres, self.tcrit,
        #    True)

        newrates, loglik = ufl.simplex(rates, self.rec1.bursts, scl.HJClik,
            opts, verbose=0)
        mec.set_rates(newrates)

        self.textBox.append('\nLog Likelihood = {0:.3f}'.
            format(loglik))
        self.mec.printout(self.log)


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

        x, y = dataset.prepare_hist(blength, self.tres)

        self.axes.clear()
        self.axes.semilogx(x, y, 'b-')
        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotDataOpen(self):
        """
        """
        self.textBox.append('\n\n\t===== PLOTTING DATA: OPEN PERIODS =====')

        self.textBox.append('\nNumber of open periods = {0:d}'.
            format(len(self.rec1.opint)))
        self.textBox.append('Average = {0:.3f} millisec'.
            format(np.average(self.rec1.opint)))
        self.textBox.append('Range: {0:.3f}'.format(min(self.rec1.opint)) +
            ' to {0:.3f} millisec'.format(max(self.rec1.opint)))

        x, y = dataset.prepare_hist(self.rec1.opint, self.tres)

        self.axes.clear()
        self.axes.semilogx(x, y, 'b-')
        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotDataShut(self):
        """
        """
        self.textBox.append('\n\n\t===== PLOTTING DATA: OPEN PERIODS =====')
        self.textBox.append('\nNumber of shut periods = {0:d}'.
            format(len(self.rec1.shint)))
        self.textBox.append('Average = {0:.3f} millisec'.
            format(np.average(self.rec1.shint)))
        self.textBox.append('Range: {0:.3f}'.format(min(self.rec1.shint)) +
            ' to {0:.3f} millisec'.format(max(self.rec1.shint)))

        x, y = dataset.prepare_hist(self.rec1.shint, self.tres)

        self.axes.clear()
        self.axes.semilogx(x, y, 'b-')
        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotCJump(self):
        """
        Display realistic concentration jump.
        """

        self.textBox.append('\n\n\t===== REALISTIC CONCENTRATION JUMP =====')
        self.textBox.append('Concentration profile- green solid line.')
        self.textBox.append('Relaxation- blue solid line.')

        dialog = CJumpParDlg(self)
        if dialog.exec_():
            jpar = dialog.return_par()

        self.textBox.append('\nConcentration pulse profile:')
        self.textBox.append('Concentration = {0:.3f} mM'
            .format(jpar['peak_conc'] * 1000))
        self.textBox.append('10- 90% rise time = {0:.0f} microsec'
            .format(jpar['rise_time']))
        self.textBox.append('Pulse width = {0:.1f} millisec'
            .format(jpar['pulse_width'] * 0.001))
            
        # TODO: get slowest relaxation tau and automaticly calculate
        # record length.
        jumpd, relaxd = cjl.rcj_single(self.mec, jpar)

        # TODO: relaxed contains Popen trace only. Need to plot occupancies too.
        t, cjump, relax = cjl.convert_to_arrays(jumpd, relaxd, self.mec.kA)
        maxR = max(relax)
        maxJ = max(cjump)
        cjump1 = (cjump / maxJ) * 0.2 * maxR + 1.02* maxR

        self.axes.clear()
        self.axes.plot(t * 0.001, relax,'b-', t * 0.001, cjump1, 'g-')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotPopen(self):
        """
        Display Popen curve.
        """

        self.textBox.append('\n\n\t===== Popen PLOT =====')
        self.textBox.append('Resolution = {0:.2f} mikrosec'.
            format(self.tres * 1000000))
        self.textBox.append('Ideal curve- red dashed line.')
        self.textBox.append('HJC curve- blue solid line.')

        # Calculate EC50, nH and maxPopen for Popen curve
        # corrected for missed events.
        emaxPopen, econc = scl.get_maxPopen(self.mec, self.tres)
        eEC50 = scl.get_EC50(self.mec, self.tres)
        enH = scl.get_nH(self.mec, self.tres)
        self.textBox.append('\nHJC Popen curve:\nmaxPopen = {0:.3f}; '
            .format(emaxPopen) + ' EC50 = {0:.3f} mikroM; '
            .format(eEC50 * 1000000) + ' nH = {0:.3f}'.format(enH))

        # Calculate EC50, nH and maxPopen for ideal Popen curve.
        imaxPopen, iconc = scl.get_maxPopen(self.mec, 0)
        iEC50 = scl.get_EC50(self.mec, 0)
        inH = scl.get_nH(self.mec, 0)
        self.textBox.append('\nIdeal Popen curve:\nmaxPopen = {0:.3f}; '
            .format(imaxPopen) + ' EC50 = {0:.3f} mikroM; '
            .format(iEC50 * 1000000) + ' nH = {0:.3f}'.format(inH))

        cmin = iEC50 / 20 #20000000.0
        cmax = iEC50 * 500 #/ 1000000.0
        logstart = int(np.log10(cmin)) - 1
        logend = int(np.log10(cmax)) - 1
        decades = int(logend - logstart)
        logstep = 0.01    # increase this if want more points per curve
        points = int(decades / logstep + 1)

        c = np.zeros(points)
        pe = np.zeros(points)
        pi = np.zeros(points)
        for i in range(points):
            c[i] = pow(10, logstart + logstep * i)
            pe[i] = scl.popen(self.mec, self.tres, c[i])
            pi[i] = scl.popen(self.mec, 0, c[i])
        c = c * 1000000 # x axis in mikroMolar scale

        self.axes.clear()
        self.axes.semilogx(c, pe, 'b-', c , pi, 'r--')
        self.axes.set_ylim(0, 1)
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        #self.axes.set_xlabel('Agonist concentration, mikroM')
        self.canvas.draw()

    def onPlotOpenTimePDF(self):
        """
        Display open time probability density function.
        """

        self.textBox.append('\n\t===== OPEN TIME PDF =====')
        self.textBox.append('Agonist concentration = {0:.6f} mikroM'.
            format(self.conc * 1000000))
        self.textBox.append('Resolution = {0:.2f} mikrosec'.
            format(self.tres * 1000000))
        self.textBox.append('Ideal pdf- red dashed line.')
        self.textBox.append('Exact pdf- blue solid line.')
        self.textBox.append('Asymptotic pdf- green solid line.')
        self.mec.set_eff('c', self.conc)
        open = True
        
        # Ideal pdf
        tau, area = scl.get_ideal_pdf_components(self.mec, open)
        self.textBox.append('\nIDEAL OPEN TIME DISTRIBUTION')
        self.textBox.append('term\ttau (ms)\tarea (%)\trate const (1/sec)')
        for i in range(self.mec.kA):
            self.textBox.append('{0:d}'.format(i+1) +
            '\t{0:.3f}'.format(tau[i] * 1000) +
            '\t{0:.3f}'.format(area[i] * 100) +
            '\t{0:.3f}'.format(1.0 / tau[i]))

        # Asymptotic pdf
        roots = scl.asymptotic_roots(self.mec, self.tres, open)
        areas = scl.asymptotic_areas(self.mec, self.tres, roots, open)
        self.textBox.append('\nASYMPTOTIC OPEN TIME DISTRIBUTION')
        self.textBox.append('term\ttau (ms)\tarea (%)\trate const (1/sec)')
        for i in range(self.mec.kA):
            self.textBox.append('{0:d}'.format(i+1) +
            '\t{0:.3f}'.format(-1.0 / roots[i] * 1000) +
            '\t{0:.3f}'.format(areas[i] * 100) +
            '\t{0:.3f}'.format(- roots[i]))
        areast0 = np.zeros(self.mec.kA)
        for i in range(self.mec.kA):
            areast0[i] = areas[i] * np.exp(- self.tres * roots[i])
        areast0 = areast0 / np.sum(areast0)
        self.textBox.append('Areas for asymptotic pdf renormalised for t=0 to\
        infinity (and sum=1), so areas can be compared with ideal pdf.')
        for i in range(self.mec.kA):
            self.textBox.append('{0:d}'.format(i+1) +
            '\t{0:.3f}'.format(areast0[i] * 100))
        mean = scl.hjc_mean_time(self.mec, self.tres, open)
        self.textBox.append('Mean open time (ms) = {0:.6f}'.format(mean * 1000))


        # Exact pdf
        eigvals, gamma00, gamma10, gamma11 = scl.GAMAxx(self.mec,
            self.tres, open)
        self.textBox.append('\nEXACT OPEN TIME DISTRIBUTION')
        self.textBox.append('eigen\tg00(m)\tg10(m)\tg11(m)')
        for i in range(self.mec.k):
            self.textBox.append('{0:.3f}'.format(eigvals[i]) +
            '\t{0:.3f}'.format(gamma00[i]) +
            '\t{0:.3f}'.format(gamma10[i]) +
            '\t{0:.3f}'.format(gamma11[i]))

        tmax = (-1 / roots.max()) * 20
        tmin = 0.00001 # 10 mikrosec
        points = 512
        step = (np.log10(tmax) - np.log10(tmin)) / (points - 1)
        t = np.zeros(points)

        # Ideal pdf.
        f = 0.0
        for i in range(self.mec.kA):
            f += area[i] * np.exp(-self.tres / tau[i])
        fac = 1 / f # Scale factor.
        ipdf = np.zeros(points)
        for i in range(points):
            t[i] = tmin * pow(10, (i * step))
            ipdf[i] = t[i] * scl.pdf_open_time(self.mec, t[i]) * fac

        # Asymptotic pdf
        apdf = np.zeros(points)
        for i in range(points):
            apdf[i] = t[i] * scl.pdf_exponential(t[i], self.tres, roots, areas)

        # Exact pdf
        epdf = np.zeros(points)
        for i in range(points):
            epdf[i] = (t[i] * scl.pdf_exact(t[i], self.tres,
                roots, areas, eigvals, gamma00, gamma10, gamma11))

        t = t * 1000 # x scale in millisec
        #self.textBox.append(text1)
        self.axes.clear()
        self.axes.semilogx(t, ipdf, 'r--', t, epdf, 'b-', t, apdf, 'g-')
        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotShutTimePDF(self):
        """
        Display shut time probability density function.
        """

        self.textBox.append('\n\t===== SHUT TIME PDF =====')
        self.textBox.append('Agonist concentration = {0:.6f} mikroM'.
            format(self.conc * 1000000))
        self.textBox.append('Resolution = {0:.2f} mikrosec'.
            format(self.tres * 1000000))
        self.textBox.append('Ideal pdf- red dashed line.')
        self.textBox.append('Exact pdf- blue solid line.')
        self.textBox.append('Asymptotic pdf- green solid line.')
        self.mec.set_eff('c', self.conc)
        open = False

        # Ideal pdf
        tau, area = scl.get_ideal_pdf_components(self.mec, open)
        self.textBox.append('\nIDEAL SHUT TIME DISTRIBUTION')
        self.textBox.append('term\ttau (ms)\tarea (%)\trate const (1/sec)')
        for i in range(self.mec.kF):
            self.textBox.append('{0:d}'.format(i+1) +
            '\t{0:.3f}'.format(tau[i] * 1000) +
            '\t{0:.3f}'.format(area[i] * 100) +
            '\t{0:.3f}'.format(1.0 / tau[i]))

        # Asymptotic pdf
        roots = scl.asymptotic_roots(self.mec, self.tres, open)
        areas = scl.asymptotic_areas(self.mec, self.tres, roots, open)
        self.textBox.append('\nASYMPTOTIC SHUT TIME DISTRIBUTION')
        self.textBox.append('term\ttau (ms)\tarea (%)\trate const (1/sec)')
        for i in range(self.mec.kF):
            self.textBox.append('{0:d}'.format(i+1) +
            '\t{0:.3f}'.format(-1.0 / roots[i] * 1000) +
            '\t{0:.3f}'.format(areas[i] * 100) +
            '\t{0:.3f}'.format(- roots[i]))
        areast0 = np.zeros(self.mec.kF)
        for i in range(self.mec.kF):
            areast0[i] = areas[i] * np.exp(- self.tres * roots[i])
        areast0 = areast0 / np.sum(areast0)
        self.textBox.append('Areas for asymptotic pdf renormalised for t=0 to\
        infinity (and sum=1), so areas can be compared with ideal pdf.')
        for i in range(self.mec.kF):
            self.textBox.append('{0:d}'.format(i+1) +
            '\t{0:.3f}'.format(areast0[i] * 100))
        mean = scl.hjc_mean_time(self.mec, self.tres, open)
        self.textBox.append('Mean shut time (ms) = {0:.6f}'.format(mean * 1000))

        # Exact pdf
        eigvals, gamma00, gamma10, gamma11 = scl.GAMAxx(self.mec,
            self.tres, open)
        self.textBox.append('\nEXACT SHUT TIME DISTRIBUTION')
        self.textBox.append('eigen\tg00(m)\tg10(m)\tg11(m)')
        for i in range(self.mec.k):
            self.textBox.append('{0:.3f}'.format(eigvals[i]) +
            '\t{0:.3f}'.format(gamma00[i]) +
            '\t{0:.3f}'.format(gamma10[i]) +
            '\t{0:.3f}'.format(gamma11[i]))


        tmax = (-1 / roots.max()) * 20
        tmin = 0.00001 # 10 mikrosec
        points = 512
        step = (np.log10(tmax) - np.log10(tmin)) / (points - 1)
        t = np.zeros(points)

        # Ideal pdf.
        f = 0.0
        for i in range(self.mec.kF):
            f += area[i] * np.exp(-self.tres / tau[i])
        fac = 1 / f # Scale factor.
        ipdf = np.zeros(points)
        for i in range(points):
            t[i] = tmin * pow(10, (i * step))
            ipdf[i] = t[i] * scl.pdf_shut_time(self.mec, t[i]) * fac

        # Asymptotic pdf
        apdf = np.zeros(points)
        for i in range(points):
            apdf[i] = t[i] * scl.pdf_exponential(t[i], self.tres, roots, areas)

        # Exact pdf
        epdf = np.zeros(points)
        for i in range(points):
            epdf[i] = (t[i] * scl.pdf_exact(t[i], self.tres,
                roots, areas, eigvals, gamma00, gamma10, gamma11))

        t = t * 1000 # x scale in millisec
        #self.textBox.append(text1)
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
        self.textBox.append('\n\t===== BURST LENGTH PDF =====')
        self.textBox.append('Agonist concentration = {0:.6f} mikroM'.
            format(self.conc * 1000000))
        self.textBox.append('Resolution = {0:.2f} mikrosec'.
            format(self.tres * 1000000))
        self.textBox.append('Ideal pdf- blue solid line.')

        self.mec.set_eff('c', self.conc)

        tau, area = scl.get_burst_ideal_pdf_components(self.mec)
        self.textBox.append('\nBURST LENGTH DISTRIBUTION')
        self.textBox.append('term\ttau (ms)\tarea (%)')
        for i in range(self.mec.kE):
            self.textBox.append('{0:d}'.format(i+1) +
            '\t{0:.3f}'.format(tau[i] * 1000) +
            '\t{0:.3f}'.format(area[i] * 100))

        m = scl.mean_burst_length(self.mec)
        self.textBox.append('\nMean burst length = {0:.3f} millisec'.
            format(m * 1000))
        mu = scl.mean_num_burst_openings(self.mec)
        self.textBox.append('Mean number of openings per burst = {0:.3f}'.
            format(mu))
        mop = scl.mean_open_time_burst(self.mec)
        self.textBox.append('The mean total open time per burst = {0:.3f} '.
            format(mop * 1000) + 'millisec')
        bpop = mop / m
        self.textBox.append('Popen WITHIN BURST = (open time/bst)/(bst length)\
            = {0:.3f} '.format(bpop))

        points = 512
        tmin = 0.00001
        tmax = max(tau) * 20
        step = (np.log10(tmax) - np.log10(tmin)) / (points - 1)

        t = np.zeros(points)
        fbst = np.zeros(points)
        for i in range(points):
            t[i] = tmin * pow(10, (i * step))
            fbst[i] = t[i] * scl.pdf_burst_length(self.mec, t[i])
        t = t * 1000 # x axis in millisec
        fbrst = fbst
        
        self.axes.clear()
        self.axes.semilogx(t, fbrst, 'b-')
        self.axes.set_yscale('sqrtscale')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onPlotBrstOpDistr(self):
        """
        Display the distribution of number of openings per burst.
        """
        self.textBox.append('\nCalculating burst parameters:')
        self.textBox.append('Agonist concentration = %e M' %self.conc)
        self.mec.set_eff('c', self.conc)
        mu = scl.mean_num_burst_openings(self.mec)
        self.textBox.append('Mean number of openings per burst = %f' %mu)

        n = 10
        r = np.arange(1, n+1)
        Pr = np.zeros(n)
        for i in range(n):
            Pr[i] = scl.distr_num_burst_openings(self.mec, r[i])

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

        cmin = 10e-9
        cmax = 0.1
        points = 100
        step = (cmax - cmin)/(points - 1)
        c = np.zeros(points)
        br = np.zeros(points)

        if self.mec.fastblk:
            brblk = np.zeros(points)
            for i in range(points):
                c[i] = cmin + step * i
                self.mec.set_eff('c', c[i])
                br[i] = scl.mean_burst_length(self.mec)
                brblk[i] = br[i] * (1 + c[i] / self.mec.KBlk)
            c = c * 1000000 # x axis scale in mikroMoles
            br = br * 1000
            brblk= brblk * 1000
            self.axes.plot(c, br,'b-', c, brblk, 'g-')
        else:
            for i in range(points):
                c[i] = cmin + step * i
                self.mec.set_eff('c', c[i])
                br[i] = scl.mean_burst_length(self.mec)
            c = c * 1000000 # x axis scale in mikroMoles
            br = br * 1000
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
        self.mec = samples.CH82()
        self.textBox.append("\nLoaded Demo.\n")
        self.mec.printout(self.log)
        
    def onLoadMecFile(self):
        """
        Load a mechanism and rates from DC's mec file.
        Called from menu Load|From Mec File...
        """
        filename = QFileDialog.getOpenFileName(self,
            "Open Mec File...", "", "DC Mec Files (*.mec)")
        self.textBox.append("\nFile to read: " + os.path.split(str(filename))[1])

        version, meclist, max_mecnum = io.mec_get_list(filename)
        self.textBox.append("Mec file version: %d; contains %d mechanisms."
            %(version, max_mecnum))

        dialog = MecListDlg(meclist, max_mecnum, self)
        if dialog.exec_():
            nrate = dialog.returnRates()

        self.mec = io.mec_load(filename, meclist[nrate][0])

        self.textBox.append("Loaded mec: " + meclist[nrate][2])
        self.textBox.append("Loaded rates: " + meclist[nrate][3] + "\n")

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

class CJumpParDlg(QDialog):
    """
    Dialog to input realistic concentration pulse parameters.
    """
    def __init__(self, parent=None):
        super(CJumpParDlg, self).__init__(parent)

        self.step = 8 # The sample step. All time params in microseconds.
        self.centre = 10000
        self.rise = 250 # list of 10-90% rise times for error functions
        self.width = 10000
        self.reclength = 50000
        self.conc = 10e-6 # in molar

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Concentration pulse profile:"))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Sampling interval (microsec):"))
        self.stepEdit = QLineEdit(unicode(8))
        self.stepEdit.setMaxLength(6)
        self.connect(self.stepEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.stepEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Pulse centre position (microsec):"))
        self.centreEdit = QLineEdit(unicode(10000))
        self.centreEdit.setMaxLength(6)
        self.connect(self.centreEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.centreEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Pulse 10-90% rise time (microsec):"))
        self.riseEdit = QLineEdit(unicode(250))
        self.riseEdit.setMaxLength(6)
        self.connect(self.riseEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.riseEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Concentration pulse width (microsec):"))
        self.widthEdit = QLineEdit(unicode(10000))
        self.widthEdit.setMaxLength(6)
        self.connect(self.widthEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.widthEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Record length (microsec):"))
        self.reclengthEdit = QLineEdit(unicode(50000))
        self.reclengthEdit.setMaxLength(6)
        self.connect(self.reclengthEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.reclengthEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Pulse concentration (mM):"))
        self.concEdit = QLineEdit(unicode(0.01))
        self.concEdit.setMaxLength(6)
        self.connect(self.concEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.concEdit)
        layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        #self.resize(1000, 500)
        self.setWindowTitle("Design realistic concentration pulse...")

    def on_par_changed(self):
        """
        """

        self.par = {}
        self.par['step_size'] = int(self.stepEdit.text())
        self.par['pulse_centre'] = int(self.centreEdit.text())
        self.par['rise_time'] = float(self.riseEdit.text())
        self.par['pulse_width'] = int(self.widthEdit.text())
        self.par['record_length'] = int(self.reclengthEdit.text())
        # Concentration convert from mM to M
        self.par['peak_conc'] = float(self.concEdit.text()) * 0.001

    def return_par(self):
        """
        Return parameter dictionary on exit.
        """
        return self.par

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
        self.tcrit = int(self.tcritEdit.text())

    def return_par(self):
        return self.tcrit * 0.001 # Return tcrit in sec

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

class SquareRootScale(mscale.ScaleBase):
    """
    Class for generating sqare root scaled axis for probability density
    function plots.
    """

    name = 'sqrtscale'
    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)
    def get_transform(self):
        """
        Set the actual transform for the axis coordinates.
        """
        return self.SqrTransform()
    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to reasonable defaults.
        """
        axis.set_major_formatter(ticker.ScalarFormatter())

    class SqrTransform(mtransforms.Transform):
        """
        """
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mtransforms.Transform.__init__(self)
        def transform(self, a):
            """
            Take numpy array and return transformed copy.
            """
            return np.sqrt(a)
        def inverted(self):
            """
            Get inverse transform.
            """
            return SquareRootScale.InvertedSqrTransform()

    class InvertedSqrTransform(mtransforms.Transform):
        """
        """
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mtransforms.Transform.__init__(self)
        def transform(self, a):
            return np.power(a, 2)
        def inverted(self):
            return SquareRootScale.SqrTransform()
