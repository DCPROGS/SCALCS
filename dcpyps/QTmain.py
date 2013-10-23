#! /usr/bin/python
"""
A simple GUI for DC_PyPs project.
Depends on pyqt and matplotlib modules.
"""
import sys
import numpy as np

try:
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")

try:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
    from matplotlib.figure import Figure
    from matplotlib import scale as mscale
    from matplotlib import cm
    import matplotlib.pyplot as plt
except:
    raise ImportError("matplotlib module is missing")

import scalcslib as scl
import cjumps
import scburst
import popen
import samples
import scplotlib as scpl

from myqtlibs.mechmenu import MechMenu
import myqtlibs.myqtcommon as myqtcommon

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
        self.data_loaded = False
        self.my_colour = ["r", "g", "b", "m", "c", "y"]
        self.present_plot = None

        self.cjprofile = 'rcj'
        self.cjlen = 0.05
        self.cjstep = 5e-6
        self.cjwidth = 0.00001
        self.cjfunc = None
        self.cjargs = None

        self.menuBar().addMenu(MechMenu(self))

        plotMenu = self.menuBar().addMenu('&Plot')
        plotPopenAction = myqtcommon.createAction(self, "&Popen curve", self.onPlotPopen)
        plotOpenTimePDFAction = myqtcommon.createAction(self, 
            "&Open time pdf", self.onPlotOpenTimePDF)
        plotShutTimePDFAction = myqtcommon.createAction(self, 
            "&Shut time pdf", self.onPlotShutTimePDF)
        plotSubsetTimePDFAction = myqtcommon.createAction(self, 
            "&Subset time pdf", self.onPlotSubsetTimePDF)
        plotBurstLenPDFAction = myqtcommon.createAction(self, 
            "&Burst length pdf", self.onPlotBrstLenPDF)
        plotBurstLenPDFActionCond = myqtcommon.createAction(self, 
            "&Conditional burst length pdf", self.onPlotBrstLenPDFCond)
        plotBurstOpeningDistrAction = myqtcommon.createAction(self, 
            "&Burst openings distribution", self.onPlotBrstOpDistr)
        plotBurstOpeningDistrActionCond = myqtcommon.createAction(self, 
            "&Conditional burst openings distribution", self.onPlotBrstOpDistrCond)
        plotBurstLenVConcAction = myqtcommon.createAction(self, 
            "&Burst length vs concentration", self.onPlotBrstLenConc)
        plotJumpPopenAction = myqtcommon.createAction(self, 
            "&Concentration jump: Popen", self.onPlotCJumpPopen)
        plotJumpOccupanciesAction = myqtcommon.createAction(self, 
            "&Concentration jump: occupancies",
            self.onPlotCJumpOccupancies)
        plotJumpOnOffTauConc = myqtcommon.createAction(self, 
            "&Concentration jump: weighted on/off tau versus concentration",
            self.onPlotCJumpRiseVConc)
        plotCorrOpenShutAction = myqtcommon.createAction(self, 
            "&Correlations", self.onPlotOpShCorr)
        plotAdjacentOpenShutAction = myqtcommon.createAction(self, 
            "&Open time adjacent to shut time range pdf", self.onPlotOpAdjShacent)
        plotMeanOpenNextShutAction = myqtcommon.createAction(self, 
            "&Mean open time preceding/next to shut time", self.onPlotMeanOpNextShut)
        plotDependencyAction = myqtcommon.createAction(self, 
            "&Dependency plot", self.onPlotDependency)
#        plotJump2PopenAction = self.createAction(self, 
#            "&Instant rise and exponential decay concentration jump: Popen", self.onPlotCJump2Popen)

        myqtcommon.addActions(plotMenu, (plotOpenTimePDFAction, plotShutTimePDFAction,
            plotAdjacentOpenShutAction, plotMeanOpenNextShutAction, 
            plotCorrOpenShutAction, plotDependencyAction,
            # setDisabled(False) to activate plotting the subset time distributions
            plotSubsetTimePDFAction.setDisabled(True),
            plotBurstLenPDFAction, plotBurstLenPDFActionCond,
            plotBurstOpeningDistrAction, plotBurstOpeningDistrActionCond,
            plotBurstLenVConcAction,
            plotJumpPopenAction, plotJumpOccupanciesAction, plotJumpOnOffTauConc,
#            plotJump2PopenAction,            
            plotPopenAction))
        plotMenu.insertSeparator(plotJumpPopenAction)
        plotMenu.insertSeparator(plotPopenAction)

        saveMenu = self.menuBar().addMenu('&Save')
        savePrintOutAction = myqtcommon.createAction(self, "&All work to text file", self.onSavePrintOut)
        savePlotASCII = myqtcommon.createAction(self, 
            "&Save current plot to text file", self.onSavePlotASCII)
        myqtcommon.addActions(saveMenu, (savePrintOutAction, savePlotASCII,
            None))

        helpMenu = self.menuBar().addMenu('&Help')
        helpAboutAction = myqtcommon.createAction(self, "&About", self.onHelpAbout)
        myqtcommon.addActions(helpMenu, (helpAboutAction, None))

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
        self.log = myqtcommon.PrintLog(self.textBox) #, sys.stdout)    
        myqtcommon.startInfo(self.log)

        plotSetLayout = QVBoxLayout()
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
        self.txtPltBox.append('Tau ON dominant component- red dashed line.')
        self.txtPltBox.append('Tau OFF - green solid line.')
        self.txtPltBox.append('X axis in mM; Y axis in ms.')
        self.txtPltBox.append("---\n")

        c, wton, ton, wtoff, toff  = scpl.conc_jump_on_off_taus_versus_conc_plot(self.mec,
            cmin, cmax, self.width)

        self.axes.clear()
        self.axes.semilogx(c, wton,'b-', c, wtoff, 'g-', c, ton[-1], 'r--')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

        self.present_plot = np.vstack((c, ton[-1], wton, wtoff))

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
        
        dialog = ResDlg(self, self.tres)
        if dialog.exec_():
            self.tres = dialog.return_par()

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
        
        dialog = ConcDlg(self, self.conc)
        if dialog.exec_():
            self.conc = dialog.return_par()
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
        
        dialog = ConcDlg(self, self.conc)
        if dialog.exec_():
            self.conc = dialog.return_par()
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
        
    def onPlotMeanOpNextShut(self):
        """
        Display mean open time preceding / next-to shut time plot.
        """
        self.txtPltBox.clear()
        self.txtPltBox.append('\t===== MEAN OPEN TIME PRECEDING / NEXT TO SHUT TIME =====')
        self.txtPltBox.append('Agonist concentration = {0:.5g} mikroM'.
            format(self.conc * 1000000))
        self.txtPltBox.append('Mean open time preceding specified shut time- red dashed line.')
        self.txtPltBox.append('Mean open time next to specified shut time- blue dashed line.')

        dialog = ConcResDlg(self, self.conc, self.tres)
        if dialog.exec_():
            self.conc, self.tres = dialog.return_par()
        self.mec.set_eff('c', self.conc)
        sht, mp, mn = scpl.mean_open_next_shut(self.mec, self.tres)
        self.present_plot = np.vstack((sht, mp, mn))

        self.axes.clear()
        self.axes.semilogx(sht, mp, 'r--', sht, mn, 'b--')
#        self.axes.set_ylim(bottom=0)
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()
        
    def onPlotDependency(self):
        """
        Display dependency plot.
        """
        
        self.txtPltBox.clear()
        self.txtPltBox.append('\t===== DEPENDENCY PLOT =====')
        self.txtPltBox.append('Agonist concentration = {0:.5g} mikroM'.
            format(self.conc * 1000000))
        self.txtPltBox.append('Resolution = {0:.5g} mikrosec'.
            format(self.tres * 1000000))
        self.txtPltBox.append('X and Y axis are in ms')
        
        dialog = ConcDlg(self, self.conc)
        if dialog.exec_():
            self.conc = dialog.return_par()
        self.mec.set_eff('c', self.conc)
        to, ts, d = scpl.dependency_plot(self.mec, self.tres, points=128)
        
        fig = plt.figure()
        fig.suptitle('Dependency plot', fontsize=12)
        ax = fig.gca(projection='3d')
        to, ts = np.meshgrid(to, ts)
        surf = ax.plot_surface(to, ts, d, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
        ax.set_zlim(-1.0, 1.0)
        
#        def log_10_product(x, pos):
#            """The two args are the value and tick position.
#            Label ticks with the product of the exponentiation"""
#            return '%1i' % (x)
#        
#        ax.set_xscale('log')
#        ax.set_yscale('log')
#        formatter = FuncFormatter(log_10_product)
#        ax.xaxis.set_major_formatter(formatter)
#        ax.yaxis.set_major_formatter(formatter)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        
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
        
        dialog = ConcResDlg(self, self.conc, self.tres)
        if dialog.exec_():
            self.conc, self.tres = dialog.return_par()
        self.mec.set_eff('c', self.conc)

        try:
            scl.printout_occupancies(self.mec, self.tres, output=self.log)
            scl.printout_distributions(self.mec, self.tres, output=self.log)
        except:
            sys.stderr.write("main: Warning: unable to prepare printout.")
        
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
        
        dialog = ConcDlg(self, self.conc)
        if dialog.exec_():
            self.conc = dialog.return_par()
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
        
        dialog = ConcResDlg(self, self.conc, self.tres)
        if dialog.exec_():
            self.conc, self.tres = dialog.return_par()

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
        
        dialog = ConcDlg(self, self.conc)
        if dialog.exec_():
            self.conc = dialog.return_par()
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
        
        dialog = ConcDlg(self, self.conc)
        if dialog.exec_():
            self.conc = dialog.return_par()
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
        
        dialog = ConcDlg(self, self.conc)
        if dialog.exec_():
            self.conc = dialog.return_par()
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
        
        dialog = ConcDlg(self, self.conc)
        if dialog.exec_():
            self.conc = dialog.return_par()
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
        self.txtPltBox.append('Solid line: mean burst length versus concentration.')
        self.txtPltBox.append('    X-axis: microMols; Y-axis: ms.')
        self.txtPltBox.append('Dashed line: corrected for fast block.')

        # TODO: need dialog to enter concentration range.
        dialog = ConcRangeDlg(self)
        if dialog.exec_():
            cmin, cmax = dialog.return_par()

#        cmin = 10e-9
#        cmax = 0.005
        c, br, brblk = scpl.burst_length_versus_conc_plot(self.mec, cmin, cmax)
        self.present_plot = np.vstack((c, br, brblk))

        self.axes.clear()
        self.axes.plot(c, br,'r-', c, brblk, 'r--')
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        self.canvas.draw()

    def onSavePrintOut(self):
        """
        """
        printOutFilename, filt = QFileDialog.getSaveFileName(self,
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

    def onSavePlotASCII(self):

        savePlotTXTFilename, filt = QFileDialog.getSaveFileName(self,
                "Save as TXT file...", self.path, ".txt",
                "TXT files (*.txt)")

        fout = open(savePlotTXTFilename,'w')
        for i in range(self.present_plot.shape[1]):
            for j in range(self.present_plot.shape[0]):
                fout.write('{0:.6e}\t'.format(self.present_plot[j, i]))
            fout.write('\n')
        fout.close()

        self.txtPltBox.append('Current plot saved in text file:')
        self.txtPltBox.append(savePlotTXTFilename)

    def onHelpAbout(self):
        """
        Display About dialog.
        Called from menu Help|About.
        """
        dialog = myqtcommon.AboutDlg(self)
        if dialog.exec_():
            pass



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
                try:
                    self.rise = cjargs[4] * 1e6 # 10-90% rise time for error function in microsec.
                    self.decay = cjargs[5] * 1e6 # 90-10% decay time for error function in microsec.
                except:
                    self.rise = 200 # 10-90% rise time for error function in microsec.
                    self.decay = 200 # 90-10% decay time for error function in microsec.
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
        layout.addWidget(QLabel("Pulse concentration (mM):"))
        self.concEdit = QLineEdit(unicode(self.cmax))
        self.concEdit.setMaxLength(12)
        self.connect(self.concEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.concEdit)
        layoutMain.addLayout(layout)



        if self.profile == 'rcj':

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Concentration pulse width (millisec):"))
            self.widthEdit = QLineEdit(unicode(self.width))
            self.widthEdit.setMaxLength(12)
            self.connect(self.widthEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.widthEdit)
            layoutMain.addLayout(layout)

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
        layout.addWidget(QLabel("Background concentration (mM):"))
        self.bckgrconcEdit = QLineEdit(unicode(self.cb))
        self.bckgrconcEdit.setMaxLength(12)
        self.connect(self.bckgrconcEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.bckgrconcEdit)
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

    def return_par(self):
        return self.tres * 1e-6 # Return tcrit in sec


