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
import popen
import samples
import scplotlib as scpl

from myqtlibs.mechmenu import MechMenu
from myqtlibs.burstmenu import BurstMenu
from myqtlibs.jumpmenu import JumpMenu
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

        self.menuBar().addMenu(MechMenu(self))

        plotMenu = self.menuBar().addMenu('&Plot')
        plotPopenAction = myqtcommon.createAction(self, "&Popen curve", self.onPlotPopen)
        plotOpenTimePDFAction = myqtcommon.createAction(self, 
            "&Open time pdf", self.onPlotOpenTimePDF)
        plotShutTimePDFAction = myqtcommon.createAction(self, 
            "&Shut time pdf", self.onPlotShutTimePDF)
        plotSubsetTimePDFAction = myqtcommon.createAction(self, 
            "&Subset time pdf", self.onPlotSubsetTimePDF)
        plotCorrOpenShutAction = myqtcommon.createAction(self, 
            "&Correlations", self.onPlotOpShCorr)
        plotAdjacentOpenShutAction = myqtcommon.createAction(self, 
            "&Open time adjacent to shut time range pdf", self.onPlotOpAdjShacent)
        plotMeanOpenNextShutAction = myqtcommon.createAction(self, 
            "&Mean open time preceding/next to shut time", self.onPlotMeanOpNextShut)
        plotDependencyAction = myqtcommon.createAction(self, 
            "&Dependency plot", self.onPlotDependency)

        myqtcommon.addActions(plotMenu, (plotOpenTimePDFAction, plotShutTimePDFAction,
            plotAdjacentOpenShutAction, plotMeanOpenNextShutAction, 
            plotCorrOpenShutAction, plotDependencyAction,
            plotSubsetTimePDFAction.setDisabled(True),
            plotPopenAction))
        plotMenu.insertSeparator(plotPopenAction)

        self.menuBar().addMenu(BurstMenu(self))
        self.menuBar().addMenu(JumpMenu(self))

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
        
        dialog = myqtcommon.ResDlg(self, self.tres)
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
        
        dialog = myqtcommon.ConcDlg(self, self.conc)
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
        
        dialog = myqtcommon.ConcDlg(self, self.conc)
        if dialog.exec_():
            self.conc = dialog.return_par()
        self.mec.set_eff('c', self.conc)
        # TODO: need dialog to enter lag value. 
        dialog = myqtcommon.ShutRangeDlg(self)
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

        dialog = myqtcommon.ConcResDlg(self, self.conc, self.tres)
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
        
        dialog = myqtcommon.ConcDlg(self, self.conc)
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
        
        dialog = myqtcommon.ConcResDlg(self, self.conc, self.tres)
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
        
        dialog = myqtcommon.ConcDlg(self, self.conc)
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
        
        dialog = myqtcommon.ConcResDlg(self, self.conc, self.tres)
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

