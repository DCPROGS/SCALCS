import sys

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

class ScalcsMenu(QMenu):
    """
    """
    def __init__(self, parent):
        super(ScalcsMenu, self).__init__(parent) 
        self.parent = parent
        self.setTitle('&Open/Shut')
        
        self.my_colour = ["r", "g", "b", "m", "c", "y"]

        plotOpenTimePDFAction = myqtcommon.createAction(self, 
            "&Open time pdf", self.onPlotOpenTimePDF)
        plotShutTimePDFAction = myqtcommon.createAction(self, 
            "&Shut time pdf", self.onPlotShutTimePDF)
        plotSubsetTimePDFAction = myqtcommon.createAction(self, 
            "&Subset time pdf", self.onPlotSubsetTimePDF)
        plotCorrOpenShutAction = myqtcommon.createAction(self, 
            "&Correlations", self.onPlotOpShCorr)
        plotAdjacentOpenShutAction = myqtcommon.createAction(self, 
            "&Open time adjacent to shut time range pdf", 
            self.onPlotOpAdjShacent)
        plotMeanOpenNextShutAction = myqtcommon.createAction(self, 
            "&Mean open time preceding/next to shut time",
            self.onPlotMeanOpNextShut)
        plotDependencyAction = myqtcommon.createAction(self, 
            "&Dependency plot", self.onPlotDependency)
        plotPopenAction = myqtcommon.createAction(self, "&Popen curve",
            self.onPlotPopen)

        self.addActions([plotOpenTimePDFAction, plotShutTimePDFAction,
            plotAdjacentOpenShutAction, plotMeanOpenNextShutAction, 
            plotCorrOpenShutAction, 
#            plotDependencyAction,
#            plotSubsetTimePDFAction.setDisabled(True),
            plotPopenAction])
#        self.insertSeparator(plotPopenAction)
        
    def onPlotPopen(self):
        """
        Display Popen curve.
        """

        self.parent.txtPltBox.clear()
        dialog = myqtcommon.ResDlg(self, self.parent.tres)
        if dialog.exec_():
            self.parent.tres = dialog.return_par()
        str = ('\t===== Popen PLOT =====\n' +
            'Resolution = {0:.5g} mikrosec\n'.format(self.parent.tres * 1000000) +
            'Ideal curve- red dashed line. \nHJC curve- blue solid line.\n')
        self.parent.txtPltBox.append(str)

        self.parent.log.write(popen.printout(self.parent.mec, self.parent.tres))
        c, pe, pi = scpl.Popen(self.parent.mec, self.parent.tres)
        self.parent.present_plot = np.vstack((c, pe, pi))

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.semilogx(c, pe, 'b-', c , pi, 'r--')
        self.parent.canvas.axes.set_ylim(0, 1)
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()

    def onPlotOpShCorr(self):
        """
        Display open, shut and open-shut time correlations.
        """

        self.parent.txtPltBox.clear()
        dialog = myqtcommon.ConcDlg(self, self.parent.conc)
        if dialog.exec_():
            self.parent.conc = dialog.return_par()
        str = ('\t===== OPEN, SHUT, OPEN/SHUT CORRELATION PLOTS =====\n' +
            'Agonist concentration = {0:.5g} mikroM\n'.
            format(self.parent.conc * 1000000) +
            'Shut time correlation - red circles.\n' +
            'Open time correlation - green circles\n' +
            'Open-shut time correlation - blue circles')
        self.parent.txtPltBox.append(str)

        self.parent.mec.set_eff('c', self.parent.conc)
        self.parent.log.write(scl.printout_correlations(self.parent.mec))
        # TODO: need dialog to enter lag value. 
        lag = 5
        n, roA, roF, roAF = scpl.corr_open_shut(self.parent.mec, lag)
        self.parent.present_plot = np.vstack((n, roA, roF, roAF))

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.plot(n, roA,'go', n, roF, 'ro', n, roAF, 'bo')
        self.parent.canvas.axes.axhline(y=0, xmin=0, xmax=1, color='k')
        self.parent.canvas.axes.set_xlim(0, 6)
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()
        
    def onPlotOpAdjShacent(self):
        """
        Display open time adjacent to shut time range pdf.
        """
        self.parent.txtPltBox.clear()
        dialog = myqtcommon.ConcDlg(self, self.parent.conc)
        if dialog.exec_():
            self.parent.conc = dialog.return_par()
        str = ('\t===== OPEN TIME ADJACENT TO SHUT TIME RANGE PDF =====\n' +
            'Agonist concentration = {0:.5g} mikroM\n'.
            format(self.parent.conc * 1000000) +
            'Ideal open time pdf- red dashed line.\n' +
            'Open times adjacent to shut time range pdf- blue solid line.\n')
        self.parent.txtPltBox.append(str)

        self.parent.mec.set_eff('c', self.parent.conc)
        # TODO: need dialog to enter lag value. 
        dialog = myqtcommon.ShutRangeDlg(self)
        if dialog.exec_():
            u1, u2 = dialog.return_par()
        self.parent.log.write(scl.printout_adjacent(self.parent.mec, u1, u2))
        
        t, ipdf, ajpdf = scpl.adjacent_open_time_pdf(self.parent.mec, 
            self.parent.tres, u1, u2)
        self.parent.present_plot = np.vstack((t, ipdf, ajpdf))

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.semilogx(t, ipdf, 'r--', t, ajpdf, 'b-')
        self.parent.canvas.axes.set_yscale('sqrtscale')
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()
        
    def onPlotMeanOpNextShut(self):
        """
        Display mean open time preceding / next-to shut time plot.
        """
        self.parent.txtPltBox.clear()
        dialog = myqtcommon.ConcResDlg(self, self.parent.conc, self.parent.tres)
        if dialog.exec_():
            self.parent.conc, self.parent.tres = dialog.return_par()
        str = ('\t===== MEAN OPEN TIME PRECEDING / NEXT TO SHUT TIME =====\n' +
            'Agonist concentration = {0:.5g} mikroM\n'.
            format(self.parent.conc * 1000000) +
            'Mean open time preceding specified shut time- red dashed line.\n' +
            'Mean open time next to specified shut time- blue dashed line.')
        self.parent.txtPltBox.append(str)

        self.parent.mec.set_eff('c', self.parent.conc)
        sht, mp, mn = scpl.mean_open_next_shut(self.parent.mec, self.parent.tres)
        self.parent.present_plot = np.vstack((sht, mp, mn))

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.semilogx(sht, mp, 'r--', sht, mn, 'b--')
#        self.axes.set_ylim(bottom=0)
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()
        
    def onPlotDependency(self):
        """
        Display dependency plot.
        """
        
        self.parent.txtPltBox.clear()
        dialog = myqtcommon.ConcDlg(self, self.parent.conc)
        if dialog.exec_():
            self.parent.conc = dialog.return_par()
        str = ('\t===== DEPENDENCY PLOT =====\n' +
            'Agonist concentration = {0:.5g} mikroM'.
            format(self.parent.conc * 1000000) +
            'Resolution = {0:.5g} mikrosec'.format(self.parent.tres * 1000000) +
            'X and Y axis are in ms')
        self.parent.txtPltBox.append(str)

        self.parent.mec.set_eff('c', self.parent.conc)
        to, ts, d = scpl.dependency_plot(self.parent.mec, self.parent.tres, points=128)
        
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

        self.parent.txtPltBox.clear()
        dialog = myqtcommon.ConcResDlg(self, self.parent.conc, self.parent.tres)
        if dialog.exec_():
            self.parent.conc, self.parent.tres = dialog.return_par()
        str = ('\t===== OPEN TIME PDF =====\n' +
            'Agonist concentration = {0:.5g} mikroM\n'.
            format(self.parent.conc * 1000000) +
            'Resolution = {0:.5g} mikrosec\n'.format(self.parent.tres * 1000000) +
            'Ideal pdf- red dashed line.\n' +
            'Exact pdf- blue solid line.\n' +
            'Asymptotic pdf- green solid line.')
        self.parent.txtPltBox.append(str)

        self.parent.mec.set_eff('c', self.parent.conc)

        try:
            text = scl.printout_occupancies(self.parent.mec, self.parent.tres)
            self.parent.log.write(text)
            text = scl.printout_distributions(self.parent.mec, self.parent.tres)
            self.parent.log.write(text)
        except:
            sys.stderr.write("main: Warning: unable to prepare printout.")
        
        t, ipdf, epdf, apdf = scpl.open_time_pdf(self.parent.mec, self.parent.tres)
        self.parent.present_plot = np.vstack((t, ipdf, epdf, apdf))

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.semilogx(t, ipdf, 'r--', t, epdf, 'b-', t, apdf, 'g-')
        self.parent.canvas.axes.set_yscale('sqrtscale')
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()

    def onPlotSubsetTimePDF(self):
        """
        """
        
        dialog = myqtcommon.ConcDlg(self, self.parent.conc)
        if dialog.exec_():
            self.parent.conc = dialog.return_par()
        self.parent.txtPltBox.clear()
        str = ('\t===== SUBSET TIME PDF =====\n' +
            'Agonist concentration = {0:.5g} mikroM\n'.
            format(self.parent.conc * 1000000) +
            'Resolution = {0:.5g} mikrosec\n'.
            format(self.parent.tres * 1000000) +
            'Ideal pdf- red dashed line.\n' +
            'Subset life time pdf- blue solid line.')
        self.parent.txtPltBox.append(str)

        self.parent.mec.set_eff('c', self.parent.conc)
        # TODO: need dialog to enter state1 and state2
        state1 = 8
        state2 = 10

        t, ipdf, spdf = scpl.subset_time_pdf(self.parent.mec, self.parent.tres,
            state1, state2)
        self.parent.present_plot = np.vstack((t, ipdf, s))

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.semilogx(t, spdf, 'b-', t, ipdf, 'r--')
        self.parent.canvas.axes.set_yscale('sqrtscale')
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()

    def onPlotShutTimePDF(self):
        """
        Display shut time probability density function.
        """

        self.parent.txtPltBox.clear()
        dialog = myqtcommon.ConcResDlg(self, self.parent.conc, self.parent.tres)
        if dialog.exec_():
            self.parent.conc, self.parent.tres = dialog.return_par()
        str = ('\t===== SHUT TIME PDF =====\n' +
            'Agonist concentration = {0:.5g} mikroM\n'.
            format(self.parent.conc * 1000000) +
            'Resolution = {0:.5g} mikrosec\n'.
            format(self.parent.tres * 1000000) +
            'Ideal pdf- red dashed line.\n' +
            'Exact pdf- blue solid line.\n' +
            'Asymptotic pdf- green solid line.')
        self.parent.txtPltBox.append(str)

        self.parent.mec.set_eff('c', self.parent.conc)
        self.parent.log.write(scl.printout_tcrit(self.parent.mec))
        t, ipdf, epdf, apdf = scpl.shut_time_pdf(self.parent.mec, self.parent.tres)
        self.parent.present_plot = np.vstack((t, ipdf, epdf, apdf))

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.semilogx(t, ipdf, 'r--', t, epdf, 'b-', t, apdf, 'g-')
        self.parent.canvas.axes.set_yscale('sqrtscale')
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()
