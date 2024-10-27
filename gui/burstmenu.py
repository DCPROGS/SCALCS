import numpy as np
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
except:
    raise ImportError("pyqt module is missing")

from gui import myqtcommon
from scalcs import scburst

class BurstMenu(QMenu):
    """
    """
    def __init__(self, parent):
        super(BurstMenu, self).__init__(parent) 
        self.parent = parent
        self.setTitle('&Bursts')
        
        self.my_colour = ["r", "g", "b", "m", "c", "y"]
        
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
            
        self.addActions([plotBurstLenPDFAction, plotBurstLenPDFActionCond,
            plotBurstOpeningDistrAction, plotBurstOpeningDistrActionCond,
            plotBurstLenVConcAction])
            
    def onPlotBrstLenPDF(self):
        """
        Display the burst length distribution.
        """
        self.parent.txtPltBox.clear()
        str = ('\t===== BURST LENGTH PDF =====\n' +
            'Agonist concentration = {0:.5g} microM\n'.
            format(self.parent.conc * 1000000) +
            'Ideal pdf- blue solid line.\n' +
            'Individual components- blue dashed lines.\n')
        self.parent.txtPltBox.append(str)
        
        dialog = myqtcommon.ConcDlg(self, self.parent.conc)
        if dialog.exec_():
            self.parent.conc = dialog.return_par()
        self.parent.mec.set_eff('c', self.parent.conc)

        q_burst = scburst.BurstDisplay(self.parent.mec)
        self.parent.log.write(q_burst.print_all)

        t, fbrst = q_burst.calculate_burst_length_pdf(multicomp=True)
        self.parent.present_plot = np.vstack((t*1000, fbrst))
        
        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.semilogx(t*1000, fbrst[0], 'b-')
        for i in range(1, len(fbrst)):
            self.parent.canvas.axes.semilogx(t*1000, fbrst[i], 'b--')
        self.parent.canvas.axes.set_yscale('sqrtscale')
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()

    def onPlotBrstLenPDFCond(self):
        """
        Display the conditional burst length distribution.
        """
        self.parent.txtPltBox.clear()
        str = ('===== BURST LENGTH PDF ' +
            '\nCONDITIONAL ON STARTING STATE =====\n' +
            'Agonist concentration = {0:.5g} microM\n'.
            format(self.parent.conc * 1000000) +
            'Ideal pdf- blue solid line.')
        self.parent.txtPltBox.append(str)
        
        dialog = myqtcommon.ConcDlg(self, self.parent.conc)
        if dialog.exec_():
            self.parent.conc = dialog.return_par()
        self.parent.mec.set_eff('c', self.parent.conc)

        q_burst = scburst.BurstDisplay(self.parent.mec)
        t, fbst, cfbst = q_burst.calculate_burst_length_pdf(conditional=True)
        self.parent.present_plot = np.vstack((t*1000, fbst, cfbst))
        self.parent.canvas.axes.clear()

        # TODO: only 6 colours are available now.        
        for i in range(self.parent.mec.kA):
            self.parent.canvas.axes.semilogx(t*1000, cfbst[i], self.my_colour[i]+'-',
                label="State {0:d}".format(i+1))
        self.parent.canvas.axes.semilogx(t*1000, fbst, 'k-', label="Not conditional")
        handles, labels = self.parent.canvas.axes.get_legend_handles_labels()
        self.parent.canvas.axes.legend(handles, labels, frameon=False)

        self.parent.canvas.axes.set_yscale('sqrtscale')
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()

    def onPlotBrstOpDistr(self):
        """
        Display the distribution of number of openings per burst.
        """

        self.parent.txtPltBox.clear()
        self.parent.txtPltBox.append('===== DISTRIBUTION OF NUMBER OF' +
            'OPENINGS PER BURST =====')
        
        dialog = myqtcommon.ConcDlg(self, self.parent.conc)
        if dialog.exec_():
            self.parent.conc = dialog.return_par()
        self.parent.mec.set_eff('c', self.parent.conc)
        q_burst = scburst.BurstDisplay(self.parent.mec)
        # TODO: need dialog to enter n
        n = 10
        r, Pr = q_burst.calculate_burst_openings_pdf(n)
        self.parent.present_plot = np.vstack((r, Pr))

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.plot(r, Pr,'ro')
        self.parent.canvas.axes.set_xlim(0, 11)
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()

    def onPlotBrstOpDistrCond(self):
        """
        Display the conditional distribution of number of openings per burst.
        """

        self.parent.txtPltBox.clear()
        self.parent.txtPltBox.append('===== DISTRIBUTION OF NUMBER OF ' +
            'OPENINGS PER BURST \nCONDITIONAL ON STARTING STATE=====')
        dialog = myqtcommon.ConcDlg(self, self.parent.conc)
        if dialog.exec_():
            self.parent.conc = dialog.return_par()
        self.parent.mec.set_eff('c', self.parent.conc)
        q_burst = scburst.BurstDisplay(self.parent.mec)
        n = 10
        r, Pr, cPr = q_burst.calculate_burst_openings_pdf(n, conditional=True)
        self.parent.present_plot = np.vstack((r, Pr, cPr))

        self.parent.canvas.axes.clear()
        # TODO: only 6 colours are available now.
        for i in range(self.parent.mec.kA):
            self.parent.canvas.axes.plot(r, cPr[i], self.my_colour[i]+'o',
                label="State {0:d}".format(i+1))
        self.parent.canvas.axes.plot(r, Pr,'ko', label="Not conditional")
        handles, labels = self.parent.canvas.axes.get_legend_handles_labels()
        self.parent.canvas.axes.legend(handles, labels, frameon=False)
        self.parent.canvas.axes.set_xlim(0, n+1)
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()

    def onPlotBrstLenConc(self):
        """
        Display mean burst length versus concentration plot.
        """
        self.parent.txtPltBox.clear()
        str = ('===== MEAN BURST LENGTH VERSUS CONCENTRATION =====\n' +
            'Solid line: mean burst length versus concentration.' +
            '    X-axis: microMols; Y-axis: ms.' +
            'Dashed line: corrected for fast block.')
        self.parent.txtPltBox.append(str)

        # TODO: need dialog to enter concentration range.
        dialog = myqtcommon.ConcRangeDlg(self)
        if dialog.exec_():
            cmin, cmax = dialog.return_par()

#        cmin = 10e-9
#        cmax = 0.005
        q_burst = scburst.BurstDisplay(self.parent.mec)
        c, br, brblk = q_burst.burst_length_versus_conc_plot(self.parent.mec, cmin, cmax)
        self.parent.present_plot = np.vstack((c, br, brblk))

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.plot(c*1e6, br*1e3,'r-', c*1e6, brblk*1e3, 'r--')
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()
        
        
