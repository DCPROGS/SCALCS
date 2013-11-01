import numpy as np
try:
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")

from dcpyps import scburst
from dcpyps import scplotlib as scpl
import myqtcommon

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
        scburst.printout_pdfs(self.parent.mec, output=self.parent.log)
        t, fbrst, mfbrst = scpl.burst_length_pdf(self.parent.mec, multicomp=True)
        self.parent.present_plot = np.vstack((t, fbrst, mfbrst))
        
        self.parent.axes.clear()
        self.parent.axes.semilogx(t, fbrst, 'b-')
        for i in range(self.parent.mec.kE):
            self.parent.axes.semilogx(t, mfbrst[i], 'b--')
        self.parent.axes.set_yscale('sqrtscale')
        self.parent.axes.xaxis.set_ticks_position('bottom')
        self.parent.axes.yaxis.set_ticks_position('left')
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

        t, fbst, cfbst = scpl.burst_length_pdf(self.parent.mec, conditional=True)
        self.parent.present_plot = np.vstack((t, fbst, cfbst))
        self.parent.axes.clear()

        # TODO: only 6 colours are available now.        
        for i in range(self.parent.mec.kA):
            self.parent.axes.semilogx(t, cfbst[i], self.my_colour[i]+'-',
                label="State {0:d}".format(i+1))
        self.parent.axes.semilogx(t, fbst, 'k-', label="Not conditional")
        handles, labels = self.parent.axes.get_legend_handles_labels()
        self.parent.axes.legend(handles, labels, frameon=False)

        self.parent.axes.set_yscale('sqrtscale')
        self.parent.axes.xaxis.set_ticks_position('bottom')
        self.parent.axes.yaxis.set_ticks_position('left')
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
        # TODO: need dialog to enter n
        n = 10
        r, Pr = scpl.burst_openings_pdf(self.parent.mec, n)
        self.parent.present_plot = np.vstack((r, Pr))

        self.parent.axes.clear()
        self.parent.axes.plot(r, Pr,'ro')
        self.parent.axes.set_xlim(0, 11)
        self.parent.axes.xaxis.set_ticks_position('bottom')
        self.parent.axes.yaxis.set_ticks_position('left')
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
        n = 10
        r, Pr, cPr = scpl.burst_openings_pdf(self.parent.mec, n, conditional=True)
        self.parent.present_plot = np.vstack((r, Pr, cPr))

        self.parent.axes.clear()
        # TODO: only 6 colours are available now.
        for i in range(self.parent.mec.kA):
            self.parent.axes.plot(r, cPr[i], self.my_colour[i]+'o',
                label="State {0:d}".format(i+1))
        self.parent.axes.plot(r, Pr,'ko', label="Not conditional")
        handles, labels = self.parent.axes.get_legend_handles_labels()
        self.parent.axes.legend(handles, labels, frameon=False)
        self.parent.axes.set_xlim(0, n+1)
        self.parent.axes.xaxis.set_ticks_position('bottom')
        self.parent.axes.yaxis.set_ticks_position('left')
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
        c, br, brblk = scpl.burst_length_versus_conc_plot(self.parent.mec, cmin, cmax)
        self.parent.present_plot = np.vstack((c, br, brblk))

        self.parent.axes.clear()
        self.parent.axes.plot(c, br,'r-', c, brblk, 'r--')
        self.parent.axes.xaxis.set_ticks_position('bottom')
        self.parent.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()
        
        
