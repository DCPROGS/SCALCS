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

class SaveMenu(QMenu):
    """
    """
    def __init__(self, parent):
        super(SaveMenu, self).__init__(parent) 
        self.parent = parent
        self.setTitle('&Save')
        
        savePrintOutAction = myqtcommon.createAction(self,
            "&All work to text file", self.onSavePrintOut)
        savePlotASCII = myqtcommon.createAction(self, 
            "&Save current plot to text file", self.onSavePlotASCII)
        self.addActions([savePrintOutAction, savePlotASCII])
        
    def onSavePrintOut(self):
        """
        """
        printOutFilename, filt = QFileDialog.getSaveFileName(self,
                "Save as PRT file...", ".prt",
                "PRT files (*.prt)")

        self.parent.textBox.selectAll()
        text = self.parent.textBox.toPlainText()
        fout = open(printOutFilename,'w')
        fout.write(text)
        fout.close()

        self.parent.txtPltBox.clear()
        self.parent.txtPltBox.append('Saved printout file:')
        self.parent.txtPltBox.append(printOutFilename)

    def onSavePlotASCII(self):

        savePlotTXTFilename, filt = QFileDialog.getSaveFileName(self,
                "Save as TXT file...", self.parent.path, ".txt",
                "TXT files (*.txt)")

        fout = open(savePlotTXTFilename,'w')
        for i in range(self.parent.present_plot.shape[1]):
            for j in range(self.parent.present_plot.shape[0]):
                fout.write('{0:.6e}\t'.format(self.parent.present_plot[j, i]))
            fout.write('\n')
        fout.close()

        self.parent.txtPltBox.append('Current plot saved in text file:')
        self.parent.txtPltBox.append(savePlotTXTFilename)


