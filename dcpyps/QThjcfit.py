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

try:
#    from PyQt4.QtCore import *
#    from PyQt4.QtGui import *
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")

try:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
    from matplotlib.figure import Figure
    from matplotlib import scale as mscale
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter, FuncFormatter
    import matplotlib.pyplot as plt
#    from matplotlib import transforms as mtransforms
#    from matplotlib import ticker
except:
    raise ImportError("matplotlib module is missing")

import numpy as np

from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.optimize import minimize

import scalcslib as scl
import cjumps
import scburst
import popen
import dcio
import samples
import scplotlib as scpl
import mechanism

import optimize
import dataset
import qtcommonlib as qtcl

from dcprogs.likelihood import Log10Likelihood

class QhjcGUI(QMainWindow):
    def __init__(self, parent=None):
        super(QhjcGUI, self).__init__(parent)
        self.resize(800, 600)     # wide, high in px
        self.mainFrame = QWidget()
        self.setWindowTitle("DC_PyPs: HJCFIT- fit of model to open-shut times with missed events")
        #self.setWindowIcon(QIcon("./dcpyps/samples/HJCFIT.ICO"))
        
        self.mec = None
        self.scnfiles = []
        self.tres = []
        self.tcrit = []
        self.conc = []
        self.recs = []
        self.bursts = []
        self.iternum = 0
        self.likelihood = []
        self.theta = None


        loadMenu = self.menuBar().addMenu('&Load Fit')
        loadDemo1Action = qtcl.createAction(self,
            "&Load demo fit: Burzomato 2004", self.onLoadDemo_Burzomato,
            None, "loaddemoburz", "Load Demo Burzomato")
        quitAction = qtcl.createAction(self, "&Quit", self.close,
            "Ctrl+Q", "appquit", "Close the application")
        qtcl.addActions(loadMenu, (loadDemo1Action, quitAction))
        
        loadMechMenu = qtcl.addMechMenuElements(self)
        
        loadMenu = self.menuBar().addMenu('&Run Fit')
        loadDemo2Action = qtcl.createAction(self,
            "&Run Fit", self.onStartFitting,
            None, "startfit", "Start Fitting")
        quitAction = qtcl.createAction(self, "&Fit Settings", self.onFittingSettings,
            None, "setfit", "Fitting settings")
        qtcl.addActions(loadMenu, (loadDemo2Action, quitAction))
        
        
        self.textBox = QTextBrowser()
        # Set here if printout to TextBox only or also to file or console.
        self.log = qtcl.PrintLog(self.textBox) #, sys.stdout)
        str1, str2, str3 = qtcl.startInfo()
        self.textBox.append(str1)
        self.textBox.append(str2)
        self.textBox.append(str3)
        
        rightVBox = QVBoxLayout()
        rightVBox.addWidget(self.textBox)
        self.mainFrame.setLayout(rightVBox)
        self.setCentralWidget(self.mainFrame)

    def onLoadDemo_Burzomato(self):
        """
        Load demo fit - GlyR set BGVH from Burzomato 2004.
        """

        # LOAD FLIP MECHANISM USED Burzomato et al 2004
        mecfn = "./dcpyps/samples/demomec.mec"
        version, meclist, max_mecnum = dcio.mec_get_list(mecfn)
        self.mec = dcio.mec_load(mecfn, meclist[2][0])
        rates = [5000.0, 500.0, 2700.0, 2000.0, 800.0, 15000.0, 300.0, 0.1200E+06, 6000.0, 0.4500E+09, 1500.0, 12000.0, 4000.0, 0.9000E+09, 7500.0, 1200.0, 3000.0, 0.4500E+07, 2000.0, 0.9000E+07, 1000, 0.135000E+08]
        self.mec.set_rateconstants(rates)
        self.textBox.append("\nLoaded Burzomato 2003 (flip) mechanism.\n")


        # Fixed rates.
        #fixed = np.array([False, False, False, False, False, False, False, True,
        #    False, False, False, False, False, False])
        #if fixed.size == len(mec.Rates):
        for i in range(len(self.mec.Rates)):
            self.mec.Rates[i].fixed = False


        # Constrained rates.
        self.mec.Rates[21].is_constrained = True
        self.mec.Rates[21].constrain_func = mechanism.constrain_rate_multiple
        self.mec.Rates[21].constrain_args = [17, 3]
        self.mec.Rates[19].is_constrained = True
        self.mec.Rates[19].constrain_func = mechanism.constrain_rate_multiple
        self.mec.Rates[19].constrain_args = [17, 2]
        self.mec.Rates[16].is_constrained = True
        self.mec.Rates[16].constrain_func = mechanism.constrain_rate_multiple
        self.mec.Rates[16].constrain_args = [20, 3]
        self.mec.Rates[18].is_constrained = True
        self.mec.Rates[18].constrain_func = mechanism.constrain_rate_multiple
        self.mec.Rates[18].constrain_args = [20, 2]
        self.mec.Rates[8].is_constrained = True
        self.mec.Rates[8].constrain_func = mechanism.constrain_rate_multiple
        self.mec.Rates[8].constrain_args = [12, 1.5]
        self.mec.Rates[13].is_constrained = True
        self.mec.Rates[13].constrain_func = mechanism.constrain_rate_multiple
        self.mec.Rates[13].constrain_args = [9, 2]

        self.mec.Rates[7].mr=True
        self.mec.Rates[15].mr=True
        self.mec.update_constrains()
        self.mec.update_mr()

        self.mec.printout(self.log)
        
        # LOAD DATA.
        self.scnfiles = ["./dcpyps/samples/A-10.scn", "./dcpyps/samples/B-30.scn",
            "./dcpyps/samples/C-100.scn", "./dcpyps/samples/D-1000.scn"]
        self.tres = [0.000030, 0.000030, 0.000030, 0.000030]
        self.tcrit = [0.004, -1, -0.06, -0.02]
        self.conc = [10e-6, 30e-6, 100e-6, 1000e-6]
        
        for i in range(len(self.scnfiles)):
            rec = load_data(self.scnfiles[i], self.tres[i], math.fabs(self.tcrit[i]), output=self.log)
            self.recs.append(rec)
            self.bursts.append(rec.bursts)

        self.theta = np.log(self.mec.theta())

        kwargs = {'nmax': 2, 'xtol': 1e-12, 'rtol': 1e-12, 'itermax': 100,
            'lower_bound': -1e6, 'upper_bound': 0}
        
        for i in range(len(self.conc)):
            self.likelihood.append(Log10Likelihood(self.bursts[i], self.mec.kA,
                self.tres[i], self.tcrit[i], **kwargs))
                
        lik = self.dcprogslik(self.theta)
        self.textBox.append("\nStarting likelihood (DCprogs)= {0:.6f}".format(-lik))
        
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
        filename, filt = QFileDialog.getOpenFileName(self,
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
        
    def onLoadPrtFile(self):
        """
        Load a mechanism and rates from DC's HJCFIT.PRT file.
        Called from menu Load|From DCPROGS .PRT File...
        """
        filename, filt = QFileDialog.getOpenFileName(self,
            "Open Mec File...", self.path, "DC Mec Files (*.prt *.PRT *.txt *.TXT)")
        self.path = os.path.split(str(filename))[0]
        self.textBox.append("\nFile to read: " + os.path.split(str(filename))[1])

        self.mec = dcio.mec_load_from_prt(filename)
        self.textBox.append("Loaded mec and rates from PRT file: " + filename)
        self.mec.printout(self.log)

    def onLoadModFile(self):
        """
        Load a mechanism and rates from Channel Lab .mod file.
        Called from menu Load|From Channel Lab .MOD File...
        """
        filename, filt = QFileDialog.getOpenFileName(self,
            "Open MOD File...", self.path, "Channel Lab MOD Files (*.mod *.MOD)")
        self.path = os.path.split(str(filename))[0]
        self.textBox.append("\nFile to read: " + os.path.split(str(filename))[1])

        self.mec, title = dcio.mod_load(filename)
        self.textBox.append("\n" + title + "\n")
        self.mec.printout(self.log)

    def onModifyMec(self):
        """
        """
        table = qtcl.RateTableDlg(self, self.mec, self.log)
        if table.exec_():
            self.mec = table.return_mec()

    def onModifyStates(self):
        """
        """
        table = qtcl.StateTableDlg(self, self.mec, self.log)
        if table.exec_():
            self.mec = table.return_mec()
        
    def onStartFitting(self):

        start = time.clock()
        success = False
        result = None
        while not success:
            #res = minimize(dcprogslik, np.log(theta), method='Powell', callback=printit, options={'maxiter': 5000, 'disp': True})
            result = minimize(self.dcprogslik, self.theta, method='Nelder-Mead', callback=self.printiter,
                options={'xtol':1e-4, 'ftol':1e-4, 'maxiter': 5000, 'maxfev': 10000,
                'disp': True})
            if result.success:
                success = True
            else:
                self.theta = result.x

        self.textBox.append("\nDCPROGS Fitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
                %time.localtime()[0:6])
        self.textBox.append('time in simplex= {0:.6f}'.format(time.clock() - start))
        self.textBox.append('\n\nresult=')
        self.textBox.append(result)

        self.textBox.append('\n Final log-likelihood = {0:.6f}'.format(-result.fun))
        self.textBox.append('\n Number of iterations = {0:d}'.format(result.nit))
        self.textBox.append('\n Number of evaluations = {0:d}'.format(result.nfev))
        self.mec.theta_unsqueeze(np.array_str(np.exp(result.x)))
        self.textBox.append("\n Final rate constants:")
        self.mec.printout(self.log)
        self.textBox.append('\n\n')
        
        open_pdfs, shut_pdfs = [], []
        for i in range(len(scnfiles)):
            open_pdfs.append(save_pdf_fig(scnfiles[i], recs[i].opint, mec, conc[i], tres[i], 'open'))
            shut_pdfs.append(save_pdf_fig(scnfiles[i], recs[i].shint, mec, conc[i], tres[i], 'shut'))
        resulthtml = save_html(str2, str3, mecfn, scnfiles, open_pdfs, shut_pdfs)
        self.textBox.append('\n\n Results saved in Loading '+resulthtml)


    def onFittingSettings(self):
        pass


    def dcprogslik(self, x, args=None):
        self.mec.theta_unsqueeze(np.exp(x))
        lik = 0
        for i in range(len(self.conc)):
            self.mec.set_eff('c', self.conc[i])
            lik += -self.likelihood[i](self.mec.Q) * math.log(10)
        return lik

    def printiter(self, theta):
        self.iternum += 1
        lik = self.dcprogslik(theta)
        print("iteration # {0:d}; log-lik = {1:.6f}".format(self.iternum, -lik))
        print(np.array_str(np.exp(theta)))        

        

        
        
        
        
def save_html(str2, str3, mecfn, scnfiles, open_pdfs, shut_pdfs):
    
    htmlstr = ("""<html>
    <p>HJCFIT: Fit of model to open-shut times with missed events
    (Uses HJC distributions, exact for first two deadtimes then asymptotic, to calculate likelihood of record)</p>""" +
    '<p>'+ str2 + '<p></p>' + str3 + """</p></html>""" )

    htmlfile = mecfn[:-4] + '.html'
    f = open(htmlfile, 'w')
    f.write(htmlstr)
    for i in range(len(scnfiles)):
        f.write("<p>{0} _____________________________________ {1}</p>".format(open_pdfs[i], shut_pdfs[i]))
        f.write("<img src={0} width='450' height='300'><img src={1} width='450' height='300'>".format(os.path.abspath(open_pdfs[i]), os.path.abspath(shut_pdfs[i])))

    #mec.printout(f)
    f.close()
    
    return htmlfile

        
def save_pdf_fig(scnfile, ints, mec, conc, tres, type):
    x, y, dx = dataset.prepare_hist(ints, tres)
    mec.set_eff('c', conc)
    if type == 'open':
        t, ipdf, epdf, apdf = scpl.open_time_pdf(mec, tres)
    elif type == 'shut':
        t, ipdf, epdf, apdf = scpl.shut_time_pdf(mec, tres)
    else:
        print 'Wrong type.'

    sipdf = scpl.scaled_pdf(t, ipdf, math.log10(dx), len(ints))
    sepdf = scpl.scaled_pdf(t, epdf, math.log10(dx), len(ints))
    figure(figsize=(6, 4))
    semilogx(x*1000, y, 'k-', t, sipdf, 'r--', t, sepdf, 'b-')

    outfile = scnfile[:-4] + '_' + type + '.png'
    savefig(outfile, bbox_inches=0)
    return outfile
        
        
def load_data(sfile, tres, tcrit, output=sys.stdout):
    output.write('\n\n Loading '+sfile)
    ioffset, nint, calfac, header = dcio.scn_read_header(sfile)
    tint, iampl, iprops = dcio.scn_read_data(sfile, ioffset, nint, calfac)
    rec = dataset.SCRecord(sfile, header, tint, iampl, iprops)
    # Impose resolution, get open/shut times and bursts.
    rec.impose_resolution(tres)
    output.write('\nNumber of resolved intervals = {0:d}'.format(len(rec.rtint)))

    rec.get_open_shut_periods()
    output.write('\nNumber of resolved periods = {0:d}'.format(len(rec.opint) + len(rec.shint)))
    output.write('\nNumber of open periods = {0:d}'.format(len(rec.opint)))
    output.write('Mean and SD of open periods = {0:.9f} +/- {1:.9f} ms'.
        format(np.average(rec.opint)*1000, np.std(rec.opint)*1000))
    output.write('Range of open periods from {0:.9f} ms to {1:.9f} ms'.
        format(np.min(rec.opint)*1000, np.max(rec.opint)*1000))
    output.write('\nNumber of shut intervals = {0:d}'.format(len(rec.shint)))
    output.write('Mean and SD of shut periods = {0:.9f} +/- {1:.9f} ms'.
        format(np.average(rec.shint)*1000, np.std(rec.shint)*1000))
    output.write('Range of shut periods from {0:.9f} ms to {1:.9f} ms'.
        format(np.min(rec.shint)*1000, np.max(rec.shint)*1000))
    output.write('Last shut period = {0:.9f} ms'.format(rec.shint[-1]*1000))

    rec.get_bursts(tcrit)
    output.write('\nNumber of bursts = {0:d}'.format(len(rec.bursts)))
    blength = rec.get_burst_length_list()
    output.write('Average length = {0:.9f} ms'.format(np.average(blength)*1000))
    output.write('Range: {0:.3f}'.format(min(blength)*1000) +
            ' to {0:.3f} millisec'.format(max(blength)*1000))
    openings = rec.get_openings_burst_list()
    output.write('Average number of openings= {0:.9f}'.format(np.average(openings)))
    return rec
