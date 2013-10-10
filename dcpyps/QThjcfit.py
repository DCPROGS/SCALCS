#! /usr/bin/python
"""
A simple GUI for DC_PyPs project.
Depends on pyqt and matplotlib modules.
"""
import time
import sys
import os
import math

try:
#    from PyQt4.QtCore import *
#    from PyQt4.QtGui import *
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")

#try:
#    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
#    from matplotlib.figure import Figure
#    from matplotlib import scale as mscale
#    from mpl_toolkits.mplot3d import Axes3D
#    from matplotlib import cm
#    from matplotlib.ticker import LinearLocator, FormatStrFormatter, FuncFormatter
#    import matplotlib.pyplot as plt
##    from matplotlib import transforms as mtransforms
##    from matplotlib import ticker
#except:
#    raise ImportError("matplotlib module is missing")

from pylab import*
import numpy as np
from scipy.optimize import minimize

import dcio
import samples
import scplotlib as scpl
import mechanism
from myqtlibs.mechmenu import MechMenu
import myqtlibs.datamenu as datamenu
import myqtlibs.myqtcommon as myqtcommon
import dataset
from dcprogs.likelihood import Log10Likelihood

class QhjcGUI(QMainWindow):
    def __init__(self, parent=None):
        super(QhjcGUI, self).__init__(parent)
        self.resize(800, 600)     # wide, high in px
        self.mainFrame = QWidget()
        self.setWindowTitle("DC_PyPs: HJCFIT- fit of model to open-shut times with missed events")
        #self.setWindowIcon(QIcon("./dcpyps/samples/HJCFIT.ICO"))
        
        self.mec = None
        self.mecfn = None
        self.path = None
        self.scnfiles = []
        self.tres = []
        self.tcrit = []
        self.conc = []
        self.recs = []
        self.recs_old = []
        self.bursts = []
        self.iternum = 0
        self.likelihood = []
        self.theta = None


        loadMenu = self.menuBar().addMenu('&Load Fit')
        loadDemo1Action = myqtcommon.createAction(self,
            "&Load demo fit: Burzomato 2004", self.onLoadDemo_Burzomato,
            None, "loaddemoburz", "Load Demo Burzomato")
        quitAction = myqtcommon.createAction(self, "&Quit", self.close,
            "Ctrl+Q", "appquit", "Close the application")
        myqtcommon.addActions(loadMenu, (loadDemo1Action, quitAction))
        
        loadMechMenu = self.menuBar().addMenu(MechMenu(self))
        
        loadMenu = self.menuBar().addMenu('&Load Data')
        loadDataAction = myqtcommon.createAction(self,
            "&New Set", self.onLoadNewSet,
            None, "newset", "New Set")
        loadSavedAction = myqtcommon.createAction(self,
            "Load Saved Set", self.onLoadSavedSet,
            None, "loadsavedset", "Load Saved Set")
        saveSetAction = myqtcommon.createAction(self,
            "Save Set", self.onSaveSet,
            None, "saveset", "Save Set")
        myqtcommon.addActions(loadMenu, (loadDataAction, loadSavedAction, saveSetAction))
        
        loadMenu = self.menuBar().addMenu('&Run Fit')
        loadDemo2Action = myqtcommon.createAction(self,
            "&Run Fit", self.onStartFitting,
            None, "startfit", "Start Fitting")
        quitAction = myqtcommon.createAction(self, "&Fit Settings", self.onFittingSettings,
            None, "setfit", "Fitting settings")
        myqtcommon.addActions(loadMenu, (loadDemo2Action, quitAction))
        
        
        self.textBox = QTextBrowser()
        # Set here if printout to TextBox only or also to file or console.
        self.log = myqtcommon.PrintLog(self.textBox) #, sys.stdout)
        self.str1, self.str2, self.str3 = myqtcommon.startInfo()
        self.textBox.append(self.str1)
        self.textBox.append(self.str2)
        self.textBox.append(self.str3)
        
        rightVBox = QVBoxLayout()
        rightVBox.addWidget(self.textBox)
        self.mainFrame.setLayout(rightVBox)
        self.setCentralWidget(self.mainFrame)

###########  Called by menu 'LoadFit'
    def onLoadDemo_Burzomato(self):
        """
        Load demo fit - GlyR set BGVH from Burzomato 2004.
        """

        # LOAD FLIP MECHANISM USED Burzomato et al 2004
        self.mecfn = "./dcpyps/samples/demomec.mec"
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
            #TODO: load more than one scan file per concentration
            rec = load_data(self.scnfiles[i][0], self.tres[i], math.fabs(self.tcrit[i]), output=self.log)
            self.recs.append(rec)
            self.bursts.append(rec.bursts)
        

#################### Called by menu 'Load Mech'
            
#################### Called by menu 'Load Data'
    def onLoadNewSet(self):
        self.recs_old = self.recs
        self.recs = []
        dialog = datamenu.NewSetDlg(self, self.recs, self.log)
        if dialog.exec_():
            self.scnfiles, self.conc, self.tres, self.tcrit = dialog.return_set()

        for i in range(len(self.scnfiles)):
            #TODO: load more than one scan file per concentration
            rec = datamenu.load_data(self.scnfiles[i][0], self.tres[i], math.fabs(self.tcrit[i]), output=self.log)
            self.recs.append(rec)
            self.bursts.append(rec.bursts)
    
    def onLoadSavedSet(self):
        pass
    
    def onSaveSet(self):
        pass
        
#################### Called by menu 'Run Fit'
    def onStartFitting(self):
        
        

        self.theta = np.log(self.mec.theta())

        kwargs = {'nmax': 2, 'xtol': 1e-12, 'rtol': 1e-12, 'itermax': 100,
            'lower_bound': -1e6, 'upper_bound': 0}
        
        for i in range(len(self.conc)):
            self.likelihood.append(Log10Likelihood(self.bursts[i], self.mec.kA,
                self.tres[i], self.tcrit[i], **kwargs))
                
        lik = self.dcprogslik(self.theta)
        self.textBox.append("\nStarting likelihood (DCprogs)= {0:.6f}".format(-lik))

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
#        self.textBox.append('\n\nresult=')
#        self.textBox.append(result)

        self.textBox.append('\n Final log-likelihood = {0:.6f}'.format(-result.fun))
        self.textBox.append('\n Number of iterations = {0:d}'.format(result.nit))
        self.textBox.append('\n Number of evaluations = {0:d}'.format(result.nfev))

        self.mec.theta_unsqueeze(np.exp(result.x))
        self.textBox.append("\n Final rate constants:")
        self.mec.printout(self.log)
        self.textBox.append('\n\n')
        
        open_pdfs, shut_pdfs = [], []
        for i in range(len(self.scnfiles)):
            open_pdfs.append(save_pdf_fig(self.scnfiles[i], self.recs[i].opint,
                self.mec, self.conc[i], self.tres[i], 'open'))
            shut_pdfs.append(save_pdf_fig(self.scnfiles[i], self.recs[i].shint,
                self.mec, self.conc[i], self.tres[i], 'shut'))
        resulthtml = save_html(self.str2, self.str3, self.mecfn, self.scnfiles, open_pdfs, shut_pdfs)
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
    (Uses HJC distributions, exact for first two deadtimes then asymptotic,
    to calculate likelihood of record)</p>""" +
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

    outfile = scnfile[0][:-4] + '_' + type + '.png'
    savefig(outfile, bbox_inches=0)
    return outfile
        
        

