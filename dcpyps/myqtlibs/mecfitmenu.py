import time
import math
import numpy as np
from scipy.optimize import minimize
from pylab import*
try:
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")

from dcpyps import dcio
from dcpyps import mechanism
from dcpyps import dataset
import myqtcommon
from dcprogs.likelihood import Log10Likelihood

class MecFitMenu(QMenu):
    """
    """
    def __init__(self, parent):
        super(MecFitMenu, self).__init__(parent) 
        self.parent = parent
        self.setTitle('&Mec Fit')
        
        
        self.iternum = 0
        self.likelihood = []
        self.theta = None
        
        loadDemo1Action = myqtcommon.createAction(self,
            "&Load demo fit: Burzomato 2004", self.onLoadDemo_Burzomato,
            None, "loaddemoburz", "Load Demo Burzomato")
        fitRunAction = myqtcommon.createAction(self,
            "&Run Fit", self.onStartFitting,
            None, "startfit", "Start Fitting")
        fitSettingsAction = myqtcommon.createAction(self, "&Fit Settings", self.onFittingSettings,
            None, "setfit", "Fitting settings")
        self.addActions([
#            loadDemo1Action, 
            fitRunAction, fitSettingsAction])    
        
    def onLoadDemo_Burzomato(self):
        """
        Load demo fit - GlyR set BGVH from Burzomato 2004.
        """

        # LOAD FLIP MECHANISM USED Burzomato et al 2004
        self.mecfn = "./dcpyps/samples/demomec.mec"
        version, meclist, max_mecnum = dcio.mec_get_list(self.mecfn)
        self.mec = dcio.mec_load(self.mecfn, meclist[2][0])
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
        self.mec.update_constrains()
        self.mec.set_mr(True, 7, 0)
        self.mec.set_mr(True, 15, 1)
        self.mec.printout(self.log)
        
        # LOAD DATA.
        self.scnfiles = [["./dcpyps/samples/glydemo/A-10.scn"], 
            ["./dcpyps/samples/glydemo/B-30.scn"],
            ["./dcpyps/samples/glydemo/C-100.scn"],
            ["./dcpyps/samples/glydemo/D-1000.scn"]]
        self.tres = [0.000030, 0.000030, 0.000030, 0.000030]
        self.tcrit = [0.004, -1, -0.06, -0.02]
        self.conc = [10e-6, 30e-6, 100e-6, 1000e-6]
        self.chs = [True, False, False, False]
          
        for i in range(len(self.scnfiles)):
            rec = dataset.SCRecord(self.scnfiles[i], self.conc[i], self.tres[i],
                self.tcrit[i], self.chs[i])
            rec.record_type = 'recorded'
            self.recs.append(rec)
            self.bursts.append(rec.bursts.intervals())
            rec.printout(self.log)
        
#################### Called by menu 'Run Fit'
    def onStartFitting(self):

        theta = np.log(self.parent.mec.theta())
        kwargs = {'nmax': 2, 'xtol': 1e-12, 'rtol': 1e-12, 'itermax': 100,
            'lower_bound': -1e6, 'upper_bound': 0}

        for i in range(len(self.parent.recs)):
            self.likelihood.append(Log10Likelihood(self.parent.bursts[i], self.parent.mec.kA,
                self.parent.recs[i].tres, self.parent.recs[i].tcrit, **kwargs))
                
        lik = self.dcprogslik(theta)
        self.parent.textBox.append("\nStarting likelihood (DCprogs)= {0:.6f}".
            format(-lik))

        start = time.clock()
        success = False
        result = None
        while not success:
            #res = minimize(dcprogslik, np.log(theta), method='Powell', callback=printit, options={'maxiter': 5000, 'disp': True})
            result = minimize(self.dcprogslik, theta, 
                method='Nelder-Mead', callback=self.printiter,
                options={'xtol':1e-4, 'ftol':1e-4, 'maxiter': 5000, 'maxfev': 10000,
                'disp': True})
            if result.success:
                success = True
            else:
                theta = result.x
        
        self.parent.mec.theta_unsqueeze(np.exp(result.x))

        str = ("\nDCPROGS Fitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
                %time.localtime()[0:6] +
            'time in simplex= {0:.6f}'.format(time.clock() - start) +
            '\n Final log-likelihood = {0:.6f}'.format(-result.fun) +
            '\n Number of iterations = {0:d}'.format(result.nit) +
            '\n Number of evaluations = {0:d}'.format(result.nfev) +
            "\n Final rate constants:")
        self.parent.log.write(str)
        self.parent.mec.printout(self.parent.log)
        self.parent.log.write('\n\n')

        
#        open_pdfs, shut_pdfs = [], []
#        for i in range(len(self.scnfiles)):
#            open_pdfs.append(save_pdf_fig(self.scnfiles[i], self.recs[i].opint,
#                self.mec, self.conc[i], self.tres[i], 'open'))
#            shut_pdfs.append(save_pdf_fig(self.scnfiles[i], self.recs[i].shint,
#                self.mec, self.conc[i], self.tres[i], 'shut'))
#        resulthtml = save_html(self.str2, self.str3, self.mecfn, self.scnfiles, open_pdfs, shut_pdfs)
#        self.textBox.append('\n\n Results saved in Loading '+resulthtml)

    def onFittingSettings(self):
        pass

    def dcprogslik(self, x, args=None):
        self.parent.mec.theta_unsqueeze(np.exp(x))
        lik = 0
        for i in range(len(self.parent.recs)):
            self.parent.mec.set_eff('c', self.parent.recs[i].conc)
            lik += -self.likelihood[i](self.parent.mec.Q) * math.log(10)
        return lik

    def printiter(self, theta):
        self.iternum += 1
        lik = self.dcprogslik(theta)
        print("iteration # {0:d}; log-lik = {1:.6f}".format(self.iternum, -lik))
        print(np.array_str(np.exp(theta)))        

    
            
