import time, math, sys, socket, os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import itemfreq
from pylab import*

from dcpyps import dcio
from dcpyps import dataset
from dcpyps import mechanism
from dcpyps import scplotlib as scpl
from dcpyps import scalcslib as scl

from dcprogs.likelihood import Log10Likelihood

str1 = "DC_PyPs: Q matrix calculations.\n"
str2 = ("Date and time of analysis: %4d/%02d/%02d %02d:%02d:%02d\n"
    %time.localtime()[0:6])
machine = socket.gethostname()
system = sys.platform
str3 = "Machine: %s; System: %s\n" %(machine, system)


# LOAD DATA.
scnfiles = [["../samples/scn/G.SCN"], ["../samples/scn/P.SCN"],
    ["../samples/scn/O.SCN"], ["../samples/scn/N.SCN"]]
tres = [0.000040, 0.000035, 0.000035, 0.000030]
tcrit = [0.01, 0.01, -0.1, -0.1]
conc = [500e-6, 1e-3, 5e-3, 10e-3]
chs = [True, True, False, False]

recs = []
bursts = []
for i in range(len(scnfiles)):
    rec = dataset.SCRecord(scnfiles[i], conc[i], tres[i], tcrit[i], chs[i])
    rec.record_type = 'recorded'
    recs.append(rec)
    bursts.append(rec.bursts)
    rec.printout()


# LOAD FLIP MECHANISM USED Burzomato et al 2004
mecfn = "../samples/mec/ke10.mec"
version, meclist, max_mecnum = dcio.mec_get_list(mecfn)
mec = dcio.mec_load(mecfn, meclist[10][0])
#mec.printout(sys.stdout)

# PREPARE RATE CONSTANTS.
#rates = [5000.0, 500.0, 2700.0, 2000.0, 800.0, 15000.0, 300.0, 0.1200E+06, 6000.0, 0.4500E+09, 1500.0, 12000.0, 4000.0, 0.9000E+09, 7500.0, 1200.0, 3000.0, 0.4500E+07, 2000.0, 0.9000E+07, 1000, 0.135000E+08]
#mec.set_rateconstants(rates)

# Fixed rates.
#fixed = np.array([False, False, False, False, False, False, False, True,
#    False, False, False, False, False, False])
#if fixed.size == len(mec.Rates):
for i in range(len(mec.Rates)):
    mec.Rates[i].fixed = False

# Constrained rates.
mec.Rates[14].is_constrained = True
mec.Rates[14].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[14].constrain_args = [18, 3]
mec.Rates[16].is_constrained = True
mec.Rates[16].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[16].constrain_args = [18, 2]
mec.Rates[17].is_constrained = True
mec.Rates[17].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[17].constrain_args = [15, 2]
mec.Rates[19].is_constrained = True
mec.Rates[19].constrain_func = mechanism.constrain_rate_multiple
mec.Rates[19].constrain_args = [15, 3]

mec.set_mr(True, 13)
mec.update_constrains()
mec.update_mr()

mec.printout(sys.stdout)
theta = np.log(mec.theta())

kwargs = {'nmax': 2, 'xtol': 1e-12, 'rtol': 1e-12, 'itermax': 100,
    'lower_bound': -1e6, 'upper_bound': 0}
likelihood = []
for i in range(len(conc)):
    likelihood.append(Log10Likelihood(bursts[i], mec.kA, tres[i], tcrit[i], **kwargs))

def dcprogslik(x, args=None):
    mec.theta_unsqueeze(np.exp(x))
    lik = 0
    if mec.check_limits():
        for i in range(len(conc)):
            mec.set_eff('c', conc[i])
            lik += -likelihood[i](mec.Q) * math.log(10)
    else:
        lik = 1e7
    return lik

def dcprogslikall(x, args=None):
    mec.theta_unsqueeze(np.exp(x))
    lik = []
    for i in range(len(conc)):
        mec.set_eff('c', conc[i])
        lik.append(-likelihood[i](mec.Q) * math.log(10))
    return lik

iternum = 0
def printiter(theta):
    global iternum
    iternum += 1
    lik = dcprogslik(theta)
#    likall = dcprogslikall(theta)
    print("iteration # {0:d}; log-lik = {1:.6f}; ".format(iternum, -lik))
#    for i in range(len(conc)):
#        print likall[i]
    print(np.exp(theta))

#lik = dcprogslik(theta)
mec.set_eff('c', conc[0])
lik = -likelihood[0](mec.Q) * math.log(10)
print ("\nStarting likelihood (DCprogs)= {0:.6f}".format(-lik))

opts = {}
opts['mec'] = mec
opts['conc'] = conc[0]
opts['tres'] = tres[0]
opts['tcrit'] = tcrit[0]
opts['isCHS'] = True
opts['data'] = bursts[0]
start_lik, th = scl.HJClik(theta, opts)
print ("\nStarting likelihood (dc-pyps)= {0:.6f}".format(-start_lik))

start = time.clock()

success = False
result = None
while not success:
    #result = minimize(dcprogslik, np.log(theta), method='Powell', callback=printiter, options={'maxiter': 5000, 'disp': True})
    result = minimize(dcprogslik, theta, method='Nelder-Mead', callback=printiter,
        options={'xtol':1e-3, 'ftol':1e-3, 'maxiter': 5000, 'maxfev': 10000,
        'disp': True})
    print 'result=', result
    if result.success:
        success = True
    else:
        theta = result.x


print ("\nDCPROGS Fitting finished: %4d/%02d/%02d %02d:%02d:%02d\n"
        %time.localtime()[0:6])
print 'time in simplex=', time.clock() - start
print '\n\nresult='
print result

print ('\n Final log-likelihood = {0:.6f}'.format(-result.fun))
print ('\n Number of iterations = {0:d}'.format(result.nit))
print ('\n Number of evaluations = {0:d}'.format(result.nfev))
mec.theta_unsqueeze(np.exp(result.x))
print "\n Final rate constants:"
mec.printout(sys.stdout)
print '\n\n'


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

#open_pdfs = []
#shut_pdfs = []
#for i in range(len(scnfiles)):
#    open_pdfs.append(save_pdf_fig(scnfiles[i], recs[i].opint, mec, conc[i], tres[i], 'open'))
#    shut_pdfs.append(save_pdf_fig(scnfiles[i], recs[i].shint, mec, conc[i], tres[i], 'shut'))
#
#htmlstr = ("""<html>
#    <p>HJCFIT: Fit of model to open-shut times with missed events
#    (Uses HJC distributions, exact for first two deadtimes then asymptotic, to calculate likelihood of record)</p>""" +
#    '<p>'+ str2 + '<p></p>' + str3 + """</p></html>""" )
#
#htmlfile = mecfn[:-4] + '.html'
#f = open(htmlfile, 'w')
#f.write(htmlstr)
#for i in range(len(scnfiles)):
#    f.write("<p>{0} _____________________________________ {1}</p>".format(open_pdfs[i], shut_pdfs[i]))
#    f.write("<img src={0} width='450' height='300'><img src={1} width='450' height='300'>".format(os.path.abspath(open_pdfs[i]), os.path.abspath(shut_pdfs[i])))
#
##mec.printout(f)
#f.close()


