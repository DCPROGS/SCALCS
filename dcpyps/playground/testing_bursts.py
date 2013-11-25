import math
from dcpyps import dataset
from dcpyps import samples

def compare_lists(hjcfit, hjcfita, dcpyps, dcpypsa):
    founddiff = False
    count = 0
    while not founddiff and (count < len(hjcfit)):
#    while count < len(hjcfit):
        if math.fabs(hjcfit[count] / dcpyps[count])-1 > 1.1:
            print "\n"
            print hjcfit[count] / dcpyps[count]
            print "interval # {0:d} is different".format(count+1)
            print "HJCFIT:", hjcfit[count-2:count+3], hjcfita[count-1:count+5]
            print "DCPYPS:", dcpyps[count-2:count+3], dcpypsa[count-1:count+5]
            founddiff = True
            
        count += 1
        
    if not founddiff:
        print "two lists are similar"
        
def get_all_and_resolved_intervals(fres):
    """
    """
    tint, ampl, opts = [], [], []
    rtint, rampl, ropts = [], [], []

    f = open(fres, 'r')
    for line in f.readlines():
        if "out =" in line:
            spline = line.split()
            try:
                #print spline
                rtint.append(float(spline[3])/1000.0)
                rampl.append(float(spline[4]))
                ropts.append(int(spline[5]))
            except:
                rtint.append(float(spline[3][:11])/1000.0)
                rampl.append(float(spline[3][11:]))
                ropts.append(int(spline[4]))
        else:
            spline = line.split()
            if int(spline[0]) > len(tint):
                try:
                    tint.append(float(spline[1])/1000.0)
                    ampl.append(float(spline[2]))
                    opts.append(int(spline[3]))
                except:
                    tint.append(float(spline[1][:11])/1000.0)
                    ampl.append(float(spline[1][11:]))
                    opts.append(int(spline[2]))
    f.close()
    return tint, ampl, opts, rtint, rampl, ropts

def get_periods(fper):
    ptint, pampl, popts = [], [], []
    f = open(fper, 'r')
    for line in f.readlines():
        if "open period:" in line:
            spline = line.split()
            ptint.append(float(spline[3])/1000.0)
            pampl.append(float(spline[4]))
            popts.append(int(spline[5]))
        if "shut time:" in line:
            spline = line.split()
            ptint.append(float(spline[3])/1000.0)
            pampl.append(float(spline[4]))
            popts.append(int(spline[5]))
    f.close() 
    return ptint, pampl, popts


# Burzomato A-10.scn
scn = [["../samples/glydemo/A-10.scn"], 10e-6, 30e-6, 4e-3, True]
fnames = ["../samples/Burzomato/Ares.txt", "../samples/Burzomato/Aper.txt"]

# CH82 simulated data CH82.scn
#scn = [["../samples/CH82/CH82.scn"], 100e-9, 100e-6, 4e-3, True]
#fnames = ["../samples/CH82/CH82br100res.txt", "../samples/CH82/CH82br100per.txt"]

# patch K
#scn = [["../samples/K.scn"], 200e-6, 20e-6, 1000e-3, False]
#fnames = ["../samples/Kres.txt", "../samples/Kper.txt"]


# scnfile, conc, tres, tcrit, CHS
rec = dataset.SCRecord(scn[0], scn[1], scn[2], scn[3], scn[4])
# intervals from text files
tint, ampl, opts, rtint, rampl, ropts = get_all_and_resolved_intervals(fnames[0])
ptint, pampl, popts = get_periods(fnames[1])


print "\nLength of lists:"
print "\tHJCFIT\tDCPYPS"

print "\nall\t{0:d}\t{1:d}".format(len(tint), len(rec.itint))
print "HJCFIT: last five:", tint[-5:], ampl[-5:]
print "DCPYPS: last five:", rec.itint[-5:], rec.iampl[-5:]
print "HJCFIT: first five:", tint[:5], ampl[:5]
print "DCPYPS: first five:", rec.itint[:5], rec.iampl[:5]

print "\nres\t{0:d}\t{1:d}".format(len(rtint), len(rec.rint))
print "HJCFIT: first five:", rtint[:5], rampl[:5]
print "DCPYPS: first five:", rec.rint[:5], rec.ramp[:5]
print "HJCFIT: last five:", rtint[-5:], rampl[-5:]
print "DCPYPS: last five:", rec.rint[-5:], rec.ramp[-5:]

print "\nper\t{0:d}\t{1:d}".format(len(ptint), len(rec.pint))
print "HJCFIT: first five:", ptint[:5], pampl[:5]
print "DCPYPS: first five:", rec.pint[:5], rec.pamp[:5]
print "HJCFIT: last five:", ptint[-5:], pampl[-5:]
print "DCPYPS: last five:", rec.pint[-5:], rec.pamp[-5:]

#compare_lists(tint, ampl, rec.itint, rec.iampl)
#compare_lists(rtint, rampl, rec.rint, rec.ramp)
compare_lists(ptint, pampl, rec.pint, rec.pamp)

#rec.print_resolved_intervals()
#rec.print_resolved_periods()
rec.print_bursts()