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


tint, ampl, opts = [], [], []
rtint, rampl, ropts = [], [], []
ptint, pampl, popts = [], [], []

fn1 = "./BURZORES.txt"
f = open(fn1, 'r')
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

fn2 = "./hjcfit-bursts.txt"
f = open(fn2, 'r')
for line in f.readlines():
    if "open period:"in line:
        spline = line.split()
        ptint.append(float(spline[3])/1000.0)
        pampl.append(float(spline[4]))
        popts.append(int(spline[5]))
    if "shut time:"in line:
        spline = line.split()
        ptint.append(float(spline[3])/1000.0)
        pampl.append(float(spline[4]))
        popts.append(int(spline[5]))
f.close() 

scnfiles = [["../samples/A-10.scn"]]
tres = [0.000030]
tcrit = [0.004]
chs = [True]
conc = [10e-6]
rec = dataset.SCRecord(scnfiles[0], conc[0], tres[0], tcrit[0], chs[0])

print "\nLength of lists:"
print "\tHJCFIT\tDCPYPS"
print "all\t{0:d}\t{1:d}".format(len(tint), len(rec.itint))
print tint[-2:], ampl[-2:]
print rec.itint[-2:], rec.iampl[-2:]
print "res\t{0:d}\t{1:d}".format(len(rtint), len(rec.rtint))
print rtint[-2:], rampl[-2:]
print rec.rtint[-2:], rec.rampl[-2:]
print "per\t{0:d}\t{1:d}".format(len(ptint), len(rec.pint))

#compare_lists(tint, ampl, rec.itint, rec.iampl)
#compare_lists(rtint, rampl, rec.rtint, rec.rampl)
compare_lists(ptint, pampl, rec.pint, rec.pamp)

print "HJCFIT:", rtint[5132:5152], "\n", rampl[5132:5152], "\n", ropts[5132:5152]
print "DCPYPS:", rec.rtint[5132:5152], "\n", rec.rampl[5132:5152], "\n", rec.rprops[5132:5152]

print "HJCFIT:", tint[5553:5583], "\n", ampl[5553:5583], "\n", opts[5553:5583]
print "DCPYPS:", rec.itint[5553:5583], "\n", rec.iampl[5553:5583], "\n", rec.iprops[5553:5583]