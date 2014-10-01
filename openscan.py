import math
from dcpyps import dataset

def compare_lists(hjcfit, hjcfita, hjopts, dcpyps, dcpypsa, dcopts):
    founddiff = False
    count = 0
    while not founddiff and (count < len(hjcfit)):
#    while count < len(hjcfit):
        print count+1, "HJCFIT:", hjcfit[count], hjopts[count], "DCPYPS:", dcpyps[count], dcopts[count]
        if ((math.fabs(hjcfit[count] / dcpyps[count])-1 > 1.00001) or (math.fabs(dcpyps[count] / hjcfit[count])-1 > 1.00001)):
            #print "\n"
            #print hjcfit[count] / dcpyps[count]
            print "interval # {0:d} is different".format(count+1)
            print "HJCFIT:", hjcfit[count-1:count+10], hjcfita[count-1:count+10]
            print "DCPYPS:", dcpyps[count-1:count+10], dcpypsa[count-1:count+10]
            founddiff = True
        count += 1

    if not founddiff:
        print "two lists are similar"

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

def get_all_and_resolved_intervals(fres):
    """
    """
    tint, ampl, opts = [], [], []
    rtint, rampl, ropts = [], [], []

    f = open(fres, 'r')
    for line in f.readlines():
        spline = line.split()
        #print spline
        if "out =" in line:

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

# LOAD DATA. AChR H2003
scnfiles = [["./dcpyps/samples/scn/001004S2.SCN"]]
perfile = "./dcpyps/samples/scn/hatton1.txt"
#resfile = "./dcpyps/samples/scn/resolved.txt"
tres = [0.000025]
tcrit = [0.002]
conc = [50e-9]
badopen = [0.02]

# LOAD DATA. GlyR B3004 sim
#scnfiles = [["./dcpyps/samples/scn/simC.scn"]]
#perfile = "./dcpyps/samples/scn/simC_per.txt"
#resfile = "./dcpyps/samples/scn/simC_res.txt"
#tres = [0.000030]
#tcrit = [0.06]
#conc = [100e-6]

for i in range(len(scnfiles)):
    rec = dataset.SCRecord(scnfiles[i], conc[i], tres[i], tcrit[i], badopen[i])
    rec.printout()

#    tint, ampl, opts, rtint, rampl, ropts = get_all_and_resolved_intervals(resfile)
    ptint, pampl, popts = get_periods(perfile)
#    compare_lists(rtint, rampl, ropts, rec.rint, rec.ramp, rec.ropt) # compare resolved intervals
    compare_lists(ptint, pampl, popts, rec.pint, rec.pamp, rec.popt) # compare periods

#    # PRINT BURSTS
#    print("\nPrinting bursts\n")
#    count = 0
#    for burst in rec.bursts.all():
#        print count
#        print burst
#        count += 1
