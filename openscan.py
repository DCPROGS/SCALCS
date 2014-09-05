from dcpyps import dataset

# LOAD DATA.
scnfiles = [["./dcpyps/samples/scn/001004S2.SCN"]]
#"./dcpyps/samples/scn/CH82.scn"
tres = [0.000025]
tcrit = [0.002]
conc = [50e-6]
chs = [True]

for i in range(len(scnfiles)):
    rec = dataset.SCRecord(scnfiles[i], conc[i], tres[i], tcrit[i])
    rec.printout()
    print("\nPrinting bursts\n")
    count = 0
    for burst in rec.bursts.all():
        print count
        print burst
        count += 1
