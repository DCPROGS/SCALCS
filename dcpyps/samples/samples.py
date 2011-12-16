import dcpyps

def CH82():

    A2RS = dcpyps.State('A', 'A2R*', 60e-12)
    ARS  = dcpyps.State('A', 'AR*', 60e-12)
    A2R  = dcpyps.State('B', 'A2R', 0.0)
    AR   = dcpyps.State('B', 'AR', 0.0)
    R    = dcpyps.State('C', 'R', 0.0)

    RateList = [
         dcpyps.Rate(15.0, AR, ARS, name='beta1', limits=[1e-15,1e+7]),
         dcpyps.Rate(15000.0, A2R, A2RS, name='beta2', limits=[1e-15,1e+7]),
         dcpyps.Rate(3000.0, ARS, AR, name='alpha1', limits=[1e-15,1e+7]),
         dcpyps.Rate(500.0, A2RS, A2R, name='alpha2', limits=[1e-15,1e+7]),
         dcpyps.Rate(2000.0, AR, R, name='k(-1)', limits=[1e-15,1e+7]),
         dcpyps.Rate(2 * 2000.0, A2R, AR, name='k(-2)', limits=[1e-15,1e+7]),
         dcpyps.Rate(2 * 5.0e07, R, AR, name='k(+1)', eff='c', limits=[1e-15,1e+10]),
         dcpyps.Rate(5.0e08, ARS, A2RS, name='k*(+2)', eff='c', limits=[1e-15,1e+10]),
         dcpyps.Rate(5.0e08, AR, A2R, name='k(+2)', eff='c', limits=[1e-15,1e+10]),
         #dcpyps.Rate(2 * 1.0 / 3.0, A2RS, ARS, name='k*(-2)', limits=[1e-15,1e+7])
         dcpyps.Rate(0.66667, A2RS, ARS, name='k*(-2)', limits=[1e-15,1e+7])
         ]

    ncyc = 1
    fastblk = False
    KBlk = 0.001

    return  dcpyps.Mechanism(RateList, ncyc) #, fastblk, KBlk)

def CCO():

    ARS  = dcpyps.State('A', 'AR*', 50e-12)
    AR   = dcpyps.State('B', 'AR', 0.0)
    R    = dcpyps.State('C', 'R', 0.0)

    RateList = [
         dcpyps.Rate(20.0, AR, ARS, name='beta', limits=[1e-15,1e+7]),
         dcpyps.Rate(50.0, ARS, AR, name='alpha', limits=[1e-15,1e+7]),
         dcpyps.Rate(5.0, AR, R, name='koff', limits=[1e-15,1e+7]),
         dcpyps.Rate(1.0e06, R, AR, name='kon', eff='c', limits=[1e-15,1e+10]),
         ]

    return  dcpyps.Mechanism(RateList)

def CO():

    RS  = dcpyps.State('A', 'O', 50e-12)
    R   = dcpyps.State('B', 'C', 0.0)

    RateList = [
         dcpyps.Rate(20.0, R, RS, name='beta', limits=[1e-15,1e+7]),
         dcpyps.Rate(50.0, RS, R, name='alpha', limits=[1e-15,1e+7]),
         ]

    return  dcpyps.Mechanism(RateList)
