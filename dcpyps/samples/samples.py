import dcpyps

def CH82():

    A2RS = dcpyps.State('A', 'A2R*', 60e-12)
    ARS  = dcpyps.State('A', 'AR*', 60e-12)
    A2R  = dcpyps.State('B', 'A2R', 0.0)
    AR   = dcpyps.State('B', 'AR', 0.0)
    R    = dcpyps.State('C', 'R', 0.0)

    RateList = [
         dcpyps.Rate(15.0, AR, A2RS, name='beta1'),
         dcpyps.Rate(15000.0, A2R, ARS, name='beta2'),
         dcpyps.Rate(3000.0, A2RS, AR, name='alpha1'),
         dcpyps.Rate(500.0, ARS, A2R, name='alpha2'),
         dcpyps.Rate(2000.0, AR, R, name='k(-1)'),
         dcpyps.Rate(2 * 2000.0, A2R, AR, name='k(-2)'),
         dcpyps.Rate(2 * 5.0e07, R, AR, name='k(+1)', eff='c'),
         dcpyps.Rate(5.0e08, A2RS, ARS, name='k*(+1)', eff='c'),
         dcpyps.Rate(5.0e08, AR, A2R, name='k(-2)', eff='c'),
         dcpyps.Rate(2 * 1.0 / 3.0, ARS, A2RS, name='k*(-2)'),
         ]

    ncyc = 1
    fastblk = False
    KBlk = 0.001

    return  dcpyps.Mechanism(RateList, ncyc) #, fastblk, KBlk)
