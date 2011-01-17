from dcpyps import mechanism as mec

def CH82():

    RateList = [
         mec.Rate(15.0, 4, 1, name='beta1'),
         mec.Rate(15000.0, 3, 2, name='beta2'),
         mec.Rate(3000.0, 1, 4, name='alpha1'),
         mec.Rate(500.0, 2, 3, name='alpha2'),
         mec.Rate(2000.0, 4, 5, name='k(-1)'),
         mec.Rate(2 * 2000.0, 3, 4, name='k(-2)'),
         mec.Rate(2 * 5.0e07, 5, 4, name='k(+1)', eff='c'),
         mec.Rate(5.0e08, 1, 2, name='k*(+1)', eff='c'),
         mec.Rate(5.0e08, 4, 3, name='k(-2)', eff='c'),
         mec.Rate(2 * 1.0 / 3.0, 2, 1, name='k*(-2)'),
         ]

    StateList = [
        mec.State(1, 'A', 'A2R*', 60e-12),
        mec.State(2, 'A', 'AR*', 60e-12),
        mec.State(3, 'B', 'A2R', 0.0),
        mec.State(4, 'B', 'AR', 0.0),
        mec.State(5, 'C', 'R', 0.0)
        ]

    ncyc = 1
    fastblk = True
    KBlk = 0.001

    return  mec.Mechanism(RateList, StateList, ncyc, fastblk, KBlk)
