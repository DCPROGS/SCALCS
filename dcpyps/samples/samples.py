import dcpyps

def CH82():

    RateList = [
         dcpyps.Rate(15.0, 4, 1, name='beta1'),
         dcpyps.Rate(15000.0, 3, 2, name='beta2'),
         dcpyps.Rate(3000.0, 1, 4, name='alpha1'),
         dcpyps.Rate(500.0, 2, 3, name='alpha2'),
         dcpyps.Rate(2000.0, 4, 5, name='k(-1)'),
         dcpyps.Rate(2 * 2000.0, 3, 4, name='k(-2)'),
         dcpyps.Rate(2 * 5.0e07, 5, 4, name='k(+1)', eff='c'),
         dcpyps.Rate(5.0e08, 1, 2, name='k*(+1)', eff='c'),
         dcpyps.Rate(5.0e08, 4, 3, name='k(-2)', eff='c'),
         dcpyps.Rate(2 * 1.0 / 3.0, 2, 1, name='k*(-2)'),
         ]

    StateList = [
        dcpyps.State(1, 'A', 'A2R*', 60e-12),
        dcpyps.State(2, 'A', 'AR*', 60e-12),
        dcpyps.State(3, 'B', 'A2R', 0.0),
        dcpyps.State(4, 'B', 'AR', 0.0),
        dcpyps.State(5, 'C', 'R', 0.0)
        ]

    ncyc = 1
    fastblk = True
    KBlk = 0.001

    return  dcpyps.Mechanism(RateList, StateList, ncyc, fastblk, KBlk)
