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
         dcpyps.Rate(2 * 2000.0, A2R, AR, name='2k(-2)', limits=[1e-15,1e+7]),
         dcpyps.Rate(2 * 5.0e07, R, AR, name='2k(+1)', eff='c', limits=[1e-15,1e+10]),
         dcpyps.Rate(5.0e08, ARS, A2RS, name='k*(+2)', eff='c', fixed=True, limits=[1e-15,1e+10]),
         dcpyps.Rate(5.0e08, AR, A2R, name='k(+2)', eff='c', limits=[1e-15,1e+10]),
         #dcpyps.Rate(2 * 1.0 / 3.0, A2RS, ARS, name='k*(-2)', limits=[1e-15,1e+7])
         dcpyps.Rate(0.66667, A2RS, ARS, name='2k*(-2)', mr=True, limits=[1e-15,1e+7])
         ]


    CycleList = [dcpyps.Cycle(['A2R*', 'AR*', 'AR', 'A2R'], ['A2R*', 'AR*'])]

    fastblk = False
    KBlk = 0.001

    return  dcpyps.Mechanism(RateList, CycleList) #, fastblk, KBlk)

def CCO():

    ARS  = dcpyps.State('A', 'AR*', 50e-12)
    AR   = dcpyps.State('B', 'AR', 0.0)
    R    = dcpyps.State('C', 'R', 0.0)

    RateList = [
         dcpyps.Rate(15000.0, AR, ARS, name='beta', limits=[1e-15,1e+7]),
         dcpyps.Rate(500.0, ARS, AR, name='alpha', limits=[1e-15,1e+7]),
         dcpyps.Rate(2000.0, AR, R, name='koff', limits=[1e-15,1e+7]),
         dcpyps.Rate(5.0e08, R, AR, name='kon', eff='c', limits=[1e-15,1e+10]),
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

def six_cycles_mec():

    A = dcpyps.State('A', 'A', 60e-12)
    B = dcpyps.State('A', 'B', 60e-12)
    C = dcpyps.State('A', 'C', 60e-12)
    D = dcpyps.State('B', 'D', 60e-12)
    E = dcpyps.State('B', 'E', 0.0)
    F = dcpyps.State('B', 'F', 0.0)
    G = dcpyps.State('B', 'G', 0.0)
    H = dcpyps.State('B', 'H', 0.0)
    I = dcpyps.State('B', 'I', 0.0)
    J = dcpyps.State('B', 'J', 0.0)
    K = dcpyps.State('B', 'K', 0.0)
    L = dcpyps.State('C', 'L', 0.0)

    RateList = [
         dcpyps.Rate(15.0, A, B, name='ab'),
         dcpyps.Rate(25.0, B, A, name='ba'),
         dcpyps.Rate(15.0, A, E, name='ae'),
         dcpyps.Rate(25.0, E, A, name='ea'),
         dcpyps.Rate(15.0, B, C, name='bc'),
         dcpyps.Rate(25.0, C, B, name='cb'),
         dcpyps.Rate(15.0, B, F, name='bf'),
         dcpyps.Rate(25.0, F, B, name='fb'),
         dcpyps.Rate(15.0, C, D, name='cd'),
         dcpyps.Rate(25.0, D, C, name='dc'),
         dcpyps.Rate(15.0, C, G, name='cg'),
         dcpyps.Rate(25.0, G, C, name='gc'),
         dcpyps.Rate(15.0, D, H, name='dh'),
         dcpyps.Rate(25.0, H, D, name='hd'),
         dcpyps.Rate(15.0, E, F, name='ef'),
         dcpyps.Rate(25.0, F, E, name='fe'),
         dcpyps.Rate(15.0, E, I, name='ei'),
         dcpyps.Rate(25.0, I, E, name='ie'),
         dcpyps.Rate(15.0, F, G, name='fg'),
         dcpyps.Rate(25.0, G, F, name='gf'),
         dcpyps.Rate(15.0, F, J, name='fj'),
         dcpyps.Rate(25.0, J, F, name='jf'),
         dcpyps.Rate(15.0, G, H, name='gh'),
         dcpyps.Rate(25.0, H, G, name='hg'),
         dcpyps.Rate(15.0, G, K, name='gk'),
         dcpyps.Rate(25.0, K, G, name='kg'),
         dcpyps.Rate(15.0, H, L, name='hl'),
         dcpyps.Rate(25.0, L, H, name='lh'),
         dcpyps.Rate(15.0, I, J, name='ij'),
         dcpyps.Rate(25.0, J, I, name='ji'),
         dcpyps.Rate(15.0, J, K, name='jk'),
         dcpyps.Rate(25.0, K, J, name='kj'),
         dcpyps.Rate(15.0, K, L, name='kl'),
         dcpyps.Rate(25.0, L, K, name='lk')
        ]

    return  dcpyps.Mechanism(RateList)
