import sys
from dcpyps import samples
from dcpyps import qmatlib as qml

if __name__ == "__main__":

    mec1 = samples.CH82()
    sys.stdout.write('%s' % mec1)
    mec1.Rates[0].rateconstants = 1500
    # Check if rates were changed inside mec
    sys.stdout.write('\n\nChanged mec:')
    sys.stdout.write('%s' % mec1)

    # Check if Q matrix was updated
    conc = 100e-9    # 100 nM
    mec = samples.CH82()
    mec.set_eff('c', conc)
    phiA = qml.phiA(mec)
    mec1.set_eff('c', conc)
    phiA1 = qml.phiA(mec1)

    print('\n\n Compare initial vectors for original and modified mecs:')
    print('\noriginal phiA=', phiA)
    print('\nmodified phiA=', phiA1)
    if (phiA == phiA1).all():
        print('\nQ matrix was NOT modified.')
    else:
        print('\nQ matrix was modified.')
