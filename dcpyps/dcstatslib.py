"""
A collection of functions for Q matrix manipulations.
"""

import sys
import math
import random

import numpy as np
from scipy.special import betainc
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel

intro_fieller = """FIELLER: calculates confidence limits for a ratio \
according Fieller''s theorem.
\nCalculates approximate SD of the ratio r=a/b, given \
the SD of a (numerator) and of b (denominator),
and the correlation coefficient between a, b (zero if they are\
independent). \n"""

intro_randomisation = '\n RANTEST: performs a randomisation test to compare two \
independent samples. According to the null hypothesis\n \
of no-difference, each outcome would have been the same \
regardless of which group the individual happened to\n \
be allocated. Therefore all N=n1+n2 observations are \
pooled and, as in the actual experiment, divided at random\n \
into groups of size n1 and n2. The fraction \
of randomisations that gives rise to a difference between the groups\n \
at least as large as that observed \
gives the P value.\
\n In the binomial case, in which the measurement is the \
fraction of ''successes'' in each sample (say r1 out of n1, and\n \
r2 out of n2) a ''success'' is given a \
score of 1, ''failure'' = 0.\n'

def fieller(a, b, sa, sb, r, tval):
    'Fieller formula calculator.'

    va = sa * sa
    vb = sb * sb
    cov = r * sa * sb
    g = tval * tval * vb /(b * b)
    ratio = a / b
    rat2 = ratio * ratio
    # Write disc in a way that does not appear to divide by vb
    # (which actually cancels) so OK to use vb=0
    # disc=va - 2.0*ratio*cov + rat2*vb - g*(va-cov*cov/vb)
    disc = va - 2.0 * ratio * cov + rat2 * vb - g * (va - r * r * va)

    if disc >= 0:
        d = (tval / b) * math.sqrt(disc)
        # Write pre in a way that does not appear to divide by vb
        # (which actually cancels) so OK to use vb=0 (use g=tval*tval*vb/(b*b))
        #	pre=ratio - g*cov/vb
        pre = ratio - (tval * tval * r * sa * sb) / (b * b)
        f = 1.0 / (1.0 - g)
        clower = f * (pre - d)
        cupper = f * (pre + d)
        dlow = clower - ratio
        dhi = cupper - ratio
        # Approximation for small g
        appsd = math.sqrt(va + rat2 * vb - 2.0 * ratio * cov) / b
        applo = ratio - tval * appsd
        apphi = ratio + tval * appsd
        cvr = 100.0 * appsd / ratio

    return clower, cupper, dlow, dhi, appsd, cvr, applo, apphi

def fieller_printout(a, b, sa, sb, r, tval, output=sys.stdout):
    """
    """
    clower, cupper, dlow, dhi, appsd, cvr, applo, apphi = fieller(a,
        b, sa, sb, r, tval)
    'Display Fieller calculation results on main frame.'
    output.write('\n Ratio (= a/b) = {0:.6f}'.format(a/b) +
        '\n Confidence limits: lower {0:.6f}, upper {1:.6f}'.
        format(clower, cupper) +
        '\n i.e deviations: lower {0:.6f}, upper {1:.6f}'.
        format(dlow, dhi) +
        '\n Approximate SD of ratio = {0:.6f}'.format(appsd) +
        '\n Approximate CV of ratio = {0:.6f}'.format(cvr) +
        '\n Approximate limits: lower {0:.6f}, upper {1:.6f}'.
        format(applo, apphi))

def t_test_binomial(n1, n2, ir1, ir2):
    """
    """
    #Use Gaussian approx to do 2 sample t test
    p1 = float(ir1) / float(n1)
    p2 = float(ir2) / float(n2)
    ppool = float(ir1 + ir2) / float(n1 + n2)
    sd1 = math.sqrt(p1 * (1.0 - p1) / float(n1))
    sd2 = math.sqrt(p2 * (1.0 - p2) / float(n2))
    sd1p = math.sqrt(ppool * (1.0 - ppool) / float(n1))
    sd2p = math.sqrt(ppool * (1.0 - ppool) / float(n2))
    sdiff = math.sqrt(sd1p * sd1p + sd2p * sd2p)

    tval = math.fabs(p1 - p2) / sdiff
    df = 100000    # to get Gaussian
    x = df / (df + tval * tval)
    P = betainc(0.5 * df, 0.5, x)

    return p1, p2, sd1, sd2, tval, P

def stats_continuous_printout(X, Y, paired, output=sys.stdout):
    """
    """
    
    nx = X.shape[0]
    xmean = np.mean(X)
    xsd = np.std(X)
    xsdm = xsd / math.sqrt(nx)

    ny = Y.shape[0]
    ymean = np.mean(Y)
    ysd = np.std(Y)
    ysdm = xsd / math.sqrt(ny)

    if nx == ny:
        D = X - Y
        dmean = np.mean(D)
        dsd = np.std(D)
        dsdm = xsd / math.sqrt(nx)

    'Display data statistics on main frame Tab2.'
    # AP 021209 : many added hard coded tabs ('\t') to ease copy and paste of data
    # First line of output now specifies source file or manual entry
    output.write('   n                \t  {0:d}'.format(nx) +
    '                \t  {0:d}'.format(ny) +
    '\n  Mean \t {0:.6f}    \t  {1:.6f}'.format(xmean, ymean) +
    '\n  SD \t {0:.6f}    \t  {1:.6f}'.format(xsd, ysd) +
    '\n  SDM \t {0:.6f}    \t  {1:.6f}'.format(xsdm, ysdm))

    if nx == ny:
        output.write('\n Mean difference (dbar) = \t {0:.6f}'.
        format(dmean) +
        '\n  s(d) = \t {0:.6f} \t s(dbar) = \t {1:.6f}'.format(dsd, dsdm))

    if paired:
        tval, P = ttest_rel(X, Y)
        output.write('\n Paired Student''s t-test:' +
            '\n t = \t {0:.6f}'.format(tval) +
            '\n two tail P = \t {0:.6f}'.format(P))

    else:
        tval, P = ttest_ind(X, Y)
        output.write('\n Two-sample unpaired Student''s t-test:' +
            '\n t = \t {0:.6f}'.format(tval) +
            '\n two tail P = \t {0:.6f}'.format(P))

def rantest_continuous_printout(X, Y, paired, nran, output=sys.stdout):
    """
    """

    ng1, nl1, na1 = rantest_continuous(X, Y, paired, nran)
    pg1 = float(ng1) / float(nran)
    pl1 = float(nl1) / float(nran)
    pa1 = float(na1) / float(nran)

    output.write('\n\n   {0:d} randomisations'.format(nran) +
        '\n P values for difference between means are:' +
        '\n  greater than or equal to observed: P =\t{0:.6f}'.format(pg1) +
        '\n  less than or equal to observed: P =\t{0:.6f}'.format(pl1) +
        '\n  greater than or equal in absolute value to observed: ' +
        'P = \t {0:.6f}'.format(pa1))
        
def stats_binomial_printout(ir1, if1, ir2, if2, output=sys.stdout):
    """
    """
    
    n1 = ir1 + if1
    n2 = ir2 + if2
    irt = ir1 + ir2
    ift = if1 + if2
    nt = n1 + n2

    p1, p2, sd1, sd2, tval, P = t_test_binomial(n1, n2, ir1, ir2)
    
    output.write('\n Set 1: {0:d} successes out of {1:d};'.
        format(ir1, n1) +
        '\n p1 = {0:.6f};   SD(p1) = {1:.6f}'.format(p1, sd1) +
        '\n Set 2: {0:d} successes out of {1:d};'.
        format(ir2, n2) +
        '\n p2 = {0:.6f};   SD(p2) = {1:.6f}'.format(p2, sd2) +
        '\n Observed difference between sets, p1-p2 = {0:.6f}'.
        format(p1 - p2))

    output.write('\n Observed 2x2 table: ' +
        '\n  Set 1:    {0:d}      {1:d}      {2:d}'.format(ir1, if1, n1) +
        '\n  Set 2:    {0:d}      {1:d}      {2:d}'.format(ir2, if2, n2) +
        '\n  Total:    {0:d}      {1:d}      {2:d}'.format(irt, ift, nt))

    output.write('\n Two-sample unpaired test using Gaussian' + 
        'approximation to binomial:' +
        '\n standard normal deviate = {0:.6f}; two tail P = {1:.6f}.'.
        format(tval, P))

def rantest_binomial_printout(ir1, if1, ir2, if2, nran, output=sys.stdout):
    """
    """
    n1 = ir1 + if1
    n2 = ir2 + if2
    ng1, nl1, na1, ne1, ne2 = rantest_binomial(n1, n2, ir1, ir2, nran)
    pg1 = float(ng1) / float(nran)
    pl1 = float(nl1) / float(nran)
    pe1 = float(ne1) / float(nran)
    pa1 = float(na1) / float(nran)
    pe2 = float(ne2) / float(nran)
    output.write('\n\n {0:d} randomisations'.format(nran) +
        '\n P values for difference between sets are: ' +
        '\n  r1 greater than or equal to observed: P = {0:.6f}'.
        format(pg1) +
        '\n  r1 less than or equal to observed: P = {0:.6f}'.format(pl1) +
        '\n  r1 equal to observed: number = {0:d} (P = {1:.6f})'.
        format(ne1, pe1))

def rantest_binomial(n1, n2, ir1, ir2, nran):
    """ """
    irt = ir1 + ir2
    dobs = ir1 / float(n1) - ir2 / float(n2)
    allobs = np.append(np.ones(irt), np.zeros(n1+n2-irt))
    n = ng1 = nl1 = na1 = ne1 = ne2 = 0
#    randiff = []
    while n < nran:
        n += 1
        random.shuffle(allobs)
        is2 = np.sum(allobs[n1:])
        is1 = irt - is2
        dran = is1 / float(n1) - is2 / float(n2)
#        randiff.append(float(is1))
        if dran >= dobs: ng1 += 1
        if dran <= dobs: nl1 += 1
        if dran == dobs: ne1 += 1
        if math.fabs(dran) == math.fabs(dobs): ne2 += 1
        if math.fabs(dran) >= math.fabs(dobs): na1 += 1
    return ng1, nl1, na1, ne1, ne2 #, dobs, randiff

def rantest_continuous(X, Y, paired, nran):
    """ """
    nx, ny = X.shape[0], Y.shape[0]
    ng1 = nl1 = na1 = n = 0    
#    randiff = []

    if nx == ny and paired:
        D = X - Y
        dobs = np.mean(D)    # observed mean difference
        allobs = np.fabs(D)
        while n < nran:
            n += 1
            sd = 0.0
            for i in range(nx):
                u = random.random()
                if u < 0.5:
                    sd = sd - allobs[i]
                else:
                    sd = sd + allobs[i]
            dran = sd / float(nx)
#            randiff.append(dran)
            if dran >= dobs: ng1 += 1
            if dran <= dobs: nl1 += 1
            if math.fabs(dran) >= math.fabs(dobs): na1 += 1
        return ng1, nl1, na1
    else:    # if not paired
        dobs = np.mean(X) - np.mean(Y)
        allobs = np.append(X, Y)
        stot = np.sum(allobs)
        while n < nran:
            n += 1
            random.shuffle(allobs)
            sy =  np.sum(allobs[nx:])
            sx = stot - sy
            dran = sx / float(nx) - sy / float(ny)
#            randiff.append(dran)
            if dran >= dobs: ng1 += 1
            if dran <= dobs: nl1 += 1
            if math.fabs(dran) >= math.fabs(dobs): na1 += 1
        return ng1, nl1, na1

def data_from_txt_file(filename):
    """"Asks for a tab delimited text file to use in randomization test.
    Expects two columns, can cope with different length groups, if missing
    points are written as '!!!' """

    f = open(filename, 'r')
    x1, y1 = np.genfromtxt(f, missing_values='!!!', unpack=True)
    f.close()
    data1 = np.array([ x for x in x1 if not math.isnan(x) ])
    data2 = np.array([ y for y in y1 if not math.isnan(y) ])
    return data1, data2
