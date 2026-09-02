"""The two AChR schemes of Colquhoun, Hatton & Hawkes (2003), J Physiol 547.

Scheme 1 is ``samples.AChR_diamond``; scheme 2 is ``samples.AChR_diamond_desens``,
which adds a desensitised state. The published quantities these check against
are Table 1 (p. 705), the EC50 quoted on p. 713, and the description of the
simulated high-concentration records on p. 719.
"""

import numpy as np
import numpy.testing as npt
import pytest

from scalcs import popen, scsim
from scalcs.samples import samples


SCHEME1_RATE_NAMES = (
    'beta1a', 'alpha1a', 'beta1b', 'alpha1b', 'beta2', 'alpha2',
    'k(-2a)', 'k(+2a)', 'k(-2b)', 'k(+2b)',
    'k(-1a)', 'k(+1a)', 'k(-1b)', 'k(+1b)',
)


def rate_dict(mec):
    return {r.name: float(np.ravel(r.rateconstants)[0]) for r in mec.Rates}


# --------------------------------------------------------------- rate sets

def test_named_rate_sets_are_table_1():
    """The stored sets are Table 1's 'true 1' and 'true 2' columns."""
    true1 = samples.CHH2003_RATES['true1']
    true2 = samples.CHH2003_RATES['true2']
    assert set(true1) == set(SCHEME1_RATE_NAMES)
    assert set(true2) == set(SCHEME1_RATE_NAMES)
    # spot values, read from the printed table
    assert true1['alpha2'] == 2000.0 and true1['beta2'] == 52000.0
    assert true2['alpha1b'] == 40000.0 and true2['k(+1a)'] == 0.2e8


@pytest.mark.parametrize('setname, published', [
    # E2, E1a, E1b, K2a, K2b, K1a, K1b -- last seven rows of Table 1,
    # equilibrium constants in M
    ('true1', (26.0, 50/6000, 0.003, 7.5e-6, 25e-6, 7.5e-6, 25e-6)),
    ('true2', (25.0, 0.02, 0.00025, 240e-6, 4e-6, 20e-6, 1e-6/3)),
])
def test_equilibrium_constants(setname, published):
    r = samples.CHH2003_RATES[setname]
    computed = (
        r['beta2'] / r['alpha2'],
        r['beta1a'] / r['alpha1a'],
        r['beta1b'] / r['alpha1b'],
        r['k(-2a)'] / r['k(+2a)'],
        r['k(-2b)'] / r['k(+2b)'],
        r['k(-1a)'] / r['k(+1a)'],
        r['k(-1b)'] / r['k(+1b)'],
    )
    npt.assert_allclose(computed, published, rtol=1e-12)


@pytest.mark.parametrize('setname', ['true1', 'true2'])
def test_microscopic_reversibility(setname):
    """K1a.K2b == K1b.K2a round the one cycle."""
    r = samples.CHH2003_RATES[setname]
    cw = (r['k(-1a)'] / r['k(+1a)']) * (r['k(-2b)'] / r['k(+2b)'])
    acw = (r['k(-1b)'] / r['k(+1b)']) * (r['k(-2a)'] / r['k(+2a)'])
    npt.assert_allclose(cw, acw, rtol=1e-12)


def test_true1_sites_independent_true2_not():
    """Eqns (9) and (10): true 1 obeys them, true 2 does not."""
    t1, t2 = samples.CHH2003_RATES['true1'], samples.CHH2003_RATES['true2']
    for a, b in [('k(-2a)', 'k(-1a)'), ('k(-2b)', 'k(-1b)'),
                 ('k(+2b)', 'k(+1b)'), ('k(+2a)', 'k(+1a)')]:
        assert t1[a] == t1[b], f'true 1 should have {a} == {b}'
    assert t2['k(-2a)'] != t2['k(-1a)']
    assert t2['k(+2a)'] != t2['k(+1a)']


# ------------------------------------------------------------- apply_rates

def test_default_is_true1_and_unchanged():
    """The no-argument call must return exactly what it always has."""
    npt.assert_equal(rate_dict(samples.AChR_diamond()),
                     dict(samples.CHH2003_RATES['true1']))


def test_apply_rates_by_name_dict_and_sequence():
    by_name = rate_dict(samples.AChR_diamond('true2'))
    npt.assert_equal(by_name, dict(samples.CHH2003_RATES['true2']))

    by_dict = rate_dict(samples.AChR_diamond(samples.CHH2003_RATES['true2']))
    npt.assert_equal(by_dict, by_name)

    ordered = [samples.CHH2003_RATES['true2'][n] for n in SCHEME1_RATE_NAMES]
    npt.assert_equal(rate_dict(samples.AChR_diamond(ordered)), by_name)


def test_apply_rates_partial_dict_leaves_the_rest():
    mec = samples.AChR_diamond({'alpha2': 1234.0})
    rates = rate_dict(mec)
    assert rates['alpha2'] == 1234.0
    assert rates['beta2'] == samples.CHH2003_RATES['true1']['beta2']
    assert mec.Q[2, 5] == 1234.0 or mec.Q[2, 5] == pytest.approx(1234.0)


def test_apply_rates_rejects_nonsense():
    with pytest.raises(ValueError, match='unknown rate set'):
        samples.AChR_diamond('true3')
    with pytest.raises(ValueError, match='not rates of this mechanism'):
        samples.AChR_diamond({'gamma': 1.0})
    with pytest.raises(ValueError, match='expected 14 rate constants'):
        samples.AChR_diamond([1.0, 2.0])


def test_loader_accepts_a_name():
    """The constrained loader used to take an ordered list only."""
    mec, names = samples.load_AChR_diamond_independent_binding('true1')
    npt.assert_equal(rate_dict(mec), dict(samples.CHH2003_RATES['true1']))
    assert len(names) == 10, 'eqn (9) should leave ten free parameters'


# ------------------------------------------------------------------- EC50

@pytest.mark.parametrize('setname, expected', [('true1', 3.2932e-6),
                                               ('true2', 9.6858e-6)])
def test_EC50(setname, expected):
    """The paper prints 3.3 uM for true 1 (p. 713) and 9.697 uM for true 2
    (p. 716). True 1 agrees; true 2 comes out 0.11 % low, which is recorded
    as an open question of the reproduction rather than fixed here."""
    mec = samples.AChR_diamond(setname)
    npt.assert_allclose(popen.EC50(mec, 0), expected, rtol=1e-4)


# --------------------------------------------------------------- scheme 2

def test_scheme2_state_counts():
    mec = samples.AChR_diamond_desens()
    assert (mec.k, mec.kA, mec.kB, mec.kC, mec.kD) == (8, 3, 3, 1, 1)
    assert [s.name for s in mec.States][-1] == 'A2D'
    # A2D sits outside the B/C shut block the Q-matrix machinery uses
    assert mec.kF == 4


def test_scheme2_desensitised_lifetime():
    """Mean sojourn in A2D is 1/alphaD; the paper takes alphaD = 1.4 s^-1,
    giving 714 ms (p. 704)."""
    mec = samples.AChR_diamond_desens()
    d = mec.States[-1].no
    npt.assert_allclose(-1.0 / mec.Q[d, d], 1 / 1.4, rtol=1e-12)
    npt.assert_allclose(-1.0 / mec.Q[d, d], 0.714, atol=0.001)

    faster = samples.AChR_diamond_desens(alphaD=1000.0)
    npt.assert_allclose(-1.0 / faster.Q[faster.States[-1].no,
                                        faster.States[-1].no], 1e-3, rtol=1e-12)


@pytest.mark.parametrize('conc', [30e-9, 10e-6])
def test_scheme2_contains_scheme1(conc):
    """Removing the leak into A2D from A2R* recovers scheme 1 exactly."""
    m1, m2 = samples.AChR_diamond(), samples.AChR_diamond_desens()
    m1.set_eff('c', conc)
    m2.set_eff('c', conc)
    assert [s.name for s in m2.States][:7] == [s.name for s in m1.States]

    block = m2.Q[:7, :7].copy()
    a2rs = next(s.no for s in m2.States if s.name == 'A2R*')
    block[a2rs, a2rs] += 5.0                      # betaD, back out of the diagonal
    npt.assert_allclose(block, m1.Q, rtol=1e-12)


def test_scheme2_rates_can_be_named():
    mec = samples.AChR_diamond_desens('true2')
    rates = rate_dict(mec)
    for name in SCHEME1_RATE_NAMES:
        assert rates[name] == samples.CHH2003_RATES['true2'][name]
    assert rates['betaD'] == 5.0 and rates['alphaD'] == 1.4


@pytest.mark.slow
def test_scheme2_clusters_at_10uM():
    """p. 719: at 10 uM the record has desensitised periods averaging 714 ms
    and clusters of about 240 ms containing about 400 openings."""
    mec = samples.AChR_diamond_desens()
    mec.set_eff('c', 10e-6)
    tints, ampls, _ = scsim.simulate_intervals(mec, nintmax=60000, seed=11)

    clusters = scsim.extract_burst_intervals(tints, ampls, 5e-3)
    assert len(clusters) > 30, 'too few clusters to say anything'
    lengths = np.array([np.sum(c) for c in clusters])
    openings = np.array([(len(c) + 1) // 2 for c in clusters])

    # loose: one run of a stochastic simulation, checked against the paper's
    # round numbers
    assert 0.15 < lengths.mean() < 0.35, lengths.mean()
    assert 250 < openings.mean() < 550, openings.mean()

    shut = tints[ampls == 0]
    desens = shut[shut > 0.1]
    npt.assert_allclose(desens.mean(), 1 / 1.4, rtol=0.35)
