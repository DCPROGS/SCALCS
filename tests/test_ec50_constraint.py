"""EC50: an accurate value, and using one to constrain a rate constant.

Both come from Colquhoun, Hatton & Hawkes (2003) J Physiol 547:699-728, which
supplies an independently measured EC50 and computes one rate constant from it
at every iteration of a fit (p. 702), reducing the free parameters by one.
"""

import numpy as np
import numpy.testing as npt
import pytest

from scalcs import mechanism, popen
from scalcs.samples import samples


KP2A = 7          # index of k(+2a) in AChR_diamond's rate list


def rate_value(mec, i):
    return float(np.ravel(mec.Rates[i].rateconstants)[0])


@pytest.fixture
def independent():
    """AChR_diamond with the two sites constrained independent: 10 free rates."""
    mec, _ = samples.load_AChR_diamond_independent_binding()
    return mec


# ------------------------------------------------------------- EC50 accuracy

@pytest.mark.parametrize('setname, published', [('true1', 3.2993e-6),
                                                ('true2', 9.6961e-6)])
def test_EC50_matches_the_published_value(setname, published):
    """Table 1 of the paper prints 3.3 uM and 9.697 uM.

    Before the tolerance was tightened this returned 3.2932 and 9.6858, wrong
    by about 0.2%, because it stopped once Popen was within 0.001 of half
    maximal instead of converging on the concentration.
    """
    mec = samples.AChR_diamond(setname)
    npt.assert_allclose(popen.EC50(mec, 0), published, rtol=1e-4)


def test_EC50_is_exactly_half_maximal():
    """The defining property, to far better than the old 1e-3."""
    mec = samples.AChR_diamond()
    ec50 = popen.EC50(mec, 0)
    P0 = popen.Popen0(mec, 0)
    maxP, _ = popen.maxPopen(mec, 0)
    response = (popen.Popen(mec, 0, ec50) - P0) / (maxP - P0)
    npt.assert_allclose(response, 0.5, atol=1e-9)


def test_EC50_of_a_flat_mechanism_is_nan():
    """No half-maximal response exists when Popen does not change."""
    mec = samples.AChR_diamond()
    for rate in mec.Rates:                      # kill the concentration terms
        if 'k(+' in rate.name:
            rate.rateconstants = 1e-12
    mec.update_submat()
    value = popen.EC50(mec, 0)
    assert value != value or value > 0          # nan, or a huge concentration


# --------------------------------------------------------- the constraint

def test_constraint_removes_one_free_parameter(independent):
    before = independent.get_free_parameter_names()
    independent.set_EC50_constraint(KP2A, 3.3e-6)
    after = independent.get_free_parameter_names()
    assert len(before) - len(after) == 1
    assert 'k(+2a)' in before and 'k(+2a)' not in after
    assert not independent.Rates[KP2A].is_free


def test_constraint_is_met(independent):
    independent.set_EC50_constraint(KP2A, 3.3e-6)
    npt.assert_allclose(popen.EC50(independent, 0), 3.3e-6, rtol=1e-6)


def test_constraint_recomputed_when_other_rates_change(independent):
    independent.set_EC50_constraint(KP2A, 3.3e-6)
    first = rate_value(independent, KP2A)
    independent.theta_unsqueeze(independent.theta() * 0.5)
    second = rate_value(independent, KP2A)
    assert second != first, 'the constrained rate should have moved'
    npt.assert_allclose(popen.EC50(independent, 0), 3.3e-6, rtol=1e-6)


def test_constraint_leaves_the_tied_rate_following(independent):
    """k(+1a) is a multiple of k(+2a), so it must follow it."""
    independent.set_EC50_constraint(KP2A, 3.3e-6)
    assert rate_value(independent, 11) == rate_value(independent, KP2A)


def test_clear_restores_the_free_parameter(independent):
    independent.set_EC50_constraint(KP2A, 3.3e-6)
    independent.clear_EC50_constraint()
    assert 'k(+2a)' in independent.get_free_parameter_names()
    assert independent.Rates[KP2A].is_free
    assert independent.ec50_clamped == 0


def test_custom_solver_is_used(independent):
    independent.set_EC50_constraint(KP2A, 3.3e-6, solver=lambda mec: 1.234e8)
    assert rate_value(independent, KP2A) == 1.234e8
    assert rate_value(independent, 11) == 1.234e8      # and the tie follows


# --------------------------------------------------- unreachable EC50 values

def unreachable_mechanism():
    """Guess 2 of the paper's Table 1 cannot be given an EC50 of 3.3 uM.

    Raising an association rate constant does not drive the EC50 to zero: it
    falls to a floor set by the other rates, 5.76 uM in this case.
    """
    guess2 = {'beta1a': 20.0, 'alpha1a': 2000.0, 'beta1b': 300.0,
              'alpha1b': 80000.0, 'beta2': 50000.0, 'alpha2': 1500.0,
              'k(-2a)': 1000.0, 'k(+2a)': 1.0e8, 'k(-2b)': 20000.0,
              'k(+2b)': 1.0e8, 'k(-1a)': 1000.0, 'k(+1a)': 1.0e8,
              'k(-1b)': 20000.0, 'k(+1b)': 1.0e8}
    mec, _ = samples.load_AChR_diamond_independent_binding(guess2)
    return mec


def test_unreachable_clamps_by_default():
    mec = unreachable_mechanism()
    mec.set_EC50_constraint(KP2A, 3.3e-6)
    assert mec.ec50_clamped == 1
    assert rate_value(mec, KP2A) == mec.Rates[KP2A].limits[0][1]
    assert popen.EC50(mec, 0) > 3.3e-6, 'clamped, so the EC50 is not met'


def test_unreachable_can_raise_instead():
    mec = unreachable_mechanism()
    with pytest.raises(ArithmeticError, match='cannot reach an EC50'):
        mec.set_EC50_constraint(KP2A, 3.3e-6, on_unreachable='raise')


def test_reachable_target_on_the_same_mechanism():
    """The same guess can be given an EC50 of 6.6 uM, above its floor."""
    mec = unreachable_mechanism()
    mec.set_EC50_constraint(KP2A, 6.6e-6)
    assert mec.ec50_clamped == 0
    npt.assert_allclose(popen.EC50(mec, 0), 6.6e-6, rtol=1e-6)


def test_solver_that_refuses_is_clamped():
    def refuse(mec):
        raise ArithmeticError('no positive root')
    mec = unreachable_mechanism()
    mec.set_EC50_constraint(KP2A, 3.3e-6, solver=refuse)
    assert mec.ec50_clamped == 1
    assert rate_value(mec, KP2A) == mec.Rates[KP2A].limits[0][1]


def test_solver_out_of_limits_is_clamped():
    mec = unreachable_mechanism()
    mec.set_EC50_constraint(KP2A, 3.3e-6, solver=lambda m: 1e30)
    assert mec.ec50_clamped == 1
    assert rate_value(mec, KP2A) == mec.Rates[KP2A].limits[0][1]


# ------------------------------------------------------------------- errors

def test_bad_arguments(independent):
    with pytest.raises(IndexError):
        independent.set_EC50_constraint(99, 3.3e-6)
    with pytest.raises(ValueError):
        independent.set_EC50_constraint(KP2A, -1.0)
    with pytest.raises(ValueError):
        independent.set_EC50_constraint(KP2A, 3.3e-6, on_unreachable='explode')


def test_is_free_covers_every_kind_of_constraint():
    mec = samples.AChR_diamond()
    assert all(rate.is_free for rate in mec.Rates)
    mec.Rates[0].fixed = True
    mec.Rates[1].mr = True
    mec.Rates[2].is_constrained = True
    mec.Rates[3].ec50_constrained = True
    assert [r.is_free for r in mec.Rates[:5]] == [False, False, False, False, True]
    assert len(mec.get_free_parameter_names()) == len(mec.Rates) - 4
