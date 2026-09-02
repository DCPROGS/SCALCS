"""Mean apparent open and shut times, and conditional means.

These pin the correction made on 2026-09-02, when
``exact_mean_open_shut_time`` was found to be too large by exactly twice the
dead time and ``HJC_adjacent_mean_open_to_shut_time_pdf`` by once. The error was
in neither function's algebra but in the offset assumed for the ``dARSdS``
term, so both are checked here against references that owe nothing to the
implementation:

* for a two-state channel, eqns (121) and (122) of Colquhoun & Hawkes (1995b),
  which give the apparent means in closed form;
* for larger mechanisms, the mean of the asymptotic HJC pdf, ``tres + sum(area
  * tau)``, computed from the roots and areas -- a different route through the
  library entirely;
* and the limit ``tres -> 0``, where the apparent means must become the ideal
  ones.
"""

import numpy as np
import numpy.testing as npt
import pytest

from scalcs import qmatlib as qml
from scalcs import scalcslib as scl
from scalcs.samples import samples

#: The C-O sample: mean open 1/50 s, mean shut 1/20 s.
CO_MEAN_OPEN, CO_MEAN_SHUT = 1 / 50.0, 1 / 20.0


def blue_book_means(mo, ms, tres):
    """Apparent mean open and shut times of a two-state channel.

    Eqns (121) and (122) of Colquhoun & Hawkes (1995b), p. 456, with the same
    dead time imposed on openings and shuttings.
    """
    emo = tres + (mo + ms) * np.exp(tres / ms) - (tres + ms)
    ems = tres + (mo + ms) * np.exp(tres / mo) - (tres + mo)
    return emo, ems


def asymptotic_mean_open(mec, tres):
    """tres + sum(area * tau) for the asymptotic HJC open time pdf."""
    roots = scl.asymptotic_roots(tres, mec.QAA, mec.QFF, mec.QAF, mec.QFA,
                                 mec.kA, mec.kF)
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    areas = scl.asymptotic_areas(tres, roots, mec.QAA, mec.QFF, mec.QAF,
                                 mec.QFA, mec.kA, mec.kF, GAF, GFA)
    return tres + float(np.sum(areas * (-1.0 / roots)))


# ------------------------------------------- two states, against the algebra

@pytest.mark.parametrize('tres', [1e-3, 2e-3, 5e-3, 1e-2, 2e-2])
def test_two_state_apparent_means(tres):
    mec = samples.CO()
    expected = blue_book_means(CO_MEAN_OPEN, CO_MEAN_SHUT, tres)
    npt.assert_allclose(scl.exact_mean_open_shut_time(mec, tres), expected,
                        rtol=1e-9)


def test_two_state_conditional_mean_is_flat():
    """A two-state channel has no correlations, so the mean open time cannot
    depend on the adjacent shut time, and must equal the apparent mean."""
    mec = samples.CO()
    tres = 5e-3
    expected, _ = blue_book_means(CO_MEAN_OPEN, CO_MEAN_SHUT, tres)
    shut = np.array([5e-3, 1e-2, 3e-2, 1e-1, 5e-1])
    mp, mn = scl.HJC_adjacent_mean_open_to_shut_time_pdf(
        shut, tres, mec.Q, mec.QAA, mec.QAF, mec.QFF, mec.QFA)
    npt.assert_allclose(mp, expected, rtol=1e-9)
    npt.assert_allclose(mn, expected, rtol=1e-9)


# -------------------------------------- more states, against a second route

@pytest.mark.parametrize('name, conc, tres', [
    ('CH82', 100e-9, 30e-6),
    ('AChR_diamond', 30e-9, 25e-6),
    ('AChR_diamond', 10e-6, 25e-6),
])
def test_mean_open_matches_the_asymptotic_pdf(name, conc, tres):
    mec = getattr(samples, name)()
    mec.set_eff('c', conc)
    mean_open, _ = scl.exact_mean_open_shut_time(mec, tres)
    npt.assert_allclose(mean_open, asymptotic_mean_open(mec, tres), rtol=1e-5)


def test_conditional_mean_agrees_at_long_shut_times():
    """Far out, the conditional mean must lie within the range the
    unconditional apparent mean and the ideal mean bracket."""
    mec = samples.AChR_diamond()
    mec.set_eff('c', 30e-9)
    tres = 25e-6
    mp, mn = scl.HJC_adjacent_mean_open_to_shut_time_pdf(
        np.array([1.0]), tres, mec.Q, mec.QAA, mec.QAF, mec.QFF, mec.QFA)
    assert 0.0 < mn[0] < scl.exact_mean_open_shut_time(mec, tres)[0]
    assert 0.0 < mp[0] < scl.exact_mean_open_shut_time(mec, tres)[0]


# ------------------------------------------------------------- the tres = 0 limit

@pytest.mark.parametrize('name, conc', [('CO', None),
                                        ('CH82', 100e-9),
                                        ('AChR_diamond', 30e-9)])
def test_zero_dead_time_gives_the_ideal_means(name, conc):
    mec = getattr(samples, name)()
    if conc is not None:
        mec.set_eff('c', conc)
    ideal_open = float(np.sum(qml.phiA(mec) @ (-np.linalg.inv(mec.QAA))))
    mean_open, _ = scl.exact_mean_open_shut_time(mec, 1e-12)
    npt.assert_allclose(mean_open, ideal_open, rtol=1e-6)
