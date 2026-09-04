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


# ---------------------------------------------------------------------------
# dARSdS itself, added 2026-09-03 when the root cause was finally re-derived.
#
# The offset was traced to one sign in `qmatlib.dARSdS`: the dead-time term of
# dVAds is -tres*GAF*expQFF*GFA and had been coded positive, which made the
# function too large by 2*tres and left every caller to cancel it. Fixing it
# means callers now ADD tres, as the theory says, and it also means anyone
# calling dARSdS directly gets a correct answer -- which the tests above,
# working only through the callers, could never have shown.
#
# The check here owes nothing to the rest of the library: it differentiates
# the Laplace transform numerically, straight from its definition.

def _AR_star(s, tres, QAA, QFF, QAF, QFA):
    """AR*(s) from its definition, for numerical differentiation.

    AR*(s) = [sI - QAA - QAF M(s) QFA]^-1,
    M(s)   = (I - exp(-(sI - QFF) tres)) (sI - QFF)^-1
    """
    from scipy.linalg import expm
    kA, kF = QAA.shape[0], QFF.shape[0]
    N = s * np.eye(kF) - QFF
    M = (np.eye(kF) - expm(-N * tres)) @ np.linalg.inv(N)
    return np.linalg.inv(s * np.eye(kA) - QAA - QAF @ M @ QFA)


@pytest.mark.parametrize("tres", [10e-6, 25e-6, 100e-6])
def test_dARSdS_is_minus_the_derivative_of_the_transform(tres):
    """dARSdS must equal [-d AR*(s)/ds] at s = 0, by finite differences."""
    mec = samples.CH82()
    mec.set_eff('c', 100e-9)
    expQFF = qml.expQt(mec.QFF, tres)
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)

    got = qml.dARSdS(tres, mec.QAA, mec.QFF, GAF, GFA, expQFF,
                     mec.kA, mec.kF)

    # central difference on s, in units set by the slowest rate
    h = 1e-4 * abs(np.diag(mec.QAA)).min()
    plus = _AR_star(h, tres, mec.QAA, mec.QFF, mec.QAF, mec.QFA)
    minus = _AR_star(-h, tres, mec.QAA, mec.QFF, mec.QAF, mec.QFA)
    expected = -(plus - minus) / (2 * h)

    npt.assert_allclose(got, expected, rtol=1e-5, atol=0)


def test_mean_is_the_dARSdS_term_plus_tres():
    """The documented relationship between dARSdS and the apparent mean.

    Callers add tres because the density is written in u = t - tres and
    integrates to 1. Check both halves: the norm, and the mean.
    """
    tres = 25e-6
    mec = samples.AChR_diamond()
    mec.set_eff('c', 100e-9)

    expQFF = qml.expQt(mec.QFF, tres)
    expQAA = qml.expQt(mec.QAA, tres)
    GAF, GFA = qml.iGs(mec.Q, mec.kA, mec.kF)
    eGAF = qml.eGs(GAF, GFA, mec.kA, mec.kF, expQFF)
    eGFA = qml.eGs(GFA, GAF, mec.kF, mec.kA, expQAA)
    phiA = qml.phiHJC(eGAF, eGFA, mec.kA)
    uF = np.ones((mec.kF, 1))

    SFF = np.eye(mec.kF) - expQFF
    VA = np.eye(mec.kA) - GAF @ SFF @ GFA
    norm = float((phiA @ np.linalg.inv(VA) @ GAF @ expQFF @ uF)[0])
    npt.assert_allclose(norm, 1.0, rtol=1e-12)

    DARS = qml.dARSdS(tres, mec.QAA, mec.QFF, GAF, GFA, expQFF,
                      mec.kA, mec.kF)
    term = float((phiA @ (DARS @ (mec.QAF @ expQFF)) @ uF)[0])
    meanA, _ = scl.exact_mean_open_shut_time(mec, tres)
    npt.assert_allclose(term + tres, meanA, rtol=1e-12)
