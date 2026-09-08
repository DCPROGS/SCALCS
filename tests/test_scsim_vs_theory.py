"""The simulator against Q-matrix theory, for mechanisms with several states.

``test_scsim.py`` already checks the two-state CO mechanism, where the means
are ``1/alpha`` and ``1/beta``. That is the easy case: with one open and one
shut state the dwell times are single exponentials, and a simulator can get
them right while getting the multi-state case wrong.

These check the cases where the answer is a Q-matrix expression rather than a
reciprocal rate:

* the ideal means, ``phi_A (-Q_AA)^-1 u`` and ``phi_F (-Q_FF)^-1 u``, for
  mechanisms with three or four states of each kind;
* ``P(open)`` against the equilibrium occupancies;
* and, after a dead time is imposed, the **apparent** means against
  :func:`scalcslib.exact_mean_open_shut_time` -- the HJC missed-events theory.

The last is the one that earns its keep. Nothing else ties the simulator to the
missed-events code: ``test_apparent_means.py`` checks that code against closed
forms and against itself by another route, but never against data. An error of
two dead times in the apparent mean open time -- exactly the error that was
there until the ``dARSdS`` sign was corrected -- would be caught here even if
every analytic cross-check had been adjusted to agree with it.

How much power that check has was measured rather than assumed, because it
depends on the dead time as a fraction of the mean open time and so differs by
mechanism. At 400 000 intervals and ``tres = 25 us``:

===============  ===========  ===========  ==================
mechanism        one tres     two tres     actual disagreement
===============  ===========  ===========  ==================
CH82             3.4 SE       6.7 SE       1.3 SE
AChR_diamond     5.6 SE       11.3 SE      1.0 SE
===============  ===========  ===========  ==================

So at the tolerance used here a **two**-dead-time error is caught on both, and
a one-dead-time error on the AChR mechanism but not on CH82, whose open times
are a hundred times the dead time.
:func:`test_the_apparent_open_time_check_has_power` asserts the two-dead-time
case, so a passing suite means the check could actually fail.

Marked slow: those standard errors need of the order of 10^5 intervals.

The tolerance is a multiple of the *measured* standard error rather than a
fixed percentage, so the tests neither pass by being loose on a sharp quantity
nor fail by being tight on a noisy one. Seeds are fixed, so a failure is a real
disagreement rather than a bad draw.
"""

import numpy as np
import pytest

from scalcs import qmatlib as qml
from scalcs import scalcslib as scl
from scalcs import scsim
from scalcs.samples import samples

#: How many standard errors of disagreement to allow. With fixed seeds these
#: are deterministic; 4 leaves room for the estimator's own skew without
#: admitting an error of the size worth catching.
NSIGMA = 4.0

TRES = 25e-6
NINT = 400000


def mechanisms():
    ch82 = samples.CH82()
    ch82.set_eff('c', 100e-9)
    achr = samples.AChR_diamond()
    achr.set_eff('c', 100e-9)
    return [("CH82", ch82), ("AChR_diamond", achr)]


def ideal_means(mec):
    """phi_A (-Q_AA)^-1 u and phi_F (-Q_FF)^-1 u, and P(open)."""
    kA, kF = mec.kA, mec.k - mec.kA
    QAA, QFF = mec.Q[:kA, :kA], mec.Q[kA:, kA:]
    phiA = np.asarray(qml.phiA(mec)).reshape(1, kA)
    phiF = np.asarray(qml.phiF(mec)).reshape(1, kF)
    mopen = float(np.squeeze(phiA @ np.linalg.inv(-QAA) @ np.ones((kA, 1))))
    mshut = float(np.squeeze(phiF @ np.linalg.inv(-QFF) @ np.ones((kF, 1))))
    popen = float(np.asarray(qml.pinf(mec.Q)).ravel()[:kA].sum())
    return mopen, mshut, popen


def measured(tints, ampls):
    """Mean open, mean shut and P(open) of a record, with standard errors."""
    op, sh = tints[ampls > 0], tints[ampls == 0]
    return dict(
        mopen=op.mean(), se_open=op.std(ddof=1) / np.sqrt(op.size),
        mshut=sh.mean(), se_shut=sh.std(ddof=1) / np.sqrt(sh.size),
        popen=op.sum() / tints.sum(), n_open=op.size)


@pytest.fixture(scope="module", params=mechanisms(), ids=lambda m: m[0])
def record(request):
    """One long simulated record per mechanism, reused by every test here."""
    name, mec = request.param
    tints, ampls, _ = scsim.simulate_intervals(mec, nintmax=NINT, seed=20030547)
    at, aa = scsim.impose_resolution(tints, ampls, TRES)
    return name, mec, tints, ampls, at, aa


@pytest.mark.slow
def test_ideal_mean_open_time(record):
    """phi_A (-Q_AA)^-1 u, not 1/alpha: several open states."""
    _, mec, ti, ai, _, _ = record
    want, _, _ = ideal_means(mec)
    got = measured(ti, ai)
    assert abs(got["mopen"] - want) < NSIGMA * got["se_open"], (
        f"simulated {got['mopen']:.6g} s, theory {want:.6g} s, "
        f"{abs(got['mopen'] - want) / got['se_open']:.1f} SE")


@pytest.mark.slow
def test_ideal_mean_shut_time(record):
    _, mec, ti, ai, _, _ = record
    _, want, _ = ideal_means(mec)
    got = measured(ti, ai)
    assert abs(got["mshut"] - want) < NSIGMA * got["se_shut"], (
        f"simulated {got['mshut']:.6g} s, theory {want:.6g} s")


@pytest.mark.slow
def test_popen(record):
    """P(open) against the equilibrium occupancies of the open states."""
    _, mec, ti, ai, _, _ = record
    _, _, want = ideal_means(mec)
    got = measured(ti, ai)["popen"]
    assert abs(got - want) / want < 0.02, f"simulated {got:.6g}, theory {want:.6g}"


@pytest.mark.slow
def test_apparent_mean_open_time_matches_the_missed_events_theory(record):
    """The simulator against exact_mean_open_shut_time, after a dead time.

    This is the check with power over the missed-events code -- the only one
    anywhere that tests it against simulated data rather than against another
    expression. See the module docstring for how much power, which depends on
    the mechanism.
    """
    _, mec, _, _, at, aa = record
    want, _ = scl.exact_mean_open_shut_time(mec, TRES)
    got = measured(at, aa)
    off = abs(got["mopen"] - want)
    assert off < NSIGMA * got["se_open"], (
        f"simulated {got['mopen']*1e3:.6g} ms, theory {want*1e3:.6g} ms, "
        f"{off / got['se_open']:.1f} SE; one dead time is "
        f"{TRES / got['se_open']:.1f} SE")


@pytest.mark.slow
def test_apparent_mean_shut_time_matches_the_missed_events_theory(record):
    """The same for shut times.

    Much less sensitive -- the apparent mean shut time is large and its
    distribution broad, so a dead time is a small fraction of a standard error
    -- but a gross error in the shut-time branch would still show.
    """
    _, mec, _, _, at, aa = record
    _, want = scl.exact_mean_open_shut_time(mec, TRES)
    got = measured(at, aa)
    assert abs(got["mshut"] - want) < NSIGMA * got["se_shut"], (
        f"simulated {got['mshut']*1e3:.6g} ms, theory {want*1e3:.6g} ms")


@pytest.mark.slow
def test_the_apparent_open_time_check_has_power(record):
    """The check above is only worth having if it would fail when it should.

    A theory value wrong by **two** dead times -- the error that was actually
    present -- must lie outside the tolerance on every mechanism here. One dead
    time is asserted only where the measurement supports it: on CH82 it is 3.4
    SE, inside the 4 SE tolerance, because its open times are a hundred times
    the dead time. Claiming otherwise would make a passing suite mean less than
    it appears to.
    """
    name, mec, _, _, at, aa = record
    want, _ = scl.exact_mean_open_shut_time(mec, TRES)
    got = measured(at, aa)
    tol = NSIGMA * got["se_open"]

    for wrong in (want + 2 * TRES, want - 2 * TRES):
        assert abs(got["mopen"] - wrong) > tol, (
            f"{name}: an answer wrong by two dead times would pass. The record "
            f"is too short or the tolerance too loose to be worth anything.")

    sensitivity = TRES / got["se_open"]
    if name == "AChR_diamond":
        assert sensitivity > NSIGMA, (
            f"one dead time is only {sensitivity:.1f} SE here; it used to be "
            f"{5.6:.1f}, so the record or the mechanism has changed")
