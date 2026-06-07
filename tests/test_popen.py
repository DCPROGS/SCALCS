"""Tests for scalcs.popen — equilibrium open probability calculations.

Reference mechanism: CH82 (Colquhoun & Hawkes 1982), tres=0 (ideal, no
missed-event correction) unless stated otherwise.

Test strategy
-------------
* tres=0 path: Popen = sum(pinf over open states) — exact analytical check.
* Monotonicity: CH82 is an agonist-gated channel → Popen increases with [A].
* Range: Popen ∈ (0, 1) at all positive concentrations.
* Regression: specific values pinned at 1 nM, 100 nM, 1 mM (100 nM is the
  primary reference, 1 mM is near-saturating).
* Non-monotonic path: CH82 + fast pore blocker produces a Popen peak and then
  decline at high concentrations; this exercises the maxPopen bisection branch
  that was previously unreachable for a plain monotonic mechanism.

maxPopen bisection bug (fixed 2026-06-07)
-----------------------------------------
The original code contained five errors in the non-monotonic bisection loop:

1. P1 evaluated at ``conc`` instead of ``conc1 = conc / fac``
2. ``conc1`` overwritten by ``conc * fac`` (should be a separate ``conc2``)
3. P2 evaluated at ``conc`` instead of ``conc2``
4. Bracket updates were backwards and referenced ``conc1`` instead of ``conc``
5. ``nstep`` never incremented → loop could not converge

The net effect: P1 == P2 always, so ``perr = 0`` after the first iteration and
the loop exited with only one bisection step performed.  For a monotonic
mechanism the bug branch is never entered; only non-monotonic Popen curves
(e.g. with a fast pore blocker) exposed it.
"""

import math
import numpy as np
import pytest

from scalcs import qmatlib as qml
from scalcs import popen as pop
from scalcs.samples.samples import CH82


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ch82():
    """Fresh CH82 mechanism — concentration set per test to avoid state bleed."""
    return CH82()


@pytest.fixture(scope="module")
def ch82_fastblk():
    """CH82 + fast pore blocker (KB = 1 mM).

    With KB = 1 mM the fast-blocker correction becomes significant only
    above ~ 1 mM, placing the Popen peak squarely in the mM range where
    CH82's Popen_ideal curve is already well-developed.  The peak is broad
    enough to be reliably bracketed by the outer search loop (sqrt(10) steps),
    which exercises the ``maxPopen`` bisection branch that is dead code for
    a plain monotonic mechanism.

    KB = 100 nM was originally tried but the peak falls at ~ 2.4 µM — a
    narrow feature that can be missed between adjacent sqrt(10) search steps
    (1 µM and 3.16 µM both lie on the same side of the peak when viewed
    step-over-step).
    """
    mec = CH82()
    mec.fastKB = 1e-3   # 1 mM → moderate fast-block effect
    return mec


# ---------------------------------------------------------------------------
# Regression constants (CH82, tres=0)
# Captured from scalcs source on modernise branch.
# ---------------------------------------------------------------------------
_POPEN_1nM   = 4.4e-7    # effectively zero at sub-nM
_POPEN_100nM = 0.00188686
_POPEN_1mM   = 0.96748731


# ---------------------------------------------------------------------------
# tres=0 (ideal) Popen
# ---------------------------------------------------------------------------

class TestPopenIdeal:

    def test_equals_pinf_sum_open(self, ch82):
        """tres=0 Popen must equal equilibrium open-state occupancy."""
        conc = 100e-9
        ch82.set_eff('c', conc)
        pinf_open = np.sum(qml.pinf(ch82.Q)[:ch82.kA])
        popen = pop.Popen(ch82, tres=0, conc=conc)
        assert popen == pytest.approx(float(pinf_open), rel=1e-8)

    def test_in_unit_interval(self, ch82):
        """Popen must be in (0, 1) for all positive concentrations."""
        for conc in [1e-9, 1e-7, 1e-5, 1e-3]:
            p = pop.Popen(ch82, tres=0, conc=conc)
            assert 0.0 < p < 1.0, f"Popen={p} at conc={conc}"

    def test_monotone_increasing(self, ch82):
        """CH82 is agonist-gated: Popen must increase with concentration."""
        concs = [1e-9, 100e-9, 1e-3]
        popens = [pop.Popen(ch82, tres=0, conc=c) for c in concs]
        assert all(popens[i] < popens[i+1] for i in range(len(popens)-1))

    def test_approaches_zero_at_zero_conc(self, ch82):
        """CH82 has no spontaneous openings: Popen→0 as conc→0."""
        p = pop.Popen(ch82, tres=0, conc=1e-12)
        assert p < 1e-6

    def test_approaches_one_at_saturation(self, ch82):
        """At very high agonist concentration Popen should approach maxPopen."""
        p = pop.Popen(ch82, tres=0, conc=0.1)   # 100 mM — near saturation
        assert p > 0.90

    # ---- regression ----

    def test_regression_100nM(self, ch82):
        p = pop.Popen(ch82, tres=0, conc=100e-9)
        assert p == pytest.approx(_POPEN_100nM, rel=1e-5)

    def test_regression_1mM(self, ch82):
        p = pop.Popen(ch82, tres=0, conc=1e-3)
        assert p == pytest.approx(_POPEN_1mM, rel=1e-5)


# ---------------------------------------------------------------------------
# Popen0 — open probability in absence of agonist
# ---------------------------------------------------------------------------

class TestPopen0:

    def test_returns_float(self, ch82):
        p0 = pop.Popen0(ch82, tres=0)
        assert isinstance(p0, float)

    def test_near_zero_for_ch82(self, ch82):
        """CH82 has no constitutive opening: Popen0 must be negligible."""
        p0 = pop.Popen0(ch82, tres=0)
        assert p0 < 1e-6

    def test_less_than_popen_at_finite_conc(self, ch82):
        """Popen at any positive concentration must exceed Popen0."""
        p0 = pop.Popen0(ch82, tres=0)
        p  = pop.Popen(ch82, tres=0, conc=1e-9)
        assert p0 <= p


# ---------------------------------------------------------------------------
# maxPopen — monotonic mechanism (CH82, no blocker)
# ---------------------------------------------------------------------------

class TestMaxPopenMonotonic:
    """For a monotonic curve maxPopen simply returns the plateau value.

    The bisection branch is NOT exercised here; these tests verify the
    outer search loop that finds the plateau.
    """

    def test_returns_tuple(self, ch82):
        result = pop.maxPopen(ch82, tres=0)
        assert len(result) == 2

    def test_max_popen_in_unit_interval(self, ch82):
        maxP, cmax = pop.maxPopen(ch82, tres=0)
        assert 0.0 < maxP <= 1.0

    def test_cmax_positive(self, ch82):
        _, cmax = pop.maxPopen(ch82, tres=0)
        assert cmax > 0.0

    def test_popen_at_cmax_matches_maxPopen(self, ch82):
        """Popen evaluated at cmax should equal (or be very close to) maxPopen."""
        maxP, cmax = pop.maxPopen(ch82, tres=0)
        p_at_cmax = pop.Popen(ch82, tres=0, conc=cmax)
        assert p_at_cmax == pytest.approx(maxP, rel=1e-4)

    def test_maxPopen_geq_regression_1mM(self, ch82):
        """maxPopen must be ≥ Popen at 1 mM."""
        maxP, _ = pop.maxPopen(ch82, tres=0)
        assert maxP >= _POPEN_1mM - 1e-6


# ---------------------------------------------------------------------------
# maxPopen — non-monotonic mechanism (CH82 + fast blocker)
# These tests exercise the bisection branch that contained the five-bug cluster.
# ---------------------------------------------------------------------------

class TestMaxPopenNonMonotonic:
    """CH82 with a fast pore blocker produces a non-monotonic Popen curve.

    The fast-block correction is ``Popen_corrected = Popen / (1 + c/KB)``.
    With KB = 100 nM, concentrations above ~10–100 µM are strongly blocked.
    The true maximum lies somewhere in between; the bisection loop must find
    it accurately.

    These tests would FAIL against the pre-fix code because the bisection
    loop exited after one trivially wrong step.
    """

    def test_curve_is_non_monotonic(self, ch82_fastblk):
        """Popen must rise then fall: medium conc > low conc AND > very high conc."""
        mec = ch82_fastblk
        p_low  = pop.Popen(mec, tres=0, conc=1e-9)    # 1 nM — well below peak
        p_mid  = pop.Popen(mec, tres=0, conc=1e-3)    # 1 mM — near peak (KB = 1 mM)
        p_high = pop.Popen(mec, tres=0, conc=1.0)     # 1 M — heavily blocked
        assert p_mid > p_low
        assert p_mid > p_high

    def test_returns_tuple(self, ch82_fastblk):
        result = pop.maxPopen(ch82_fastblk, tres=0)
        assert len(result) == 2

    def test_max_popen_in_unit_interval(self, ch82_fastblk):
        maxP, _ = pop.maxPopen(ch82_fastblk, tres=0)
        assert 0.0 < maxP <= 1.0

    def test_cmax_positive(self, ch82_fastblk):
        _, cmax = pop.maxPopen(ch82_fastblk, tres=0)
        assert cmax > 0.0

    def test_popen_at_cmax_matches_maxPopen(self, ch82_fastblk):
        """Key regression: Popen(cmax) must equal maxP to 0.1%.

        Pre-fix: the bisection did only 1 wrong step → cmax was far off the
        true peak → this assertion failed with errors of several percent.
        Post-fix: the bracket converges to within epsc = c1/1000.
        """
        maxP, cmax = pop.maxPopen(ch82_fastblk, tres=0)
        p_at_cmax = pop.Popen(ch82_fastblk, tres=0, conc=cmax)
        assert p_at_cmax == pytest.approx(maxP, rel=1e-3)

    def test_maxP_exceeds_popen_at_nearby_concentrations(self, ch82_fastblk):
        """maxP must be ≥ Popen at concentrations on either side of the peak."""
        maxP, cmax = pop.maxPopen(ch82_fastblk, tres=0)
        for factor in [0.1, 0.5, 2.0, 10.0]:
            p = pop.Popen(ch82_fastblk, tres=0, conc=cmax * factor)
            assert maxP >= p - 1e-6, (
                f"maxP={maxP:.6g} < Popen at {cmax*factor:.3g} M = {p:.6g}"
            )

    def test_cmax_is_in_reasonable_range(self, ch82_fastblk):
        """For KB=1 mM, peak should be in the 1 µM – 100 mM range."""
        _, cmax = pop.maxPopen(ch82_fastblk, tres=0)
        assert 1e-6 < cmax < 100e-3

    def test_bisection_is_near_true_maximum(self, ch82_fastblk):
        """Verify maxPopen finds a genuine local maximum via dense scan.

        Scan 200 concentrations covering two decades around cmax on a log scale
        and verify that maxP is within 2 % (relative) of the scan maximum.
        The 2 % tolerance reflects the width of the initial search bracket
        (one sqrt(10) step) that seeds the bisection.
        """
        maxP, cmax = pop.maxPopen(ch82_fastblk, tres=0)
        # Scan a ×10 window on each side of the returned cmax
        c_scan = np.logspace(math.log10(cmax / 10), math.log10(cmax * 10), 200)
        p_scan = np.array([pop.Popen(ch82_fastblk, tres=0, conc=c) for c in c_scan])
        scan_max = float(p_scan.max())
        assert maxP == pytest.approx(scan_max, rel=0.02), (
            f"maxPopen returned {maxP:.5g} but scan found {scan_max:.5g} "
            f"at c = {float(c_scan[p_scan.argmax()]):.3g} M"
        )


# ---------------------------------------------------------------------------
# EC50 and nH — depend on maxPopen being correct
# ---------------------------------------------------------------------------

class TestEC50andnH:

    def test_ec50_positive(self, ch82):
        ec50 = pop.EC50(ch82, tres=0)
        assert ec50 > 0.0

    def test_ec50_between_zero_and_cmax(self, ch82):
        ec50 = pop.EC50(ch82, tres=0)
        _, cmax = pop.maxPopen(ch82, tres=0)
        assert ec50 < cmax

    def test_popen_at_ec50_is_half_max(self, ch82):
        """Popen(EC50) must be 50% of maxPopen (relative to Popen0)."""
        ec50  = pop.EC50(ch82, tres=0)
        P0    = pop.Popen0(ch82, tres=0)
        maxP, _ = pop.maxPopen(ch82, tres=0)
        p_ec50 = pop.Popen(ch82, tres=0, conc=ec50)
        half_response = (p_ec50 - P0) / (maxP - P0)
        assert half_response == pytest.approx(0.5, abs=1e-3)

    def test_nh_positive_for_agonist(self, ch82):
        """Hill slope must be positive for an agonist (Popen increases with [A])."""
        nh = pop.nH(ch82, tres=0)
        assert nh > 0.0

    def test_nh_reasonable_range(self, ch82):
        """Hill slope for CH82 should be between 0.5 and 3."""
        nh = pop.nH(ch82, tres=0)
        assert 0.5 < nh < 3.0
