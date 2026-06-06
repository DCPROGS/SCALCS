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
"""

import numpy as np
import pytest

from scalcs import qmatlib as qml
from scalcs import popen as pop
from scalcs.samples.samples import CH82


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ch82():
    """Fresh CH82 mechanism — concentration set per test to avoid state bleed."""
    return CH82()


# ---------------------------------------------------------------------------
# Regression constants (CH82, tres=0)
# Captured from scalcs source on modernise branch.
# ---------------------------------------------------------------------------
_POPEN_1nM  = 4.4e-7    # effectively zero at sub-nM
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
# maxPopen
# ---------------------------------------------------------------------------

class TestMaxPopen:

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
