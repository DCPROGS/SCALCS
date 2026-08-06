"""Tests for scalcs.scalcslib — ideal dwell-time PDFs and related utilities.

Reference mechanisms
--------------------
CO  : 2-state (O↔C), alpha=50 s⁻¹, beta=20 s⁻¹.
      Analytical ground truth available for all quantities.
CH82: 5-state (Colquhoun & Hawkes 1982), set at conc=100 nM.
      Regression values recorded from a verified run.

Test strategy
-------------
* CO  → analytical / property tests  (eigenvalue, area, pdf value)
* CH82→ regression tests (specific numerical outputs pinned)
        + property tests (areas sum to 1, taus positive, etc.)
"""

import math
import numpy as np
import pytest

from scalcs import qmatlib as qml
from scalcs import scalcslib as scl
from scalcs.mechanism import Mechanism, Rate, State
from scalcs.samples.samples import CH82


# ---------------------------------------------------------------------------
# Module-level constants — CO 2-state analytical reference
# ---------------------------------------------------------------------------
ALPHA = 50.0        # O→C rate (s⁻¹)
BETA  = 20.0        # C→O rate (s⁻¹)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def co():
    """2-state open↔closed mechanism with known analytical solution."""
    sO = State('A', 'O', 50e-12)
    sC = State('C', 'C', 0.0)
    rates = [
        Rate(ALPHA, sO, sC, name='alpha'),
        Rate(BETA,  sC, sO, name='beta'),
    ]
    return Mechanism(rates, mtitle='CO')


@pytest.fixture(scope="module")
def ch82():
    """CH82 5-state mechanism at 100 nM."""
    mec = CH82()
    mec.set_eff('c', 100e-9)
    return mec


# ---------------------------------------------------------------------------
# ideal_dwell_time_pdf
# ---------------------------------------------------------------------------

class TestIdealDwellTimePDF:

    def test_co_open_pdf_analytical(self, co):
        """f(t) = alpha * exp(-alpha*t) for single-open-state mechanism."""
        t = 0.01  # s
        expected = ALPHA * math.exp(-ALPHA * t)
        result = float(scl.ideal_dwell_time_pdf(t, co.QAA, qml.phiA(co)))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_co_open_pdf_at_zero_is_alpha(self, co):
        """f(0+) = alpha for exponential with rate alpha."""
        result = float(scl.ideal_dwell_time_pdf(1e-12, co.QAA, qml.phiA(co)))
        assert result == pytest.approx(ALPHA, rel=1e-6)

    def test_co_shut_pdf_analytical(self, co):
        """f(t) = beta * exp(-beta*t) for single-shut-state mechanism."""
        t = 0.02
        expected = BETA * math.exp(-BETA * t)
        result = float(scl.ideal_dwell_time_pdf(t, co.QII, qml.phiF(co)))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_ch82_open_pdf_positive(self, ch82):
        """Open time pdf must be positive for t > 0."""
        for t in [1e-4, 1e-3, 5e-3, 1e-2]:
            f = float(scl.ideal_dwell_time_pdf(t, ch82.QAA, qml.phiA(ch82)))
            assert f > 0.0, f"pdf({t}) = {f} is not positive"

    def test_ch82_open_pdf_decreasing(self, ch82):
        """Open time pdf must decrease monotonically (sum-of-exponentials)."""
        times = [1e-4, 5e-4, 1e-3, 5e-3]
        vals = [float(scl.ideal_dwell_time_pdf(t, ch82.QAA, qml.phiA(ch82)))
                for t in times]
        assert all(vals[i] > vals[i+1] for i in range(len(vals)-1))


# ---------------------------------------------------------------------------
# ideal_dwell_time_pdf_components
# ---------------------------------------------------------------------------

class TestIdealDwellTimePDFComponents:

    def test_co_open_single_component(self, co):
        """Single open state → one eigenvalue = alpha, area = 1."""
        eigs, w = scl.ideal_dwell_time_pdf_components(co.QAA, qml.phiA(co))
        assert len(eigs) == 1
        assert eigs[0] == pytest.approx(ALPHA, rel=1e-10)
        # area = w / eig = 1
        assert (w[0] / eigs[0]) == pytest.approx(1.0, rel=1e-10)

    def test_co_shut_single_component(self, co):
        """Single shut state → one eigenvalue = beta, area = 1."""
        eigs, w = scl.ideal_dwell_time_pdf_components(co.QII, qml.phiF(co))
        assert len(eigs) == 1
        assert eigs[0] == pytest.approx(BETA, rel=1e-10)
        assert (w[0] / eigs[0]) == pytest.approx(1.0, rel=1e-10)

    def test_ch82_open_returns_kA_components(self, ch82):
        """CH82 has kA=2 open states → 2 eigenvalue/amplitude pairs."""
        eigs, w = scl.ideal_dwell_time_pdf_components(ch82.QAA, qml.phiA(ch82))
        assert len(eigs) == ch82.kA == 2
        assert len(w) == ch82.kA

    def test_ch82_open_eigenvalues_positive(self, ch82):
        """Eigenvalues of -QAA must be positive (QAA is negative-definite)."""
        eigs, _ = scl.ideal_dwell_time_pdf_components(ch82.QAA, qml.phiA(ch82))
        assert np.all(eigs > 0)

    def test_ch82_open_areas_sum_to_one(self, ch82):
        """Areas of ideal open time pdf must sum to 1."""
        eigs, w = scl.ideal_dwell_time_pdf_components(ch82.QAA, qml.phiA(ch82))
        areas = w / eigs
        assert areas.sum() == pytest.approx(1.0, abs=1e-8)

    def test_ch82_shut_returns_kI_components(self, ch82):
        """CH82 has kI=3 shut states → 3 components."""
        eigs, w = scl.ideal_dwell_time_pdf_components(ch82.QFF, qml.phiF(ch82))
        kI = ch82.kB + ch82.kC   # kF = kI for CH82 (no D states)
        assert len(eigs) == kI

    def test_ch82_shut_areas_sum_to_one(self, ch82):
        eigs, w = scl.ideal_dwell_time_pdf_components(ch82.QFF, qml.phiF(ch82))
        areas = w / eigs
        assert areas.sum() == pytest.approx(1.0, abs=1e-8)

    # ---- regression: CH82 at 100 nM ----
    def test_ch82_open_eigs_regression(self, ch82):
        """Regression: CH82 ideal open time eigenvalues at 100 nM."""
        eigs, _ = scl.ideal_dwell_time_pdf_components(ch82.QAA, qml.phiA(ch82))
        np.testing.assert_allclose(sorted(eigs),
                                   sorted([500.65359469, 3050.01307531]),
                                   rtol=1e-6)

    def test_ch82_open_areas_regression(self, ch82):
        """Regression: CH82 ideal open time area fractions at 100 nM."""
        eigs, w = scl.ideal_dwell_time_pdf_components(ch82.QAA, qml.phiA(ch82))
        areas = w / eigs
        # areas sorted by ascending eigenvalue
        idx = np.argsort(eigs)
        np.testing.assert_allclose(areas[idx], [0.92761649, 0.07238351],
                                   rtol=1e-5)

    def test_ch82_shut_taus_regression(self, ch82):
        """Regression: CH82 ideal shut time constants (ms) at 100 nM."""
        eigs, _ = scl.ideal_dwell_time_pdf_components(ch82.QFF, qml.phiF(ch82))
        taus_ms = sorted(1000.0 / eigs)
        expected_ms = sorted([3789.38053, 0.48474655, 0.052598906])
        np.testing.assert_allclose(taus_ms, expected_ms, rtol=1e-5)


# ---------------------------------------------------------------------------
# transition_probability / transition_frequency
# ---------------------------------------------------------------------------

class TestTransitionProbability:

    def test_diagonal_is_zero(self, co):
        """Diagonal of transition probability matrix must be zero."""
        tp = scl.transition_probability(co.Q)
        np.testing.assert_allclose(tp.diagonal(), 0.0, atol=1e-15)

    def test_row_sums_one(self, co):
        """Each row must sum to 1 (excluding diagonal zeros)."""
        tp = scl.transition_probability(co.Q)
        np.testing.assert_allclose(tp.sum(axis=1), 1.0, atol=1e-14)

    def test_off_diagonal_non_negative(self, co):
        tp = scl.transition_probability(co.Q)
        k = tp.shape[0]
        off_diag = [tp[i, j] for i in range(k) for j in range(k) if i != j]
        assert all(v >= 0.0 for v in off_diag)

    def test_co_off_diagonal_is_one(self, co):
        """2-state CO: every off-diagonal element is 1 (only one transition)."""
        tp = scl.transition_probability(co.Q)
        # tp[O, C] = alpha / alpha = 1; tp[C, O] = beta / beta = 1
        assert tp[0, 1] == pytest.approx(1.0)
        assert tp[1, 0] == pytest.approx(1.0)

    def test_ch82_row_sums_one(self, ch82):
        tp = scl.transition_probability(ch82.Q)
        np.testing.assert_allclose(tp.sum(axis=1), 1.0, atol=1e-12)

    def test_ch82_diagonal_zero(self, ch82):
        tp = scl.transition_probability(ch82.Q)
        np.testing.assert_allclose(tp.diagonal(), 0.0, atol=1e-15)
