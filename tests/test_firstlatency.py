"""Tests for scalcs.firstlatency — pdf of first latency after a concentration step.

Physical scenario
-----------------
The channel is held at zero agonist concentration, so at t = 0 the entire
population occupies shut states with equilibrium probability

    phi_shut = pinf(Q_at_c0)[kA:]

At t = 0+ the concentration steps to c1.  The first-latency pdf f_L(t) is the
pdf of the time to the first opening.

Three levels of approximation
------------------------------
ideal       Ignores missed events:
                f_L(t) = phi_shut * exp(QFF_c1 * t) * (-QFF_c1) * u_F
            Gives kF exponential components with eigenvalues of -QFF_c1.

asymptotic  HJC approximation (Colquhoun & Hawkes 1977, 1982).  Valid for
            t >> tres.  kF components whose roots are solutions of
            det[ W_F(s) ] = 0.  Same roots as ordinary shut-time asymptotic
            pdf; only the area (= initial vector) changes.

exact       HJC exact correction for tres <= t <= 3*tres, asymptotic beyond.

References
----------
CHME97 : Colquhoun, Hawkes, Merlushkin & Edmonds (1997)
         Phil Trans R Soc Lond A 355, 1743-1786.
CH82   : Colquhoun & Hawkes (1982)
         Phil Trans R Soc Lond B 300, 1-59.

Test strategy
-------------
*CO (2-state)*  Analytical ground truth: f_L(t) = beta * exp(-beta * t)
                Used for exact property tests and tres -> 0 limit.
*CH82*          5-state; c0 = 0, c1 = 0.1 mM, tres = 0.1 ms.
                Regression values pinned from verified Phase-0 run 2026-06-07.
*CHME97*        5-state with desensitisation; c0 = 0, c1 = 1 mM, tres = 0.7 ms.
                Regression values pinned from verified Phase-0 run 2026-06-07.
                CHME97 fixture requires scalcs.samples.samples.CHME97 to exist.

Fixture design
--------------
Module-scoped fixtures are used throughout so that expensive derived quantities
(asymptotic_roots, gamma_coefficients) are computed exactly once per test
session rather than once per test method:

    ch82_step / chme97_step   -- mechanism objects + phi_shut + tres
    ch82_roots / chme97_roots -- asymptotic_roots(tres, mec)    [expensive]
    ch82_areas / chme97_areas -- asymptotic_areas(...)           [fast]
    ch82_gamma / chme97_gamma -- gamma_coefficients(...)         [expensive]

All numerical regression values recorded from Phase-0 notebook execution and
cross-checked against Fortran pdfLATs.for / HJCASYMP.FOR / F0HJC.FOR.
"""

import math

import numpy as np
import pytest
from scipy import integrate

from scalcs import qmatlib as qm
from scalcs import qmatlib as qml
from scalcs import scalcslib as scl
from scalcs import firstlatency as fl          # does not exist yet -> RED
from scalcs.mechanism import Mechanism, Rate, State
from scalcs.samples.samples import CH82, CHME97   # CHME97 not in samples -> RED


# ---------------------------------------------------------------------------
# Module-level analytical constants for CO 2-state mechanism
# ---------------------------------------------------------------------------

BETA_CO  = 20.0    # s^-1  shut -> open rate (concentration-independent)
ALPHA_CO = 50.0    # s^-1  open -> shut rate


# ---------------------------------------------------------------------------
# Base fixtures (module-scoped — setup is non-trivial)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def co():
    """2-state open<->shut mechanism (analytical ground truth).

    At c = 0 the population is entirely in the shut state, so phi_shut = [1].
    Post-jump QFF = [-BETA_CO].  Ideal f_L(t) = BETA_CO * exp(-BETA_CO * t).
    """
    sO = State('A', 'O', 50e-12)
    sC = State('C', 'C', 0.0)
    rates = [
        Rate(ALPHA_CO, sO, sC, name='alpha'),
        Rate(BETA_CO,  sC, sO, name='beta'),
    ]
    return Mechanism(rates, mtitle='CO')


@pytest.fixture(scope="module")
def ch82_step():
    """CH82 mechanism: pre-jump at c = 0, post-jump at c = 0.1 mM.

    Returns (mec_post_jump, phi_shut, tres).
    phi_shut = equilibrium shut occupancies at c = 0 = [0, 0, 1]
    tres = 0.1 ms = 1e-4 s
    """
    mec0 = CH82()
    mec0.set_eff('c', 0.0)
    mec1 = CH82()
    mec1.set_eff('c', 0.0001)
    phi_shut = qml.pinf(mec0.Q)[mec0.kA:]
    tres = 1e-4   # 0.1 ms
    return mec1, phi_shut, tres


@pytest.fixture(scope="module")
def chme97_step():
    """CHME97 mechanism: pre-jump at c = 0, post-jump at c = 1 mM.

    Returns (mec_post_jump, phi_shut, tres).
    phi_shut = [0, 0, 0, 1] — all probability in unliganded state R at c = 0.
    tres = 0.7 ms = 7e-4 s
    """
    mec0 = CHME97()
    mec0.set_eff('c', 0.0)
    mec1 = CHME97()
    mec1.set_eff('c', 0.001)
    phi_shut = qml.pinf(mec0.Q)[mec0.kA:]
    tres = 7e-4   # 0.7 ms
    return mec1, phi_shut, tres


# ---------------------------------------------------------------------------
# Derived fixtures — expensive computations cached module-wide
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ch82_roots(ch82_step):
    """Asymptotic roots for CH82.  Computed once; reused by all test methods."""
    mec1, phi_shut, tres = ch82_step
    return fl.asymptotic_roots(tres, mec1)


@pytest.fixture(scope="module")
def ch82_areas(ch82_step, ch82_roots):
    """Asymptotic areas for CH82."""
    mec1, phi_shut, tres = ch82_step
    return fl.asymptotic_areas(tres, ch82_roots, phi_shut, mec1)


@pytest.fixture(scope="module")
def ch82_gamma(ch82_step):
    """Gamma coefficients (eigvals, g00, g10, g11) for CH82 exact pdf."""
    mec1, phi_shut, tres = ch82_step
    return fl.gamma_coefficients(tres, phi_shut, mec1)


@pytest.fixture(scope="module")
def chme97_roots(chme97_step):
    """Asymptotic roots for CHME97.  Computed once; reused by all test methods."""
    mec1, phi_shut, tres = chme97_step
    return fl.asymptotic_roots(tres, mec1)


@pytest.fixture(scope="module")
def chme97_areas(chme97_step, chme97_roots):
    """Asymptotic areas for CHME97."""
    mec1, phi_shut, tres = chme97_step
    return fl.asymptotic_areas(tres, chme97_roots, phi_shut, mec1)


@pytest.fixture(scope="module")
def chme97_gamma(chme97_step):
    """Gamma coefficients (eigvals, g00, g10, g11) for CHME97 exact pdf."""
    mec1, phi_shut, tres = chme97_step
    return fl.gamma_coefficients(tres, phi_shut, mec1)


# ---------------------------------------------------------------------------
# TestSamples — CHME97 must exist in scalcs.samples
# ---------------------------------------------------------------------------

class TestSamples:
    """CHME97 must be importable from scalcs.samples.samples."""

    def test_chme97_callable(self):
        mec = CHME97()
        assert mec is not None

    def test_chme97_state_counts(self):
        """kA=1 (one open state), kF=4 (B=2 within-burst, C=2 long-lived shut)."""
        mec = CHME97()
        mec.set_eff('c', 0.001)
        assert mec.kA == 1
        assert mec.kF == 4
        assert mec.k  == 5

    def test_chme97_title(self):
        mec = CHME97()
        assert 'CHME97' in mec.mtitle or 'CHME' in mec.mtitle

    def test_chme97_set_eff_changes_Q(self):
        """Q matrix should differ between c = 0 and c = 1 mM."""
        mec = CHME97()
        mec.set_eff('c', 0.0)
        Q0 = mec.Q.copy()
        mec.set_eff('c', 0.001)
        Q1 = mec.Q.copy()
        assert not np.allclose(Q0, Q1), "Q should change with concentration"


# ---------------------------------------------------------------------------
# TestIdealComponents — fl.ideal_components(QFF, phi_shut) -> (eigs, areas)
# ---------------------------------------------------------------------------

class TestIdealComponents:
    """ideal_components returns eigenvalues and areas of the ideal first-latency pdf."""

    def test_co_single_component(self, co):
        """CO has one shut state -> one component; area = 1, eig = BETA."""
        phi_shut = np.array([1.0])
        eigs, areas = fl.ideal_components(co.QFF, phi_shut)
        assert eigs.shape == (1,)
        assert areas.shape == (1,)
        assert eigs[0] == pytest.approx(BETA_CO, rel=1e-10)
        assert areas[0] == pytest.approx(1.0, rel=1e-10)

    def test_co_areas_sum_to_one(self, co):
        phi_shut = np.array([1.0])
        eigs, areas = fl.ideal_components(co.QFF, phi_shut)
        assert areas.sum() == pytest.approx(1.0, abs=1e-10)

    def test_ch82_returns_kF_components(self, ch82_step):
        mec1, phi_shut, _ = ch82_step
        eigs, areas = fl.ideal_components(mec1.QFF, phi_shut)
        assert len(eigs) == mec1.kF == 3
        assert len(areas) == mec1.kF

    def test_ch82_eigenvalues_positive(self, ch82_step):
        """Eigenvalues of -QFF must be positive."""
        mec1, phi_shut, _ = ch82_step
        eigs, _ = fl.ideal_components(mec1.QFF, phi_shut)
        assert np.all(eigs > 0)

    def test_ch82_areas_sum_to_one(self, ch82_step):
        mec1, phi_shut, _ = ch82_step
        _, areas = fl.ideal_components(mec1.QFF, phi_shut)
        assert areas.sum() == pytest.approx(1.0, abs=1e-7)

    # Regression: eigenvalues (Phase-0, 2026-06-07)
    def test_ch82_eigenvalues_regression(self, ch82_step):
        """Eigenvalues of -QFF_c1 pinned to Phase-0 run."""
        mec1, phi_shut, _ = ch82_step
        eigs, _ = fl.ideal_components(mec1.QFF, phi_shut)
        expected = np.array([9117.3892, 14283.1619, 57614.4489])
        np.testing.assert_allclose(np.sort(eigs), np.sort(expected), rtol=1e-5)

    def test_ch82_areas_regression(self, ch82_step):
        """Component areas pinned to Phase-0 run."""
        mec1, phi_shut, _ = ch82_step
        eigs, areas = fl.ideal_components(mec1.QFF, phi_shut)
        # Sort by eigenvalue to get stable ordering
        idx = np.argsort(eigs)
        areas_sorted = areas[idx]
        expected = np.array([3.28417, -2.34607, 0.06190])
        np.testing.assert_allclose(areas_sorted, expected, rtol=1e-4)

    def test_chme97_returns_kF_components(self, chme97_step):
        mec1, phi_shut, _ = chme97_step
        eigs, areas = fl.ideal_components(mec1.QFF, phi_shut)
        assert len(eigs) == mec1.kF == 4

    def test_chme97_areas_sum_to_one(self, chme97_step):
        mec1, phi_shut, _ = chme97_step
        _, areas = fl.ideal_components(mec1.QFF, phi_shut)
        assert areas.sum() == pytest.approx(1.0, abs=1e-7)

    def test_chme97_eigenvalues_regression(self, chme97_step):
        mec1, phi_shut, _ = chme97_step
        eigs, _ = fl.ideal_components(mec1.QFF, phi_shut)
        expected = np.array([1.5167, 55.079, 5004.80, 10009.40])
        np.testing.assert_allclose(np.sort(eigs), np.sort(expected), rtol=1e-4)

    def test_chme97_areas_regression(self, chme97_step):
        mec1, phi_shut, _ = chme97_step
        eigs, areas = fl.ideal_components(mec1.QFF, phi_shut)
        idx = np.argsort(eigs)
        areas_sorted = areas[idx]
        expected = np.array([0.16189, 0.85220, -0.018753, 0.004663])
        np.testing.assert_allclose(areas_sorted, expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# TestIdealPdf — fl.ideal_pdf(t, QFF, phi_shut) -> float or ndarray
# ---------------------------------------------------------------------------

class TestIdealPdf:
    """ideal_pdf evaluates the ideal first-latency pdf at given time(s)."""

    def test_co_analytical_scalar(self, co):
        """f_L(t) = BETA_CO * exp(-BETA_CO * t) for 2-state CO mechanism."""
        phi_shut = np.array([1.0])
        t = 0.05
        expected = BETA_CO * math.exp(-BETA_CO * t)
        result = fl.ideal_pdf(t, co.QFF, phi_shut)
        assert float(result) == pytest.approx(expected, rel=1e-8)

    def test_co_analytical_array(self, co):
        """Array input returns array with correct shape."""
        phi_shut = np.array([1.0])
        t = np.array([0.01, 0.05, 0.1])
        expected = BETA_CO * np.exp(-BETA_CO * t)
        result = fl.ideal_pdf(t, co.QFF, phi_shut)
        assert result.shape == t.shape
        np.testing.assert_allclose(result, expected, rtol=1e-8)

    def test_ch82_positive(self, ch82_step):
        """Ideal pdf must be positive for t > 0."""
        mec1, phi_shut, _ = ch82_step
        for t in [1e-4, 5e-4, 1e-3]:
            f = float(fl.ideal_pdf(t, mec1.QFF, phi_shut))
            assert f > 0.0, f"ideal_pdf({t}) = {f} <= 0"

    # Regression against Phase-0 notebook output
    def test_ch82_pdf_value_regression(self, ch82_step):
        mec1, phi_shut, _ = ch82_step
        f = float(fl.ideal_pdf(2e-4, mec1.QFF, phi_shut))
        assert f == pytest.approx(2909.231, rel=1e-4)

    def test_chme97_pdf_value_regression(self, chme97_step):
        mec1, phi_shut, _ = chme97_step
        f = float(fl.ideal_pdf(0.01, mec1.QFF, phi_shut))
        assert f == pytest.approx(27.3015, rel=1e-3)


# ---------------------------------------------------------------------------
# TestAsymptoticRoots — fl.asymptotic_roots(tres, mec) -> roots
# ---------------------------------------------------------------------------

class TestAsymptoticRoots:
    """asymptotic_roots returns kF negative roots for the shut-state pdf."""

    def test_ch82_returns_kF_negative_roots(self, ch82_step, ch82_roots):
        mec1, _, tres = ch82_step
        assert len(ch82_roots) == mec1.kF == 3
        assert np.all(ch82_roots < 0), "All roots must be negative"

    def test_ch82_roots_regression(self, ch82_roots):
        """Roots pinned to Phase-0 values (sorted by magnitude)."""
        expected = np.array([-55346.36, -13091.32, -9013.82])
        np.testing.assert_allclose(
            np.sort(ch82_roots), np.sort(expected), rtol=1e-4
        )

    def test_chme97_returns_kF_negative_roots(self, chme97_step, chme97_roots):
        mec1, _, tres = chme97_step
        assert len(chme97_roots) == mec1.kF == 4
        assert np.all(chme97_roots < 0)

    def test_chme97_roots_regression(self, chme97_roots):
        expected = np.array([-10009.40, -5004.49, -33.081, -1.3209])
        np.testing.assert_allclose(
            np.sort(chme97_roots), np.sort(expected), rtol=1e-3
        )


# ---------------------------------------------------------------------------
# TestAsymptoticAreas — fl.asymptotic_areas(tres, roots, phi_shut, mec)
# ---------------------------------------------------------------------------

class TestAsymptoticAreas:
    """asymptotic_areas returns the first-latency HJC component areas."""

    def test_ch82_shape(self, ch82_step, ch82_areas):
        mec1, phi_shut, tres = ch82_step
        assert ch82_areas.shape == (mec1.kF,)

    def test_ch82_areas_sum_near_one(self, ch82_areas):
        """Sum of asymptotic areas < 1 (HJC correction), but within 2%."""
        assert ch82_areas.sum() == pytest.approx(1.0, abs=0.02)

    def test_ch82_areas_regression(self, ch82_roots, ch82_areas):
        """Areas pinned to Phase-0 values (matched by root order)."""
        # Sort both roots and areas by root magnitude for stable comparison
        idx = np.argsort(np.abs(ch82_roots))
        areas_sorted = ch82_areas[idx]
        expected = np.array([3.74943, -2.79281, 0.035272])  # |roots| ascending
        np.testing.assert_allclose(areas_sorted, expected, rtol=1e-3)

    def test_chme97_shape(self, chme97_step, chme97_areas):
        mec1, phi_shut, tres = chme97_step
        assert chme97_areas.shape == (mec1.kF,)

    def test_chme97_areas_sum_near_one(self, chme97_areas):
        """CHME97 asymptotic areas sum to within 0.1% of 1."""
        assert chme97_areas.sum() == pytest.approx(1.0, abs=0.001)

    def test_chme97_areas_regression(self, chme97_roots, chme97_areas):
        idx = np.argsort(np.abs(chme97_roots))
        areas_sorted = chme97_areas[idx]
        expected = np.array([0.27736, 0.72993, -0.009545, 0.001928])
        np.testing.assert_allclose(areas_sorted, expected, rtol=1e-3)

    def test_distinct_from_ideal_areas(self, ch82_step, ch82_areas):
        """Asymptotic areas must differ from ideal areas (HJC correction exists)."""
        mec1, phi_shut, tres = ch82_step
        _, ideal_areas = fl.ideal_components(mec1.QFF, phi_shut)
        # The HJC asymptotic pdf is NOT the same as the ideal pdf
        assert not np.allclose(np.sort(ch82_areas), np.sort(ideal_areas), atol=0.01)


# ---------------------------------------------------------------------------
# TestAsymptoticPdf — fl.asymptotic_pdf(t, tres, tau, areas) -> float/ndarray
# ---------------------------------------------------------------------------

class TestAsymptoticPdf:
    """asymptotic_pdf is zero before tres, expPDF(t - tres) after."""

    def test_zero_before_tres(self, ch82_step, ch82_roots, ch82_areas):
        """f_L(t) = 0 for t < tres (no events before dead time)."""
        mec1, phi_shut, tres = ch82_step
        tau = -1.0 / ch82_roots
        for t in [0.0, tres * 0.1, tres * 0.5, tres * 0.9999]:
            f = fl.asymptotic_pdf(t, tres, tau, ch82_areas)
            assert float(f) == 0.0, f"asymptotic_pdf({t}) should be 0 before tres={tres}"

    def test_nonzero_after_tres(self, ch82_step, ch82_roots, ch82_areas):
        """f_L(t) > 0 for t slightly above tres (positive overall amplitude)."""
        mec1, phi_shut, tres = ch82_step
        tau = -1.0 / ch82_roots
        t = tres * 1.5
        f = fl.asymptotic_pdf(t, tres, tau, ch82_areas)
        assert float(f) > 0.0

    def test_scalar_input(self, ch82_step, ch82_roots, ch82_areas):
        """scalar t must work without raising AttributeError."""
        mec1, phi_shut, tres = ch82_step
        tau = -1.0 / ch82_roots
        f = fl.asymptotic_pdf(2e-4, tres, tau, ch82_areas)
        assert isinstance(float(f), float)

    def test_array_input(self, ch82_step, ch82_roots, ch82_areas):
        """Array t must return array of same shape."""
        mec1, phi_shut, tres = ch82_step
        tau = -1.0 / ch82_roots
        t = np.array([0.5e-4, 1.0e-4, 2.0e-4, 5.0e-4])
        f = fl.asymptotic_pdf(t, tres, tau, ch82_areas)
        assert f.shape == t.shape
        # t < tres -> 0; t > tres -> positive
        assert f[0] == 0.0   # 0.5e-4 < 1e-4 = tres
        assert f[1] == 0.0   # exactly tres -- convention: f(tres) = 0
        assert f[2] > 0.0    # 2e-4 > tres
        assert f[3] > 0.0

    def test_ch82_pdf_value_regression(self, ch82_step, ch82_roots, ch82_areas):
        """f_L(t=0.0002) = 3855.8 s^-1 pinned to Phase-0 run."""
        mec1, phi_shut, tres = ch82_step
        tau = -1.0 / ch82_roots
        f = fl.asymptotic_pdf(2e-4, tres, tau, ch82_areas)
        assert float(f) == pytest.approx(3855.84, rel=1e-3)

    def test_chme97_pdf_value_regression(self, chme97_step, chme97_roots, chme97_areas):
        """f_L(t=10 ms) pinned to Phase-0 run."""
        mec1, phi_shut, tres = chme97_step
        tau = -1.0 / chme97_roots
        f = fl.asymptotic_pdf(0.01, tres, tau, chme97_areas)
        assert float(f) == pytest.approx(18.114, rel=1e-3)


# ---------------------------------------------------------------------------
# TestGammaCoefficients — fl.gamma_coefficients(tres, phi_shut, mec)
# ---------------------------------------------------------------------------

class TestGammaCoefficients:
    """gamma_coefficients returns eigenvalues and (g00, g10, g11) for exact pdf."""

    def test_ch82_returns_four_arrays(self, ch82_step, ch82_gamma):
        mec1, phi_shut, tres = ch82_step
        eigvals, g00, g10, g11 = ch82_gamma
        assert eigvals.shape == (mec1.k,)
        assert g00.shape == (mec1.k,)
        assert g10.shape == (mec1.k,)
        assert g11.shape == (mec1.k,)

    def test_ch82_eigenvalues_regression(self, ch82_gamma):
        """Eigenvalues of -Q (full) pinned to Phase-0 run.

        There are k=5 eigenvalues; one is ~0 (stationary distribution).
        """
        eigvals, _, _, _ = ch82_gamma
        # Eigenvalues of -Q sorted ascending; first ~= 0
        ev = np.sort(eigvals)
        assert ev[0] == pytest.approx(0.0, abs=1e-6)
        # Remaining four should be positive
        assert np.all(ev[1:] > 0)

    def test_ch82_g00_regression(self, ch82_gamma):
        """g00 coefficients pinned to Phase-0 run (tolerance 1%)."""
        eigvals, g00, _, _ = ch82_gamma
        # Sort by eigenvalue for stable ordering
        idx = np.argsort(eigvals)
        g00_sorted = g00[idx]
        # Phase-0 values (sorted by eigenvalue ascending):
        #   [ 458.88, 25121.5, -28978.8,   7.709,  3390.7 ]
        expected = np.array([458.88, 25121.5, -28978.8, 7.709, 3390.7])
        np.testing.assert_allclose(g00_sorted, expected, rtol=0.01)

    def test_chme97_returns_four_arrays(self, chme97_step, chme97_gamma):
        mec1, phi_shut, tres = chme97_step
        eigvals, g00, g10, g11 = chme97_gamma
        assert eigvals.shape == (mec1.k,)

    def test_chme97_eigenvalues_one_near_zero(self, chme97_gamma):
        """k=5; one eigenvalue of -Q is ~= 0 (equilibrium)."""
        eigvals, _, _, _ = chme97_gamma
        assert np.min(np.abs(eigvals)) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestExactPdf — fl.exact_pdf(t, tres, roots, areas, eigvals, g00, g10, g11)
# ---------------------------------------------------------------------------

class TestExactPdf:
    """exact_pdf evaluates the HJC-exact first-latency pdf.

    Key requirement: must accept both scalar and array t.
    (The existing scl.exact_pdf requires array t — this is fixed here.)
    """

    def test_scalar_input_no_error(self, ch82_step, ch82_roots, ch82_areas, ch82_gamma):
        """fl.exact_pdf must not raise AttributeError for scalar t."""
        mec1, phi_shut, tres = ch82_step
        eigvals, g00, g10, g11 = ch82_gamma
        f = fl.exact_pdf(2e-4, tres, ch82_roots, ch82_areas, eigvals, g00, g10, g11)
        assert isinstance(float(f), float)

    def test_array_input(self, ch82_step, ch82_roots, ch82_areas, ch82_gamma):
        """Array t must return array of same shape."""
        mec1, phi_shut, tres = ch82_step
        eigvals, g00, g10, g11 = ch82_gamma
        t = np.array([2e-4, 5e-4, 1e-3])
        f = fl.exact_pdf(t, tres, ch82_roots, ch82_areas, eigvals, g00, g10, g11)
        assert f.shape == t.shape

    def test_zero_before_tres(self, ch82_step, ch82_roots, ch82_areas, ch82_gamma):
        """f = 0 for t < tres."""
        mec1, phi_shut, tres = ch82_step
        eigvals, g00, g10, g11 = ch82_gamma
        for t in [0.0, tres * 0.5, tres * 0.999]:
            f = fl.exact_pdf(t, tres, ch82_roots, ch82_areas, eigvals, g00, g10, g11)
            assert float(f) == 0.0

    def test_equals_asymptotic_for_large_t(self, ch82_step, ch82_roots, ch82_areas, ch82_gamma):
        """For t >> 3*tres exact and asymptotic pdf must agree to 0.1%."""
        mec1, phi_shut, tres = ch82_step
        eigvals, g00, g10, g11 = ch82_gamma
        tau = -1.0 / ch82_roots
        for t in [0.002, 0.005, 0.01]:
            f_exact = float(fl.exact_pdf(t, tres, ch82_roots, ch82_areas, eigvals, g00, g10, g11))
            f_asym  = float(fl.asymptotic_pdf(t, tres, tau, ch82_areas))
            assert f_exact == pytest.approx(f_asym, rel=1e-3), (
                f"exact ({f_exact:.6e}) != asymptotic ({f_asym:.6e}) at t={t}"
            )

    def test_differs_from_asymptotic_near_tres(self, chme97_step, chme97_roots, chme97_areas, chme97_gamma):
        """Just above tres exact and asymptotic differ (exact correction matters)."""
        mec1, phi_shut, tres = chme97_step
        eigvals, g00, g10, g11 = chme97_gamma
        tau = -1.0 / chme97_roots
        t_near = tres * 1.14   # t = 0.8 ms when tres = 0.7 ms
        f_exact = float(fl.exact_pdf(t_near, tres, chme97_roots, chme97_areas, eigvals, g00, g10, g11))
        f_asym  = float(fl.asymptotic_pdf(t_near, tres, tau, chme97_areas))
        # Phase-0: exact=3.782, asymptotic=2.568 -- differ by ~47%
        assert not math.isclose(f_exact, f_asym, rel_tol=0.1), (
            f"exact and asymptotic should differ near tres; got {f_exact:.4f} vs {f_asym:.4f}"
        )

    # -- Regression tests (Phase-0 values) ----------------------------------

    def test_ch82_exact_value_at_200us(self, ch82_step, ch82_roots, ch82_areas, ch82_gamma):
        """Phase-0: exact_pdf(t=0.2 ms) = 3855.84 s^-1  (= asymptotic at this t)."""
        mec1, phi_shut, tres = ch82_step
        eigvals, g00, g10, g11 = ch82_gamma
        f = float(fl.exact_pdf(2e-4, tres, ch82_roots, ch82_areas, eigvals, g00, g10, g11))
        assert f == pytest.approx(3855.84, rel=1e-3)

    def test_ch82_exact_value_at_1ms(self, ch82_step, ch82_roots, ch82_areas, ch82_gamma):
        """Phase-0: exact_pdf(t=1 ms) = 9.8525 s^-1."""
        mec1, phi_shut, tres = ch82_step
        eigvals, g00, g10, g11 = ch82_gamma
        f = float(fl.exact_pdf(1e-3, tres, ch82_roots, ch82_areas, eigvals, g00, g10, g11))
        assert f == pytest.approx(9.8525, rel=1e-3)

    def test_chme97_exact_value_at_1ms(self, chme97_step, chme97_roots, chme97_areas, chme97_gamma):
        """Phase-0: exact_pdf(t=1 ms) = 14.659 s^-1  (differs from asymptotic 14.589)."""
        mec1, phi_shut, tres = chme97_step
        eigvals, g00, g10, g11 = chme97_gamma
        f = float(fl.exact_pdf(1e-3, tres, chme97_roots, chme97_areas, eigvals, g00, g10, g11))
        assert f == pytest.approx(14.659, rel=1e-2)

    def test_chme97_exact_value_at_10ms(self, chme97_step, chme97_roots, chme97_areas, chme97_gamma):
        """Phase-0: exact_pdf(t=10 ms) = 18.114 s^-1  (= asymptotic at this t)."""
        mec1, phi_shut, tres = chme97_step
        eigvals, g00, g10, g11 = chme97_gamma
        f = float(fl.exact_pdf(0.01, tres, chme97_roots, chme97_areas, eigvals, g00, g10, g11))
        assert f == pytest.approx(18.114, rel=1e-3)


# ---------------------------------------------------------------------------
# TestProperties — mathematical properties of the first-latency pdf
# ---------------------------------------------------------------------------

class TestProperties:
    """Mathematical properties that must hold regardless of mechanism."""

    def test_ideal_pdf_integrates_to_one_co(self, co):
        """For CO, ideal first-latency pdf integrates analytically to 1."""
        phi_shut = np.array([1.0])
        eigs, areas = fl.ideal_components(co.QFF, phi_shut)
        # For a sum-of-exponentials, integral = sum(areas * tau) * (rate) normalised
        # = sum(areas) = 1 analytically; verify numerically too
        result, _ = integrate.quad(
            lambda t: float(fl.ideal_pdf(t, co.QFF, phi_shut)),
            0.0, 10.0
        )
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_ideal_pdf_integrates_to_one_ch82(self, ch82_step):
        """Ideal CH82 first-latency pdf integrates to 1 (sum of areas = 1)."""
        mec1, phi_shut, _ = ch82_step
        eigs, areas = fl.ideal_components(mec1.QFF, phi_shut)
        assert areas.sum() == pytest.approx(1.0, abs=1e-7)

    def test_ideal_pdf_integrates_to_one_chme97(self, chme97_step):
        """Ideal CHME97 first-latency pdf integrates to 1."""
        mec1, phi_shut, _ = chme97_step
        eigs, areas = fl.ideal_components(mec1.QFF, phi_shut)
        assert areas.sum() == pytest.approx(1.0, abs=1e-7)

    def test_asymptotic_areas_near_one_ch82(self, ch82_areas):
        """Asymptotic areas sum < 1 but within 2% (missed-events effect)."""
        s = ch82_areas.sum()
        assert 0.98 < s <= 1.0, f"sum(asym_areas) = {s:.6f}; expected in (0.98, 1.0]"

    def test_asymptotic_areas_near_one_chme97(self, chme97_areas):
        """CHME97: asymptotic areas within 0.1% of 1."""
        s = chme97_areas.sum()
        assert 0.999 < s <= 1.0, f"sum(asym_areas) = {s:.6f}; expected > 0.999"

    def test_tres_zero_limit_co(self, co):
        """At tres ~= 0 the asymptotic pdf reduces to the ideal pdf.

        For CO (single shut state), ideal f_L(t) = BETA_CO * exp(-BETA_CO * t).
        At tres = 0 the asymptotic areas must equal the ideal areas (= [1.0])
        and the roots must equal the ideal eigenvalues (= [BETA_CO]).
        """
        phi_shut = np.array([1.0])
        tres = 0.0
        roots = fl.asymptotic_roots(tres, co)
        areas = fl.asymptotic_areas(tres, roots, phi_shut, co)
        eigs, ideal_areas = fl.ideal_components(co.QFF, phi_shut)

        # At tres=0 the single root must equal the single eigenvalue of -QFF
        assert np.abs(roots[0]) == pytest.approx(eigs[0], rel=1e-6)
        # Area must equal ideal area = 1
        assert areas[0] == pytest.approx(ideal_areas[0], rel=1e-6)

    def test_tres_zero_pdf_equals_ideal_co(self, co):
        """At tres=0 exact_pdf(t) == ideal_pdf(t) for all t > 0."""
        phi_shut = np.array([1.0])
        tres = 0.0
        roots = fl.asymptotic_roots(tres, co)
        areas = fl.asymptotic_areas(tres, roots, phi_shut, co)
        eigvals, g00, g10, g11 = fl.gamma_coefficients(tres, phi_shut, co)

        for t in [0.01, 0.05, 0.1, 0.5]:
            f_exact = float(fl.exact_pdf(t, tres, roots, areas, eigvals, g00, g10, g11))
            f_ideal = float(fl.ideal_pdf(t, co.QFF, phi_shut))
            assert f_exact == pytest.approx(f_ideal, rel=1e-5), (
                f"exact ({f_exact:.6e}) != ideal ({f_ideal:.6e}) at t={t}, tres=0"
            )

    def test_exact_agrees_with_asymptotic_for_t_gg_3tres_ch82(
        self, ch82_step, ch82_roots, ch82_areas, ch82_gamma
    ):
        """For t >> 3*tres, exact == asymptotic to five significant figures."""
        mec1, phi_shut, tres = ch82_step
        eigvals, g00, g10, g11 = ch82_gamma
        tau = -1.0 / ch82_roots

        # t = 0.005 s >> 3*tres = 0.3 ms
        for t in [5e-3, 1e-2]:
            f_e = float(fl.exact_pdf(t, tres, ch82_roots, ch82_areas, eigvals, g00, g10, g11))
            f_a = float(fl.asymptotic_pdf(t, tres, tau, ch82_areas))
            assert f_e == pytest.approx(f_a, rel=1e-5)

    def test_ideal_decreasing_ch82(self, ch82_step):
        """For CH82, ideal f_L(t) decreases monotonically for t > 0.

        This holds because the mechanism has no anti-mode (the shut-state
        pdf, starting from the unliganded state, is monotone decreasing).
        """
        mec1, phi_shut, _ = ch82_step
        times = [1e-5, 1e-4, 3e-4, 1e-3, 5e-3]
        vals = [float(fl.ideal_pdf(t, mec1.QFF, phi_shut)) for t in times]
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i + 1], (
                f"ideal_pdf not decreasing: f({times[i]}) = {vals[i]:.4e},"
                f" f({times[i+1]}) = {vals[i+1]:.4e}"
            )
