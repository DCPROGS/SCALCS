"""Tests for scalcs.cjumps — concentration jump calculations.

Reference mechanism: CH82 (Colquhoun & Hawkes 1982).

Module structure after refactoring
-----------------------------------
cjumps exposes:

  Pulse dataclasses
  -----------------
  ErfPulse          Realistic jump: erf-shaped rise and fall ('rcj' profile).
  SquarePulse       Ideal square pulse.
  InstExpPulse      Instantaneous rise, exponential decay.
  PairedSquarePulse Two square pulses separated by an interval.

  Result dataclasses
  ------------------
  JumpResult        t, c, Popen, P — time course from solve().
                    Supports 4-tuple unpacking for backward compatibility.
  RelaxationResult  Analytical on/off time constants and amplitudes.

  Public functions
  ----------------
  solve(mec, pulse, reclen, step, method='ode')
      Solve macroscopic response by ODE integration (default) or step-wise
      Q-matrix calculation.  Returns JumpResult.

  relaxation_taus(mec, pulse)
      Analytical on/off relaxation for a SquarePulse.
      Returns RelaxationResult (fixes the broken 2-vs-4 return bug in the
      old weighted_taus()).

  jump_summary(mec, pulse, gamma, Vm)
      Compute analytical jump properties as a plain dict — no formatting.
      gamma and Vm are explicit parameters, not hard-coded.

  printout(mec, pulse, gamma, Vm)
      Human-readable report built from jump_summary().

Test strategy
-------------
* Pulse profile tests: shape, scalar input, boundary values.
* Solver tests: shapes, Popen in [0,1], occupancy sum, initial condition,
  ODE vs matrix agreement, JumpResult tuple unpacking.
* RelaxationResult tests: signs, component count, fix for sccurves bug.
* jump_summary tests: key presence, current scaling, tau consistency.
* printout tests: smoke test and configurable biophysics.

All tests in this file are written BEFORE the refactored cjumps.py exists
(red phase of TDD).  They define the contract the implementation must meet.
"""

import math
import numpy as np
import pytest

from scalcs import qmatlib as qml
from scalcs import cjumps
from scalcs.samples.samples import CH82


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mec():
    """CH82 mechanism — shared across all tests in this module."""
    return CH82()


@pytest.fixture(scope="module")
def erf_pulse():
    """Typical realistic concentration jump for CH82: 1 µM peak, 10 ms pulse."""
    return cjumps.ErfPulse(
        cmax=1e-6,
        width=10e-3,
        cb=0.0,
        centre=10e-3,
        rise=200e-6,
        decay=200e-6,
    )


@pytest.fixture(scope="module")
def square_pulse():
    """Square pulse: 1 µM, 10 ms width."""
    return cjumps.SquarePulse(cmax=1e-6, width=10e-3, cb=0.0, prepulse=5e-3)


@pytest.fixture(scope="module")
def instexp_pulse():
    """Instantaneous rise + exponential decay: 10 µM, tdec = 2.5 ms."""
    return cjumps.InstExpPulse(cmax=10e-6, tdec=2.5e-3, cb=0.0, prepulse=5e-3)


@pytest.fixture(scope="module")
def paired_pulse():
    """Paired square pulses: 1 µM, 10 ms each, 10 ms apart."""
    return cjumps.PairedSquarePulse(
        cmax=1e-6, width=10e-3, inter=10e-3, cb=0.0, prepulse=5e-3
    )


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

RECLEN = 50e-3   # 50 ms recording
STEP   = 5e-6    # 5 µs sampling interval
N_SAMPLES = int(RECLEN / STEP)   # expected number of time points


# ===========================================================================
# 1. Pulse profile dataclasses
# ===========================================================================

class TestErfPulse:
    """ErfPulse — realistic concentration jump with erf-shaped rise and fall."""

    def test_profile_array_shape(self, erf_pulse):
        t = np.linspace(0, RECLEN, N_SAMPLES)
        c = erf_pulse.profile(t)
        assert c.shape == t.shape

    def test_profile_scalar_input(self, erf_pulse):
        """profile() must accept a scalar time and return a scalar (or 0-d array)."""
        c = erf_pulse.profile(0.0)
        assert np.ndim(c) == 0 or np.isscalar(c)

    def test_profile_baseline_before_pulse(self, erf_pulse):
        """Concentration at t=0 should be negligibly close to cb (=0)."""
        c0 = float(erf_pulse.profile(0.0))
        assert abs(c0 - erf_pulse.cb) < 1e-12

    def test_profile_peak_near_cmax(self, erf_pulse):
        """Peak concentration should be close to cmax."""
        t = np.linspace(0, RECLEN, 10_000)
        assert float(erf_pulse.profile(t).max()) == pytest.approx(erf_pulse.cmax, rel=0.01)

    def test_profile_non_negative(self, erf_pulse):
        t = np.linspace(0, RECLEN, N_SAMPLES)
        assert np.all(erf_pulse.profile(t) >= -1e-15)

    def test_default_background_zero(self):
        p = cjumps.ErfPulse(cmax=1e-6, width=10e-3)
        assert p.cb == 0.0

    def test_default_rise_decay(self):
        p = cjumps.ErfPulse(cmax=1e-6, width=10e-3)
        assert p.rise == pytest.approx(200e-6)
        assert p.decay == pytest.approx(200e-6)

    def test_default_centre(self):
        p = cjumps.ErfPulse(cmax=1e-6, width=10e-3)
        assert p.centre == pytest.approx(10e-3)

    def test_invalid_cmax_raises(self):
        """cmax must be positive."""
        with pytest.raises((ValueError, AssertionError)):
            cjumps.ErfPulse(cmax=-1e-6, width=10e-3)

    def test_invalid_width_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            cjumps.ErfPulse(cmax=1e-6, width=0.0)


class TestSquarePulse:
    """SquarePulse — ideal square concentration step."""

    def test_profile_array_shape(self, square_pulse):
        t = np.linspace(0, RECLEN, N_SAMPLES)
        assert square_pulse.profile(t).shape == t.shape

    def test_profile_scalar_input(self, square_pulse):
        c = square_pulse.profile(0.0)
        assert np.ndim(c) == 0 or np.isscalar(c)

    def test_profile_zero_before_prepulse(self, square_pulse):
        """Before the pulse the concentration must equal cb."""
        c = float(square_pulse.profile(square_pulse.prepulse * 0.5))
        assert abs(c - square_pulse.cb) < 1e-15

    def test_profile_cmax_during_pulse(self, square_pulse):
        """During the pulse concentration must equal cmax."""
        t_mid = square_pulse.prepulse + square_pulse.width * 0.5
        c = float(square_pulse.profile(t_mid))
        assert c == pytest.approx(square_pulse.cmax)

    def test_profile_zero_after_pulse(self, square_pulse):
        """After the pulse the concentration must return to cb."""
        t_after = square_pulse.prepulse + square_pulse.width * 1.5
        c = float(square_pulse.profile(t_after))
        assert abs(c - square_pulse.cb) < 1e-15

    def test_default_background_zero(self):
        p = cjumps.SquarePulse(cmax=1e-6, width=10e-3)
        assert p.cb == 0.0


class TestInstExpPulse:
    """InstExpPulse — instantaneous rise, exponential decay."""

    def test_profile_array_shape(self, instexp_pulse):
        t = np.linspace(0, RECLEN, N_SAMPLES)
        assert instexp_pulse.profile(t).shape == t.shape

    def test_profile_scalar_input(self, instexp_pulse):
        c = instexp_pulse.profile(0.0)
        assert np.ndim(c) == 0 or np.isscalar(c)

    def test_profile_zero_before_prepulse(self, instexp_pulse):
        c = float(instexp_pulse.profile(instexp_pulse.prepulse * 0.5))
        assert abs(c - instexp_pulse.cb) < 1e-15

    def test_profile_decays_after_prepulse(self, instexp_pulse):
        """Concentration should decrease monotonically after the step."""
        t1 = instexp_pulse.prepulse + instexp_pulse.tdec
        t2 = instexp_pulse.prepulse + instexp_pulse.tdec * 3
        c1 = float(instexp_pulse.profile(t1))
        c2 = float(instexp_pulse.profile(t2))
        assert c1 > c2

    def test_profile_decays_to_one_over_e(self, instexp_pulse):
        """At t = prepulse + tdec, concentration should be cmax/e (above cb)."""
        t_tau = instexp_pulse.prepulse + instexp_pulse.tdec
        c = float(instexp_pulse.profile(t_tau)) - instexp_pulse.cb
        expected = instexp_pulse.cmax / math.e
        assert c == pytest.approx(expected, rel=1e-4)


class TestPairedSquarePulse:
    """PairedSquarePulse — two square pulses with a gap between them."""

    def test_profile_array_shape(self, paired_pulse):
        t = np.linspace(0, RECLEN, N_SAMPLES)
        assert paired_pulse.profile(t).shape == t.shape

    def test_profile_cmax_during_first_pulse(self, paired_pulse):
        t_mid1 = paired_pulse.prepulse + paired_pulse.width * 0.5
        c = float(paired_pulse.profile(t_mid1))
        assert c == pytest.approx(paired_pulse.cmax)

    def test_profile_baseline_between_pulses(self, paired_pulse):
        t_gap = paired_pulse.prepulse + paired_pulse.width + paired_pulse.inter * 0.5
        c = float(paired_pulse.profile(t_gap))
        assert abs(c - paired_pulse.cb) < 1e-15

    def test_profile_cmax_during_second_pulse(self, paired_pulse):
        t_mid2 = (paired_pulse.prepulse + paired_pulse.width
                  + paired_pulse.inter + paired_pulse.width * 0.5)
        c = float(paired_pulse.profile(t_mid2))
        assert c == pytest.approx(paired_pulse.cmax)


# ===========================================================================
# 2. solve() — macroscopic response
# ===========================================================================

class TestSolveODE:
    """solve() with method='ode' (default) — scipy.odeint integration."""

    @pytest.fixture(scope="class")
    def result_erf(self, mec, erf_pulse):
        return cjumps.solve(mec, erf_pulse, RECLEN, STEP)

    @pytest.fixture(scope="class")
    def result_square(self, mec, square_pulse):
        return cjumps.solve(mec, square_pulse, RECLEN, STEP)

    @pytest.fixture(scope="class")
    def result_instexp(self, mec, instexp_pulse):
        return cjumps.solve(mec, instexp_pulse, RECLEN, STEP)

    # ---- shapes ----

    def test_t_shape(self, result_erf):
        assert result_erf.t.shape == (N_SAMPLES,)

    def test_c_shape(self, result_erf):
        assert result_erf.c.shape == (N_SAMPLES,)

    def test_Popen_shape(self, result_erf):
        assert result_erf.Popen.shape == (N_SAMPLES,)

    def test_P_shape(self, mec, result_erf):
        assert result_erf.P.shape == (mec.k, N_SAMPLES)

    # ---- physical constraints ----

    def test_Popen_in_unit_interval(self, result_erf):
        assert np.all(result_erf.Popen >= -1e-10)
        assert np.all(result_erf.Popen <= 1.0 + 1e-10)

    def test_occupancy_sum_unity(self, result_erf):
        """Sum of all state occupancies must equal 1 at every time point."""
        col_sums = result_erf.P.sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-6)

    def test_Popen_equals_open_state_sum(self, mec, result_erf):
        """Popen must equal sum of open-state (kA) occupancies."""
        Popen_from_P = result_erf.P[:mec.kA].sum(axis=0)
        assert np.allclose(result_erf.Popen, Popen_from_P, atol=1e-10)

    def test_initial_condition_at_cb_equilibrium(self, mec, erf_pulse, result_erf):
        """At t=0 the system must be at equilibrium for concentration cb."""
        mec.set_eff('c', erf_pulse.cb)
        pinf = qml.pinf(mec.Q)
        assert np.allclose(result_erf.P[:, 0], pinf, atol=1e-6)

    def test_Popen_rises_during_pulse(self, result_erf):
        """Popen must exceed baseline during the concentration pulse."""
        assert result_erf.Popen.max() > result_erf.Popen[0] + 1e-8

    # ---- concentration profile stored ----

    def test_c_matches_pulse_profile(self, erf_pulse, result_erf):
        """c array stored in JumpResult must equal pulse.profile(t)."""
        c_expected = erf_pulse.profile(result_erf.t)
        assert np.allclose(result_erf.c, c_expected, atol=1e-15)

    def test_pulse_stored_in_result(self, erf_pulse, result_erf):
        """JumpResult should carry a reference to the pulse used."""
        assert result_erf.pulse is erf_pulse

    # ---- works for all pulse types ----

    def test_solve_instexp(self, result_instexp):
        assert result_instexp.Popen.max() > result_instexp.Popen[0]

    def test_solve_square(self, result_square):
        assert result_square.Popen.max() > result_square.Popen[0]

    def test_solve_paired(self, mec, paired_pulse):
        result = cjumps.solve(mec, paired_pulse, RECLEN, STEP)
        assert result.Popen.shape == (N_SAMPLES,)


class TestSolveMatrix:
    """solve() with method='matrix' — step-wise Q-matrix calculation."""

    @pytest.fixture(scope="class")
    def result_matrix(self, mec, instexp_pulse):
        return cjumps.solve(mec, instexp_pulse, RECLEN, STEP, method='matrix')

    def test_Popen_in_unit_interval(self, result_matrix):
        assert np.all(result_matrix.Popen >= -1e-10)
        assert np.all(result_matrix.Popen <= 1.0 + 1e-10)

    def test_occupancy_sum_unity(self, result_matrix):
        col_sums = result_matrix.P.sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-5)


class TestSolveODEvsMatrix:
    """ODE and matrix methods must agree to within numerical tolerance."""

    @pytest.fixture(scope="class")
    def both(self, mec, instexp_pulse):
        r_ode    = cjumps.solve(mec, instexp_pulse, RECLEN, STEP, method='ode')
        r_matrix = cjumps.solve(mec, instexp_pulse, RECLEN, STEP, method='matrix')
        return r_ode, r_matrix

    def test_maxPopen_agrees(self, both):
        r_ode, r_matrix = both
        assert r_ode.Popen.max() == pytest.approx(r_matrix.Popen.max(), rel=1e-2)

    def test_popen_profiles_agree(self, both):
        """Point-wise agreement within 1% across the full trace."""
        r_ode, r_matrix = both
        # Skip first few samples where initial transient may differ
        assert np.allclose(r_ode.Popen[5:], r_matrix.Popen[5:], atol=1e-3)


class TestJumpResultUnpacking:
    """JumpResult must support 4-tuple unpacking for backward compatibility."""

    def test_tuple_unpacking(self, mec, erf_pulse):
        result = cjumps.solve(mec, erf_pulse, RECLEN, STEP)
        t, c, Popen, P = result
        assert t.shape == (N_SAMPLES,)
        assert c.shape == (N_SAMPLES,)
        assert Popen.shape == (N_SAMPLES,)
        assert P.ndim == 2

    def test_invalid_method_raises(self, mec, erf_pulse):
        with pytest.raises((ValueError, NotImplementedError)):
            cjumps.solve(mec, erf_pulse, RECLEN, STEP, method='bogus')


# ===========================================================================
# 3. relaxation_taus() — analytical on/off time constants
# ===========================================================================

class TestRelaxationTaus:
    """relaxation_taus() returns a RelaxationResult for a SquarePulse.

    This replaces the broken weighted_taus() which claimed to return 4 values
    but actually returned 2, breaking sccurves.conc_jump_on_off_taus_versus_conc_plot.
    """

    @pytest.fixture(scope="class")
    def result(self, mec, square_pulse):
        return cjumps.relaxation_taus(mec, square_pulse)

    # ---- weighted scalars ----

    def test_tau_on_weighted_positive(self, result):
        assert result.tau_on_weighted > 0.0

    def test_tau_off_weighted_positive(self, result):
        assert result.tau_off_weighted > 0.0

    def test_tau_on_and_off_both_positive(self, result):
        """Both weighted time constants must be positive regardless of mechanism
        or concentration.  The relative ordering of tau_on vs tau_off depends on
        the mechanism and concentration and is not universally predictable."""
        assert result.tau_on_weighted > 0.0
        assert result.tau_off_weighted > 0.0

    # ---- component arrays ----

    def test_tau_on_array_length(self, mec, result):
        """Individual on time constants: k-1 components."""
        assert len(result.tau_on) == mec.k - 1

    def test_tau_off_array_length(self, mec, result):
        assert len(result.tau_off) == mec.k - 1

    def test_tau_on_all_positive(self, result):
        assert np.all(result.tau_on > 0)

    def test_tau_off_all_positive(self, result):
        assert np.all(result.tau_off > 0)

    # ---- amplitude arrays ----

    def test_ampl_on_length(self, mec, result):
        assert len(result.ampl_on) == mec.k - 1

    def test_ampl_off_length(self, mec, result):
        assert len(result.ampl_off) == mec.k - 1

    # ---- sccurves compatibility: can unpack all four arrays ----

    def test_four_array_unpack(self, mec, square_pulse):
        """Simulate sccurves usage: unpack tau_on_weighted, tau_on,
        tau_off_weighted, tau_off — this must NOT crash and all must be valid."""
        result = cjumps.relaxation_taus(mec, square_pulse)
        wton  = result.tau_on_weighted
        ton   = result.tau_on
        wtoff = result.tau_off_weighted
        toff  = result.tau_off
        assert wton > 0
        assert wtoff > 0
        assert len(ton) == mec.k - 1
        assert len(toff) == mec.k - 1

    # ---- wrong pulse type ----

    def test_raises_for_erf_pulse(self, mec, erf_pulse):
        """relaxation_taus() is only valid for a SquarePulse (ideal step)."""
        with pytest.raises((TypeError, ValueError)):
            cjumps.relaxation_taus(mec, erf_pulse)


# ===========================================================================
# 4. jump_summary() — analytical properties as a dict
# ===========================================================================

class TestJumpSummary:
    """jump_summary() returns a plain dict of analytical jump properties.

    All physics is computable and testable here, independently of formatting.
    """

    @pytest.fixture(scope="class")
    def summary(self, mec, square_pulse):
        return cjumps.jump_summary(mec, square_pulse)

    # ---- required keys ----

    def test_has_tau_on_weighted(self, summary):
        assert 'tau_on_weighted' in summary

    def test_has_tau_off_weighted(self, summary):
        assert 'tau_off_weighted' in summary

    def test_has_eigenvalues_on(self, summary):
        assert 'eigenvalues_on' in summary

    def test_has_eigenvalues_off(self, summary):
        assert 'eigenvalues_off' in summary

    def test_has_amplitudes_on(self, summary):
        assert 'amplitudes_on' in summary

    def test_has_amplitudes_off(self, summary):
        assert 'amplitudes_off' in summary

    def test_has_areas_on(self, summary):
        assert 'areas_on' in summary

    def test_has_areas_off(self, summary):
        assert 'areas_off' in summary

    def test_has_p0(self, summary):
        """Equilibrium occupancies before pulse."""
        assert 'p0' in summary

    def test_has_pinf(self, summary):
        """Equilibrium occupancies at peak concentration."""
        assert 'pinf' in summary

    def test_has_pt(self, summary):
        """Occupancies at end of pulse."""
        assert 'pt' in summary

    def test_has_gamma(self, summary):
        assert 'gamma' in summary

    def test_has_Vm(self, summary):
        assert 'Vm' in summary

    # ---- physical constraints ----

    def test_p0_sums_to_one(self, summary):
        assert float(np.sum(summary['p0'])) == pytest.approx(1.0, abs=1e-8)

    def test_pinf_sums_to_one(self, summary):
        assert float(np.sum(summary['pinf'])) == pytest.approx(1.0, abs=1e-8)

    def test_tau_on_positive(self, summary):
        assert summary['tau_on_weighted'] > 0

    def test_tau_off_positive(self, summary):
        assert summary['tau_off_weighted'] > 0

    # ---- configurable biophysics ----

    def test_gamma_stored(self, mec, square_pulse):
        s = cjumps.jump_summary(mec, square_pulse, gamma=50e-12, Vm=-80e-3)
        assert s['gamma'] == pytest.approx(50e-12)

    def test_Vm_stored(self, mec, square_pulse):
        s = cjumps.jump_summary(mec, square_pulse, gamma=30e-12, Vm=-60e-3)
        assert s['Vm'] == pytest.approx(-60e-3)

    def test_current_scales_with_gamma(self, mec, square_pulse):
        """Doubling gamma must double all currents."""
        s1 = cjumps.jump_summary(mec, square_pulse, gamma=30e-12, Vm=-80e-3)
        s2 = cjumps.jump_summary(mec, square_pulse, gamma=60e-12, Vm=-80e-3)
        # amplitudes_on are currents: I = ampl * gamma * Vm
        ratio = np.array(s2['amplitudes_on']) / np.array(s1['amplitudes_on'])
        assert np.allclose(ratio, 2.0, rtol=1e-6)

    def test_current_scales_with_Vm(self, mec, square_pulse):
        """Halving Vm must halve all currents."""
        s1 = cjumps.jump_summary(mec, square_pulse, gamma=30e-12, Vm=-80e-3)
        s2 = cjumps.jump_summary(mec, square_pulse, gamma=30e-12, Vm=-40e-3)
        ratio = np.array(s2['amplitudes_on']) / np.array(s1['amplitudes_on'])
        assert np.allclose(ratio, 0.5, rtol=1e-6)

    def test_default_gamma_and_Vm(self, mec, square_pulse):
        """Default gamma=30 pS, Vm=-80 mV must be used when not specified."""
        s = cjumps.jump_summary(mec, square_pulse)
        assert s['gamma'] == pytest.approx(30e-12)
        assert s['Vm'] == pytest.approx(-80e-3)


# ===========================================================================
# 5. printout() — formatted report
# ===========================================================================

class TestPrintout:
    """printout() returns a formatted string summary of jump properties."""

    @pytest.fixture(scope="class")
    def text(self, mec, square_pulse):
        return cjumps.printout(mec, square_pulse)

    def test_returns_string(self, text):
        assert isinstance(text, str)

    def test_non_empty(self, text):
        assert len(text) > 50

    def test_contains_concentration(self, mec, square_pulse, text):
        """The peak concentration should appear in the output."""
        cmax_mM = square_pulse.cmax * 1000
        assert str(round(cmax_mM, 3)) in text or '{:.5g}'.format(cmax_mM) in text

    def test_contains_on_relaxation_header(self, text):
        assert 'ON' in text.upper()

    def test_contains_off_relaxation_header(self, text):
        assert 'OFF' in text.upper()

    def test_custom_gamma_changes_current_values(self, mec, square_pulse):
        """Changing gamma must produce different current numbers in the output."""
        text1 = cjumps.printout(mec, square_pulse, gamma=30e-12, Vm=-80e-3)
        text2 = cjumps.printout(mec, square_pulse, gamma=60e-12, Vm=-80e-3)
        assert text1 != text2

    def test_custom_Vm_changes_current_values(self, mec, square_pulse):
        text1 = cjumps.printout(mec, square_pulse, gamma=30e-12, Vm=-80e-3)
        text2 = cjumps.printout(mec, square_pulse, gamma=30e-12, Vm=-40e-3)
        assert text1 != text2

    def test_raises_for_erf_pulse(self, mec, erf_pulse):
        """printout() is only defined for a SquarePulse."""
        with pytest.raises((TypeError, ValueError)):
            cjumps.printout(mec, erf_pulse)
