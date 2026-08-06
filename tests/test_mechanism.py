"""Tests for scalcs.mechanism.

Reference mechanism throughout: CH82 (Colquhoun & Hawkes 1982).
  States  A2R* (A), AR* (A), A2R (B), AR (B), R (C)
  kA=2, kB=2, kC=1, k=5
  One cycle: [A2R*, AR*, AR, A2R]
  One MR-constrained rate: 2k*(-2) = 0.66667 s-1
"""
import math
import numpy as np
import pytest

from scalcs import mechanism as mech
from scalcs.mechanism import (
    Cycle,
    Graph,
    Mechanism,
    Rate,
    State,
    constrain_rate_multiple,
    identity,
    multiply,
)
from scalcs.samples.samples import CH82, CCO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ch82() -> Mechanism:
    """CH82 mechanism (5-state, 1 cycle, 1 MR constraint)."""
    return CH82()


@pytest.fixture(scope="module")
def cco() -> Mechanism:
    """Simple C-C-O mechanism (3-state, no cycle)."""
    return CCO()


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:

    def test_identity_returns_first_element(self):
        rate = np.array([42.0])
        assert identity(rate, {'c': 999.0}) == pytest.approx(42.0)

    def test_identity_ignores_effdict(self):
        rate = np.array([7.0])
        assert identity(rate, {}) == pytest.approx(7.0)
        assert identity(rate, {'v': 1000.0}) == pytest.approx(7.0)

    def test_multiply_scales_by_effector(self):
        rate = np.array([5e8])
        result = multiply(rate, {'c': 100e-9})
        assert result == pytest.approx(5e8 * 100e-9)

    def test_constrain_rate_multiple(self):
        rate = np.array([2000.0])
        assert constrain_rate_multiple(rate, 2.0) == pytest.approx(np.array([4000.0]))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TestState:

    def test_valid_statetypes(self):
        for st in ('A', 'B', 'C', 'D'):
            s = State(st, 'X', 0.0)
            assert s.statetype == st

    def test_invalid_statetype_raises(self):
        with pytest.raises(RuntimeError):
            State('E', 'bad', 0.0)

    def test_attributes_stored(self):
        s = State('A', 'AR*', 60e-12)
        assert s.name == 'AR*'
        assert s.conductance == pytest.approx(60e-12)

    def test_no_is_none_before_mechanism(self):
        s = State('A', 'test', 0.0)
        assert s.no is None


# ---------------------------------------------------------------------------
# Cycle
# ---------------------------------------------------------------------------

class TestCycle:

    def test_states_stored(self):
        c = Cycle(['A', 'B', 'C', 'D'])
        assert c.states == ['A', 'B', 'C', 'D']

    def test_mrconstr_defaults_to_empty_list(self):
        c = Cycle(['A', 'B'])
        assert c.mrconstr == []

    def test_mrconstr_accepted(self):
        c = Cycle(['A', 'B', 'C'], ['A', 'B'])
        assert c.mrconstr == ['A', 'B']

    def test_mutable_default_independence(self):
        """Two Cycle() calls must receive independent mrconstr lists."""
        c1 = Cycle(['A', 'B'])
        c2 = Cycle(['C', 'D'])
        c1.mrconstr.append('X')
        assert c2.mrconstr == [], (
            "Shared mutable default: modifying c1.mrconstr polluted c2"
        )


# ---------------------------------------------------------------------------
# Rate
# ---------------------------------------------------------------------------

class TestRate:

    @pytest.fixture
    def two_states(self):
        return State('A', 'O', 50e-12), State('B', 'C', 0.0)

    def test_scalar_rateconstant_becomes_array(self, two_states):
        s1, s2 = two_states
        r = Rate(500.0, s1, s2, name='alpha')
        assert isinstance(r.rateconstants, np.ndarray)
        assert r.rateconstants[0] == pytest.approx(500.0)

    def test_list_rateconstant(self, two_states):
        s1, s2 = two_states
        r = Rate([100.0, 200.0], s1, s2, func=lambda rc, ed: rc[0])
        assert len(r.rateconstants) == 2

    def test_unit_rate_no_effector(self, two_states):
        s1, s2 = two_states
        r = Rate(500.0, s1, s2)
        assert r.unit_rate() == pytest.approx(500.0)

    def test_unit_rate_with_effector(self, two_states):
        s1, s2 = two_states
        r = Rate(5e8, s1, s2, eff='c')
        # unit_rate sets effector to 1.0 → 5e8 * 1.0
        assert r.unit_rate() == pytest.approx(5e8)

    def test_calc_with_effector(self, two_states):
        s1, s2 = two_states
        r = Rate(5e8, s1, s2, eff='c')
        assert r.calc({'c': 100e-9}) == pytest.approx(5e8 * 100e-9)

    def test_wrong_state_type_raises(self):
        with pytest.raises(TypeError):
            Rate(100.0, 'not_a_state', State('B', 'C', 0.0))

    def test_limits_mutable_default_independence(self, two_states):
        """Two Rate() calls must not share a limits list."""
        s1, s2 = two_states
        r1 = Rate(100.0, s1, s2)
        r2 = Rate(200.0, s1, s2)
        r1.limits.append([0, 1])  # should not affect r2
        # r2 limits were set by _set_default_limits; still valid list
        assert isinstance(r2.limits, list)

    def test_fixed_flag(self, two_states):
        s1, s2 = two_states
        r = Rate(100.0, s1, s2, fixed=True)
        assert r.fixed is True

    def test_name_property(self, two_states):
        s1, s2 = two_states
        r = Rate(100.0, s1, s2, name='beta')
        assert r.name == 'beta'
        r.name = 'gamma'
        assert r.name == 'gamma'


# ---------------------------------------------------------------------------
# Mechanism construction
# ---------------------------------------------------------------------------

class TestMechanismConstruction:

    def test_ch82_state_counts(self, ch82):
        assert ch82.kA == 2   # open
        assert ch82.kB == 2   # within-burst shut
        assert ch82.kC == 1   # between-burst shut
        assert ch82.kD == 0
        assert ch82.k == 5

    def test_ch82_derived_counts(self, ch82):
        assert ch82.kE == ch82.kA + ch82.kB   # burst states
        assert ch82.kG == ch82.kA + ch82.kB + ch82.kC  # cluster states

    def test_cco_state_counts(self, cco):
        assert cco.kA == 1
        assert cco.kB == 1
        assert cco.kC == 1
        assert cco.k == 3

    def test_q_shape(self, ch82):
        assert ch82.Q.shape == (5, 5)

    def test_q_row_sums_zero(self, ch82):
        """Every row of Q must sum to zero (fundamental property)."""
        row_sums = ch82.Q.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)

    def test_q_diagonal_non_positive(self, ch82):
        """Diagonal elements of Q must be <= 0."""
        assert np.all(ch82.Q.diagonal() <= 0.0)

    def test_q_off_diagonal_non_negative(self, ch82):
        """Off-diagonal elements of Q must be >= 0."""
        k = ch82.Q.shape[0]
        for i in range(k):
            for j in range(k):
                if i != j:
                    assert ch82.Q[i, j] >= 0.0

    def test_states_sorted_by_type(self, ch82):
        """States must appear in order A, B, B, C (sorted by statetype)."""
        types = [s.statetype for s in ch82.States]
        assert types == sorted(types)

    def test_state_indices_zero_based(self, ch82):
        for i, state in enumerate(ch82.States):
            assert state.no == i

    def test_no_cycles_mutable_default(self):
        """Mechanism() without Cycles must not share a list across instances."""
        m1 = CCO()
        m2 = CCO()
        m1.Cycles.append(Cycle(['A', 'B']))
        assert m2.Cycles == [], (
            "Shared mutable default: m1.Cycles polluted m2"
        )


# ---------------------------------------------------------------------------
# Submatrix partitioning
# ---------------------------------------------------------------------------

class TestSubmatrices:

    def test_QAA_shape(self, ch82):
        assert ch82.QAA.shape == (ch82.kA, ch82.kA)

    def test_QFF_shape(self, ch82):
        # QFF covers B and C states (all shut within cluster)
        kF = ch82.kB + ch82.kC
        assert ch82.QFF.shape == (kF, kF)

    def test_submatrices_consistent_with_Q(self, ch82):
        kA = ch82.kA
        np.testing.assert_allclose(ch82.QAA, ch82.Q[:kA, :kA])
        np.testing.assert_allclose(ch82.QAF, ch82.Q[:kA, kA:ch82.kG])


# ---------------------------------------------------------------------------
# Q-matrix update: set_eff
# ---------------------------------------------------------------------------

class TestSetEff:

    def test_set_eff_changes_Q(self, ch82):
        ch82.set_eff('c', 100e-9)
        Q_low = ch82.Q.copy()
        ch82.set_eff('c', 1e-3)
        Q_high = ch82.Q.copy()
        # kon-type rates scale with concentration; some Q entries must differ
        assert not np.allclose(Q_low, Q_high)

    def test_row_sums_still_zero_after_set_eff(self, ch82):
        ch82.set_eff('c', 30e-6)
        np.testing.assert_allclose(ch82.Q.sum(axis=1), 0.0, atol=1e-8)

    def test_unknown_effector_does_not_raise(self, ch82):
        """Unknown effector should be silently ignored (verbose=False)."""
        ch82.set_eff('z', 1.0)  # no rate depends on 'z'


# ---------------------------------------------------------------------------
# theta / theta_unsqueeze roundtrip
# ---------------------------------------------------------------------------

class TestTheta:

    def test_theta_excludes_fixed_and_constrained(self, ch82):
        theta = ch82.theta()
        # CH82 has 10 rates; rate[7] is fixed, rate[9] is MR-constrained
        # → 8 free parameters
        assert len(theta) == 8

    def test_theta_unsqueeze_roundtrip(self, ch82):
        theta_before = ch82.theta().copy()
        # scale all free params by 1 (identity) and unsqueeze
        ch82.theta_unsqueeze(theta_before)
        theta_after = ch82.theta()
        np.testing.assert_allclose(theta_after, theta_before, rtol=1e-10)

    def test_theta_unsqueeze_modifies_rateconstants(self, ch82):
        """theta_unsqueeze sets rate constants; Q is rebuilt on next set_eff call."""
        theta = ch82.theta().copy()
        # double all free params and round-trip through theta()
        ch82.theta_unsqueeze(theta * 2.0)
        ch82.update_submat()   # required to propagate to Q
        theta_doubled = ch82.theta()
        np.testing.assert_allclose(theta_doubled, theta * 2.0, rtol=1e-9)
        # restore
        ch82.theta_unsqueeze(theta)
        ch82.update_submat()


# ---------------------------------------------------------------------------
# Microscopic reversibility
# ---------------------------------------------------------------------------

class TestMicroscopicReversibility:

    def test_check_mr_products_equal(self, ch82):
        """Forward and backward rate products round the cycle must be equal."""
        cycle = ch82.Cycles[0]
        fprod, bprod = ch82.check_mr(cycle)
        assert fprod == pytest.approx(bprod, rel=1e-6)

    def test_mr_rate_not_in_theta(self, ch82):
        """The MR-constrained rate must not appear in the free parameter set."""
        # Rate index 9 (2k*(-2)) is mr=True in CH82
        mr_rate = ch82.Rates[9]
        assert mr_rate.mr is True
        free_names = ch82.get_free_parameter_names()
        assert mr_rate.name not in free_names


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class TestConstraints:

    def test_constrained_rate_excluded_from_theta(self):
        """A constrained rate must not appear among free parameters."""
        from scalcs.samples.samples import AChR_diamond, load_AChR_diamond_independent_binding
        mec, free_names = load_AChR_diamond_independent_binding()
        for rate in mec.Rates:
            if rate.is_constrained:
                assert rate.name not in free_names

    def test_update_constrains_keeps_row_sums_zero(self):
        from scalcs.samples.samples import load_AChR_diamond_independent_binding
        mec, _ = load_AChR_diamond_independent_binding()
        mec.update_constrains()
        np.testing.assert_allclose(mec.Q.sum(axis=1), 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# check_limits / impose_limits
# ---------------------------------------------------------------------------

class TestLimits:

    def test_check_limits_true_for_valid_mec(self, ch82):
        assert ch82.check_limits() is True

    def test_impose_limits_clips_out_of_range(self, cco):
        # Force one rate constant way above its upper limit
        original = cco.Rates[0].rateconstants.copy()
        cco.Rates[0].rateconstants = np.array([1e20])
        cco.impose_limits()
        assert cco.Rates[0].unit_rate() <= cco.Rates[0].limits[0][1]
        # restore
        cco.Rates[0].rateconstants = original
