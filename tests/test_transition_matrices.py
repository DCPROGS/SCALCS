"""Tests for the embedded-chain and transition-frequency matrices.

Both quantities had grown three separate implementations - two in
``scalcslib`` and one in ``scsim`` - which is why they are consolidated in
``qmatlib`` and re-exported from the old homes.

Reference mechanism: CH82 at 100 nM.
"""
import numpy as np
import pytest

from scalcs import qmatlib as qml
from scalcs import scalcslib, scsim
from scalcs.samples.samples import CH82


@pytest.fixture(scope="module")
def ch82():
    mec = CH82()
    mec.set_eff('c', 100e-9)
    return mec


@pytest.fixture(scope="module")
def Q(ch82):
    return np.asarray(ch82.Q, dtype=float).copy()


class TestTransitionProbability:
    r"""pi_ij = -q_ij / q_ii: given that the channel leaves state i, the
    probability that the jump is to j."""

    def test_rows_sum_to_one(self, Q):
        pi = qml.transition_probability(Q)
        assert pi.sum(axis=1) == pytest.approx(np.ones(Q.shape[0]))

    def test_diagonal_is_zero(self, Q):
        pi = qml.transition_probability(Q)
        assert np.diag(pi) == pytest.approx(np.zeros(Q.shape[0]))

    def test_matches_the_definition(self, Q):
        pi = qml.transition_probability(Q)
        for i in range(Q.shape[0]):
            for j in range(Q.shape[0]):
                if i != j:
                    assert pi[i, j] == pytest.approx(-Q[i, j] / Q[i, i])

    def test_an_integer_q_matrix_is_not_truncated(self):
        """A Q of integer dtype must not do integer division.

        Contrived, but it is the one way these functions could return
        silently wrong numbers rather than raising.
        """
        Qint = np.array([[-3, 3], [1, -1]])
        pi = qml.transition_probability(Qint)
        assert pi[0, 1] == pytest.approx(1.0)
        assert pi[1, 0] == pytest.approx(1.0)

    def test_does_not_modify_its_argument(self, Q):
        before = Q.copy()
        qml.transition_probability(Q)
        assert Q == pytest.approx(before)


class TestTransitionFrequency:
    """f_ij = p_i q_ij: transitions from i to j per second."""

    def test_defaults_to_equilibrium_occupancies(self, Q):
        assert qml.transition_frequency(Q) == pytest.approx(
            qml.transition_frequency(Q, qml.pinf(Q)))

    def test_accepts_occupancies_that_are_not_at_equilibrium(self, Q):
        """The reason for taking p as an argument: during a concentration
        jump the occupancies are not the equilibrium ones."""
        p = np.zeros(Q.shape[0])
        p[-1] = 1.0                      # everything in the last state
        f = qml.transition_frequency(Q, p)
        assert f[:-1, :] == pytest.approx(np.zeros_like(f[:-1, :]))
        assert f[-1].sum() == pytest.approx(-Q[-1, -1])

    def test_diagonal_is_zero(self, Q):
        assert np.diag(qml.transition_frequency(Q)) == pytest.approx(
            np.zeros(Q.shape[0]))

    def test_detailed_balance_holds_at_equilibrium(self, Q):
        """f_ij = f_ji edge by edge, which is what microscopic reversibility
        means. Not exact here: CH82's MR-constrained rate is stored rounded
        to 0.66667, so its cycle closes only to about 1e-9.
        """
        f = qml.transition_frequency(Q)
        assert np.abs(f - f.T).max() / np.abs(f).max() < 1e-8

    def test_does_not_modify_its_argument(self, Q):
        before = Q.copy()
        qml.transition_frequency(Q)
        assert Q == pytest.approx(before)


class TestOldHomesStillWork:
    """scalcslib and scsim keep their names, forwarding to qmatlib, so that
    existing notebooks and scripts do not break."""

    def test_scalcslib_forwards_both(self, Q):
        assert scalcslib.transition_probability(Q) == pytest.approx(
            qml.transition_probability(Q))
        assert scalcslib.transition_frequency(Q) == pytest.approx(
            qml.transition_frequency(Q))

    def test_scsim_forwards_the_probability_matrix(self, Q):
        assert scsim.transition_probability(Q) == pytest.approx(
            qml.transition_probability(Q))

    def test_scalcslib_transition_frequency_still_takes_one_argument(self, Q):
        """Its old signature computed pinf internally; that must still be
        what a single-argument call gives."""
        assert scalcslib.transition_frequency(Q) == pytest.approx(
            qml.transition_frequency(Q, qml.pinf(Q)))
