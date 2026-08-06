"""Tests for scalcs.qmatlib.

Mathematical reference throughout:
  CH82 : Colquhoun D, Hawkes AG (1982) Phil Trans R Soc Lond B 300, 1-59.
  HJC92: Hawkes AG, Jalali A, Colquhoun D (1992) Phil Trans R Soc Lond B
          337, 383-404.
  CHS96: Colquhoun D, Hawkes AG, Srodzinski K (1996) Phil Trans R Soc
          Lond A 354, 2555-2590.

Two reference mechanisms are used:

CO (2-state, analytically tractable)
  States: O (A, open), C (B, shut)
  alpha = 50 s-1 (O→C),  beta = 20 s-1 (C→O)
  Q = [[-50, 50], [20, -20]]
  Eigenvalues: 0, -70
  pinf = [2/7, 5/7]   (O, C)
  expQt[0,0](t) = 2/7 + 5/7 * exp(-70t)
  expQt[0,1](t) = 5/7 * (1 - exp(-70t))
  expQt[1,0](t) = 2/7 * (1 - exp(-70t))
  expQt[1,1](t) = 5/7 + 2/7 * exp(-70t)

CH82 (5-state, used for shape and consistency checks only)
  kA=2, kB=2, kC=1, k=5, one cycle, one MR-constrained rate.
"""

import math
import numpy as np
import pytest
from numpy import linalg as nplin

from scalcs import qmatlib as qml
from scalcs.samples.samples import CH82, CO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def co():
    return CO()

@pytest.fixture(scope="module")
def ch82():
    mec = CH82()
    mec.set_eff('c', 100e-9)   # realistic concentration for burst-mode tests
    return mec

@pytest.fixture(scope="module")
def Q_co(co):
    """Raw Q matrix for the 2-state CO mechanism."""
    return co.Q.copy()

@pytest.fixture(scope="module")
def Q_ch82(ch82):
    return ch82.Q.copy()

# Analytical values for CO
ALPHA = 50.0   # O→C
BETA  = 20.0   # C→O
RATE_SUM = ALPHA + BETA   # 70.0
PINF_O = BETA / RATE_SUM  # 2/7
PINF_C = ALPHA / RATE_SUM  # 5/7


# ---------------------------------------------------------------------------
# eigs
# ---------------------------------------------------------------------------

class TestEigs:

    def test_returns_two_objects(self, Q_co):
        result = qml.eigs(Q_co)
        assert len(result) == 2

    def test_eigvals_shape(self, Q_co):
        eigvals, A = qml.eigs(Q_co)
        assert eigvals.shape == (2,)

    def test_spectral_shape(self, Q_co):
        eigvals, A = qml.eigs(Q_co)
        assert A.shape == (2, 2, 2)

    def test_co_eigenvalues(self, Q_co):
        """CO Q has eigenvalues 0 and -(alpha+beta) = -70."""
        eigvals, _ = qml.eigs(Q_co)
        vals = sorted(eigvals.real)
        assert vals[0] == pytest.approx(-RATE_SUM, rel=1e-10)
        assert vals[1] == pytest.approx(0.0, abs=1e-10)

    def test_spectral_sum_is_identity(self, Q_co):
        """Sum of all spectral matrices must equal the identity (Eq. A3.1, CH95b)."""
        _, A = qml.eigs(Q_co)
        k = Q_co.shape[0]
        np.testing.assert_allclose(A.sum(axis=0), np.eye(k), atol=1e-12)

    def test_spectral_decomposition_reconstructs_Q(self, Q_co):
        """Q = Σ_i λ_i * A_i (spectral decomposition)."""
        eigvals, A = qml.eigs(Q_co)
        Q_reconstructed = np.sum(
            A * eigvals.reshape(-1, 1, 1), axis=0
        )
        np.testing.assert_allclose(Q_reconstructed, Q_co, atol=1e-10)

    def test_spectral_decomposition_ch82(self, Q_ch82):
        eigvals, A = qml.eigs(Q_ch82)
        k = Q_ch82.shape[0]
        np.testing.assert_allclose(A.sum(axis=0), np.eye(k), atol=1e-10)
        Q_rec = np.sum(A * eigvals.reshape(-1, 1, 1), axis=0)
        np.testing.assert_allclose(Q_rec, Q_ch82, atol=1e-8)


class TestEigsSorted:

    def test_eigenvalues_ascending(self, Q_ch82):
        eigvals, _ = qml.eigs_sorted(Q_ch82)
        real_parts = eigvals.real
        assert np.all(np.diff(real_parts) >= 0)

    def test_same_set_as_eigs(self, Q_co):
        ev_unsorted, _ = qml.eigs(Q_co)
        ev_sorted, _   = qml.eigs_sorted(Q_co)
        np.testing.assert_allclose(
            sorted(ev_unsorted.real), sorted(ev_sorted.real), atol=1e-12
        )


# ---------------------------------------------------------------------------
# expQt
# ---------------------------------------------------------------------------

class TestExpQt:

    def test_at_t0_is_identity(self, Q_co):
        expM = qml.expQt(Q_co, 0.0)
        np.testing.assert_allclose(expM, np.eye(2), atol=1e-12)

    def test_semigroup_property(self, Q_co):
        """exp(Q*(t1+t2)) = exp(Q*t1) @ exp(Q*t2)."""
        t1, t2 = 1e-3, 2e-3
        lhs = qml.expQt(Q_co, t1 + t2)
        rhs = qml.expQt(Q_co, t1) @ qml.expQt(Q_co, t2)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_rows_sum_to_one(self, Q_co):
        """Each row of exp(Qt) is a probability vector — must sum to 1."""
        expM = qml.expQt(Q_co, 0.5e-3)
        np.testing.assert_allclose(expM.sum(axis=1), [1.0, 1.0], atol=1e-12)

    def test_entries_non_negative(self, Q_co):
        expM = qml.expQt(Q_co, 0.5e-3)
        assert np.all(expM >= -1e-14)

    def test_analytical_co_diagonal(self, Q_co):
        """Check expQt[0,0] and expQt[1,1] against 2-state analytical solution."""
        t = 0.01  # 10 ms
        expM = qml.expQt(Q_co, t)
        exp70t = math.exp(-RATE_SUM * t)
        expected_00 = PINF_O + PINF_C * exp70t
        expected_11 = PINF_C + PINF_O * exp70t
        assert expM[0, 0] == pytest.approx(expected_00, rel=1e-10)
        assert expM[1, 1] == pytest.approx(expected_11, rel=1e-10)

    def test_analytical_co_off_diagonal(self, Q_co):
        """Check expQt[0,1] and expQt[1,0] against 2-state analytical solution."""
        t = 0.01
        expM = qml.expQt(Q_co, t)
        exp70t = math.exp(-RATE_SUM * t)
        expected_01 = PINF_C * (1 - exp70t)
        expected_10 = PINF_O * (1 - exp70t)
        assert expM[0, 1] == pytest.approx(expected_01, rel=1e-10)
        assert expM[1, 0] == pytest.approx(expected_10, rel=1e-10)

    def test_large_t_converges_to_pinf(self, Q_co):
        """As t → ∞, every row of exp(Qt) → pinf (equilibrium)."""
        expM = qml.expQt(Q_co, 10.0)   # 10 s >> 1/70 s
        for row in expM:
            assert row[0] == pytest.approx(PINF_O, abs=1e-6)
            assert row[1] == pytest.approx(PINF_C, abs=1e-6)

    def test_ch82_at_t0_is_identity(self, Q_ch82):
        expM = qml.expQt(Q_ch82, 0.0)
        np.testing.assert_allclose(expM, np.eye(5), atol=1e-10)

    def test_ch82_semigroup(self, Q_ch82):
        t1, t2 = 1e-4, 3e-4
        lhs = qml.expQt(Q_ch82, t1 + t2)
        rhs = qml.expQt(Q_ch82, t1) @ qml.expQt(Q_ch82, t2)
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)


# ---------------------------------------------------------------------------
# pinf / pinf1
# ---------------------------------------------------------------------------

class TestPinf:

    def test_co_sums_to_one(self, Q_co):
        p = qml.pinf(Q_co)
        assert p.sum() == pytest.approx(1.0, rel=1e-12)

    def test_co_analytical_values(self, Q_co):
        """pinf for CO: [PINF_O, PINF_C] = [2/7, 5/7]."""
        p = qml.pinf(Q_co)
        assert p[0] == pytest.approx(PINF_O, rel=1e-10)
        assert p[1] == pytest.approx(PINF_C, rel=1e-10)

    def test_co_satisfies_balance(self, Q_co):
        """pinf @ Q = 0 (global balance / stationarity)."""
        p = qml.pinf(Q_co)
        np.testing.assert_allclose(p @ Q_co, 0.0, atol=1e-12)

    def test_ch82_sums_to_one(self, Q_ch82):
        p = qml.pinf(Q_ch82)
        assert p.sum() == pytest.approx(1.0, rel=1e-10)

    def test_ch82_non_negative(self, Q_ch82):
        p = qml.pinf(Q_ch82)
        assert np.all(p >= -1e-14)

    def test_ch82_satisfies_balance(self, Q_ch82):
        p = qml.pinf(Q_ch82)
        np.testing.assert_allclose(p @ Q_ch82, 0.0, atol=1e-10)

    def test_pinf1_agrees_with_pinf_co(self, Q_co):
        p1 = qml.pinf(Q_co)
        p2 = qml.pinf1(Q_co)
        np.testing.assert_allclose(p1, p2, atol=1e-10)

    def test_pinf1_agrees_with_pinf_ch82(self, Q_ch82):
        p1 = qml.pinf(Q_ch82)
        p2 = qml.pinf1(Q_ch82)
        np.testing.assert_allclose(p1, p2, atol=1e-8)


# ---------------------------------------------------------------------------
# iGs
# ---------------------------------------------------------------------------

class TestIGs:

    def test_co_shapes(self, co):
        """CO: kA=1, kB=1 → GAB shape (1,1), GBA shape (1,1)."""
        Q = co.Q
        GAB, GBA = qml.iGs(Q, co.kA, co.kB)
        assert GAB.shape == (co.kA, co.kB)
        assert GBA.shape == (co.kB, co.kA)

    def test_co_gab_is_one(self, co):
        """For 2-state CO with only one B class, O can only go to C: GAB = [[1]]."""
        Q = co.Q
        GAB, GBA = qml.iGs(Q, co.kA, co.kB)
        assert GAB[0, 0] == pytest.approx(1.0, rel=1e-10)

    def test_co_gba_is_one(self, co):
        """Symmetrically: GBA = [[1]] for the same reason."""
        Q = co.Q
        GAB, GBA = qml.iGs(Q, co.kA, co.kB)
        assert GBA[0, 0] == pytest.approx(1.0, rel=1e-10)

    def test_co_definition_gab(self, co):
        """GAB = -inv(QAA) @ QAB  (Eq. 1.25, CH82)."""
        Q = co.Q
        QAA = Q[:co.kA, :co.kA]
        QAB = Q[:co.kA, co.kA:co.kA + co.kB]
        GAB_expected = np.dot(nplin.inv(-QAA), QAB)
        GAB, _ = qml.iGs(Q, co.kA, co.kB)
        np.testing.assert_allclose(GAB, GAB_expected, atol=1e-12)

    def test_co_definition_gba(self, co):
        """GBA = -inv(QBB) @ QBA  (Eq. 1.25, CH82)."""
        Q = co.Q
        kA, kB = co.kA, co.kB
        QBB = Q[kA:kA+kB, kA:kA+kB]
        QBA = Q[kA:kA+kB, :kA]
        GBA_expected = np.dot(nplin.inv(-QBB), QBA)
        _, GBA = qml.iGs(Q, kA, kB)
        np.testing.assert_allclose(GBA, GBA_expected, atol=1e-12)

    def test_ch82_shapes(self, ch82):
        Q = ch82.Q
        kA, kB = ch82.kA, ch82.kB
        GAB, GBA = qml.iGs(Q, kA, kB)
        assert GAB.shape == (kA, kB)
        assert GBA.shape == (kB, kA)

    def test_ch82_gba_rows_leq_one(self, ch82):
        """Row sums of GBA must be ≤ 1 (B states can also exit to C)."""
        Q = ch82.Q
        _, GBA = qml.iGs(Q, ch82.kA, ch82.kB)
        row_sums = GBA.sum(axis=1)
        assert np.all(row_sums <= 1.0 + 1e-10)
        assert np.all(row_sums >= 0.0 - 1e-10)

    def test_ch82_gab_rows_leq_one(self, ch82):
        """Row sums of GAB (open → within-burst shut) must be ≤ 1."""
        Q = ch82.Q
        GAB, _ = qml.iGs(Q, ch82.kA, ch82.kB)
        row_sums = GAB.sum(axis=1)
        assert np.all(row_sums <= 1.0 + 1e-10)
        assert np.all(row_sums >= 0.0 - 1e-10)


# ---------------------------------------------------------------------------
# phiA / phiF (initial vectors)
# ---------------------------------------------------------------------------

class TestInitialVectors:

    def test_phiA_sums_to_one(self, ch82):
        phi = qml.phiA(ch82)
        assert phi.sum() == pytest.approx(1.0, rel=1e-10)

    def test_phiA_non_negative(self, ch82):
        phi = qml.phiA(ch82)
        assert np.all(phi >= -1e-12)

    def test_phiA_shape(self, ch82):
        phi = qml.phiA(ch82)
        assert phi.shape == (ch82.kA,)

    def test_phiF_sums_to_one(self, ch82):
        phi = qml.phiF(ch82)
        assert phi.sum() == pytest.approx(1.0, rel=1e-10)

    def test_phiF_non_negative(self, ch82):
        phi = qml.phiF(ch82)
        assert np.all(phi >= -1e-12)

    def test_phiF_shape(self, ch82):
        phi = qml.phiF(ch82)
        assert phi.shape == (ch82.kI,)

    def test_co_phiA_is_one(self, co):
        """CO has kA=1: phiA must be [1.0]."""
        phi = qml.phiA(co)
        assert phi.shape == (1,)
        assert phi[0] == pytest.approx(1.0, rel=1e-10)


# ---------------------------------------------------------------------------
# H, W, detW  (HJC92 functions)
# ---------------------------------------------------------------------------

class TestHWDetW:

    @pytest.fixture(scope="class")
    def ch82_hjc_params(self, ch82):
        """Pre-compute submatrices and tres for HJC tests."""
        Q = ch82.Q
        kA, kG = ch82.kA, ch82.kG
        kF = kG - kA   # = kB + kC
        QAA = Q[:kA, :kA]
        QFF = Q[kA:kG, kA:kG]
        QAF = Q[:kA, kA:kG]
        QFA = Q[kA:kG, :kA]
        tres = 100e-6  # 100 µs dead time
        return dict(QAA=QAA, QFF=QFF, QAF=QAF, QFA=QFA,
                    kA=kA, kF=kF, tres=tres)

    def test_H_shape(self, ch82, ch82_hjc_params):
        p = ch82_hjc_params
        s = 0.0
        Hmat = qml.H(s, p['tres'], p['QAA'], p['QFF'], p['QAF'], p['QFA'], p['kF'])
        assert Hmat.shape == (p['kA'], p['kA'])

    def test_W_shape(self, ch82, ch82_hjc_params):
        p = ch82_hjc_params
        s = 100.0
        Wmat = qml.W(s, p['tres'], p['QAA'], p['QFF'], p['QAF'], p['QFA'],
                     p['kA'], p['kF'])
        assert Wmat.shape == (p['kA'], p['kA'])

    def test_W_equals_sI_minus_H(self, ch82, ch82_hjc_params):
        """W(s) = s*I - H(s)  (Eq. 52, HJC92)."""
        p = ch82_hjc_params
        s = 500.0
        Hmat = qml.H(s, p['tres'], p['QAA'], p['QFF'], p['QAF'], p['QFA'], p['kF'])
        Wmat = qml.W(s, p['tres'], p['QAA'], p['QFF'], p['QAF'], p['QFA'],
                     p['kA'], p['kF'])
        expected = s * np.eye(p['kA']) - Hmat
        np.testing.assert_allclose(Wmat, expected, atol=1e-10)

    def test_detW_consistent_with_W(self, ch82, ch82_hjc_params):
        """detW must equal det(W) computed directly."""
        p = ch82_hjc_params
        s = 200.0
        Wmat = qml.W(s, p['tres'], p['QAA'], p['QFF'], p['QAF'], p['QFA'],
                     p['kA'], p['kF'])
        det_direct = nplin.det(Wmat)
        det_func   = qml.detW(s, p['tres'], p['QAA'], p['QFF'], p['QAF'],
                               p['QFA'], p['kA'], p['kF'])
        assert det_func == pytest.approx(det_direct, rel=1e-10)

    def test_dW_shape(self, ch82, ch82_hjc_params):
        p = ch82_hjc_params
        s = 100.0
        dWmat = qml.dW(s, p['tres'], p['QAF'], p['QFF'], p['QFA'],
                       p['kA'], p['kF'])
        assert dWmat.shape == (p['kA'], p['kA'])


# ---------------------------------------------------------------------------
# iGt  (time-dependent G matrix)
# ---------------------------------------------------------------------------

class TestIGt:

    def test_shape(self, ch82):
        Q = ch82.Q
        kA = ch82.kA
        kG = ch82.kG
        QAA = Q[:kA, :kA]
        QAB = Q[:kA, kA:kG]
        t = 1e-3
        G = qml.iGt(t, QAA, QAB)
        assert G.shape == (kA, kG - kA)

    def test_at_t0_equals_QAB_definition(self, ch82):
        """iGt(t=0) = exp(QAA*0) @ QAB = I @ QAB = QAB  (Eq. 1.20, CH82)."""
        Q = ch82.Q
        kA = ch82.kA
        kG = ch82.kG
        QAA = Q[:kA, :kA]
        QAB = Q[:kA, kA:kG]
        G0 = qml.iGt(0.0, QAA, QAB)
        np.testing.assert_allclose(G0, QAB, atol=1e-10)
