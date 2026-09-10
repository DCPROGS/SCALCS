"""The integral of a matrix exponential, and the pole it exists to survive.

``qmatlib.H`` evaluates

    QAA + QAF X^-1 (I - exp(-X tres)) QFA,      X = sI - QFF

and used to compute ``X^-1`` directly. That breaks down when ``s`` reaches an
eigenvalue of ``QFF`` -- and the asymptotic root search that calls ``H`` walks
``s`` across exactly that range, because the roots of the determinant equation
are *interlaced* with those eigenvalues.

Whether it *raises* or merely returns nonsense depends on whether the
eigenvalue is exactly representable. For a mechanism whose rates are round
numbers -- a rate sitting at a limit of 1e6, say -- it lands on the pole
exactly and ``LinAlgError`` follows; otherwise the matrix is simply
catastrophically ill-conditioned and the answer is quietly wrong. The second is
the worse failure, and the reason this is a fix rather than a guard.

The quantity itself is perfectly well behaved there. ``X^-1 (I - exp(-X t))``
is the integral of ``exp(-Xu)`` over ``[0, t]``, and the offending mode
contributes ``(1 - e^{-mu t})/mu``, whose limit as ``mu -> 0`` is ``t``. The
singularity is removable; only the factored form has a pole.

**It was not a hypothetical.** Two of the first twelve fits of one scenario in
the reproduction of Colquhoun, Hatton & Hawkes (2003) failed this way. Both had
a rate constant sitting at its upper limit of 1e6, which puts an eigenvalue of
``QAA`` at -1e6, and the shut-time root search then walked straight into it.
:func:`test_a_rate_at_its_limit_no_longer_breaks_the_shut_time_pdf` is that
case, reduced to a mechanism built here so the test needs no cached data.
"""

import numpy as np
import numpy.linalg as nplin
import pytest

from scalcs import qmatlib as qml
from scalcs import scalcslib as scl
from scalcs.samples import samples

TRES = 25e-6


# ------------------------------------------------------------ the integral

@pytest.mark.parametrize("seed", range(8))
def test_agrees_with_the_closed_form_when_the_matrix_is_invertible(seed):
    """Against inv(M)(exp(Mt) - I), which is what it replaced."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, 6))
    M = rng.normal(0.0, 3.0, (k, k))
    t = float(rng.uniform(1e-5, 1e-2))

    got = qml.integral_expQt(M, t)
    want = nplin.inv(M) @ (qml.expQt(M, t) - np.eye(k))
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-14)


def test_a_singular_matrix_gives_a_finite_answer():
    """The whole point. The closed form raises here; this must not."""
    M = np.diag([0.0, -5000.0, -20000.0])
    got = qml.integral_expQt(M, TRES)

    with pytest.raises(nplin.LinAlgError):
        nplin.inv(M)
    assert np.all(np.isfinite(got))


def test_the_zero_mode_contributes_exactly_the_interval():
    """(e^{mu t} - 1)/mu -> t as mu -> 0, so a zero eigenvalue gives t."""
    M = np.diag([0.0, -5000.0])
    got = qml.integral_expQt(M, TRES)
    assert got[0, 0] == pytest.approx(TRES, rel=1e-14)


@pytest.mark.parametrize("mu", [1e-3, 1e-6, 1e-9, 1e-12, 0.0])
def test_it_is_continuous_through_the_pole(mu):
    """No jump, and no loss of accuracy, as an eigenvalue passes zero.

    A naive ``(exp(z) - 1) / z`` loses every significant figure as ``z``
    shrinks; this checks the answer stays on the smooth curve that the limit
    demands.
    """
    M = np.diag([-mu, -5000.0])
    got = qml.integral_expQt(M, TRES)[0, 0]
    # the exact value, expanded about zero: t (1 - mu t / 2 + ...)
    want = TRES * (1.0 - mu * TRES / 2.0)
    assert got == pytest.approx(want, rel=1e-9)


def test_the_integral_really_is_the_integral():
    """Against a quadrature, so the identity is not just asserted."""
    from scipy import integrate

    rng = np.random.default_rng(11)
    M = rng.normal(0.0, 500.0, (3, 3))
    t = 1e-3
    got = qml.integral_expQt(M, t)
    want = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            want[i, j] = integrate.quad(
                lambda u: qml.expQt(M, u)[i, j], 0.0, t, limit=200)[0]
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-12)


# ----------------------------------------------------------------- and H

def test_H_is_unchanged_away_from_the_pole():
    """The fix must not move any answer that already worked.

    Checked against the old expression, written out here, at values of s well
    away from any eigenvalue of QFF.
    """
    mec = samples.CH82()
    mec.set_eff('c', 100e-9)
    kA = mec.kA
    kF = mec.k - kA
    Q = mec.Q
    QAA, QFF = Q[:kA, :kA], Q[kA:, kA:]
    QAF, QFA = Q[:kA, kA:], Q[kA:, :kA]

    for s in (-1.0, -137.0, -4321.0, -98765.0):
        X = s * np.eye(kF) - QFF
        old = QAA + QAF @ nplin.inv(X) @ (np.eye(kF) - qml.expQt(-X, TRES)) @ QFA
        new = qml.H(s, TRES, QAA, QFF, QAF, QFA, kF)
        np.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-12)


def test_H_is_finite_and_continuous_through_the_pole():
    """The property that matters, and it does not depend on floating-point luck.

    Whether ``inv(sI - QFF)`` actually *raises* at a computed eigenvalue is a
    matter of whether that eigenvalue is exactly representable -- for a
    mechanism whose rates are round numbers it often is, which is how the real
    failure arose, but in general the matrix merely becomes catastrophically
    ill-conditioned instead. So the test is that H is finite **at** the pole
    and agrees with its own limit from either side.
    """
    mec = samples.CH82()
    mec.set_eff('c', 100e-9)
    kA = mec.kA
    kF = mec.k - kA
    Q = mec.Q
    QAA, QFF = Q[:kA, :kA], Q[kA:, kA:]
    QAF, QFA = Q[:kA, kA:], Q[kA:, :kA]

    s = float(np.sort(nplin.eigvals(QFF).real)[0])
    # the old factored form is hopeless here, whether or not it raises
    assert nplin.cond(s * np.eye(kF) - QFF) > 1e12

    at = qml.H(s, TRES, QAA, QFF, QAF, QFA, kF)
    assert np.all(np.isfinite(at))

    step = abs(s) * 1e-6
    below = qml.H(s - step, TRES, QAA, QFF, QAF, QFA, kF)
    above = qml.H(s + step, TRES, QAA, QFF, QAF, QFA, kF)
    np.testing.assert_allclose(at, below, rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(at, above, rtol=1e-4, atol=1e-9)


def test_a_rate_at_its_limit_no_longer_breaks_the_shut_time_pdf():
    """The case from the reproduction, reduced to a mechanism built here.

    A rate constant that has run to its upper limit of 1e6 puts an eigenvalue
    of QAA there, and the shut-time root search walks across it. Two of twelve
    fits of one scenario failed exactly this way.
    """
    mec = samples.AChR_diamond()
    for rate in mec.Rates:
        if rate.name == "alpha1a":
            rate.rateconstants = 1.0e6         # at the limit, as a fit can be
    mec.update_constrains()
    mec.set_eff('c', 10e-6)

    kA = mec.kA
    kF = mec.k - kA
    Q = mec.Q
    QAA, QFF = Q[:kA, :kA], Q[kA:, kA:]
    QAF, QFA = Q[:kA, kA:], Q[kA:, :kA]

    assert np.min(np.abs(nplin.eigvals(QAA).real + 1e6)) < 1.0, \
        "the mechanism should carry an eigenvalue at the limit"

    pdf = scl.HJC_shut_time_pdf(np.array([1e-3]), TRES, Q, QAA, QAF, QFF, QFA)
    assert np.all(np.isfinite(pdf))
    assert pdf[0] > 0.0
