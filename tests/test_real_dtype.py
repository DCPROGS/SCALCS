"""Results for a real Q matrix must come back real.

numpy 2.5 changed linalg.eig: for a real matrix whose eigenvalues are all real
it now returns complex128 arrays, where 2.4 and earlier returned float64.
Nothing in the library defended against that, so exp(Qt) - which is real by
definition for a real generator - came back complex, and everything downstream
of it followed. The visible symptoms were Popen and maxPopen returning
np.complex128, and a stream of ComplexWarnings out of popen.py where math.fabs
is applied to the result.

There are 66 call sites of eigs/eigs_sorted/expQt in the package, so this is
fixed at the source rather than at the point where it happened to be noticed.

Genuinely complex eigenvalues must still be reported as complex; only a
negligible imaginary part is dropped.
"""
import math
import warnings

import numpy as np
import pytest

from scalcs import popen as popenlib
from scalcs import qmatlib as qml
from scalcs import scalcslib as scl
from scalcs.samples import samples

TRES = 30e-6


@pytest.fixture(scope='module')
def mec():
    m = samples.CH82()
    m.set_eff('c', 1e-7)
    return m


# --- the source: eigen decomposition and the matrix exponential -------------

def test_expQt_is_real_for_a_real_matrix(mec):
    """exp(Qt) of a real generator is real; any imaginary part is round-off."""
    for M in (mec.Q, mec.QAA, mec.QFF):
        expM = qml.expQt(M, TRES)
        assert not np.iscomplexobj(expM), (
            'expQt returned %s for a real matrix' % expM.dtype)


def test_eigs_real_when_spectrum_is_real(mec):
    """CH82 has a real spectrum, so eigenvalues and spectral matrices are real."""
    for fn in (qml.eigs, qml.eigs_sorted):
        eigvals, A = fn(mec.Q)
        assert not np.iscomplexobj(eigvals), '%s: %s' % (fn.__name__, eigvals.dtype)
        assert not np.iscomplexobj(A), '%s: A is %s' % (fn.__name__, A.dtype)


def test_genuinely_complex_eigenvalues_are_preserved():
    """A rotation-like matrix has complex eigenvalues; they must not be discarded."""
    M = np.array([[0.0, -1.0], [1.0, 0.0]])       # eigenvalues +-i
    eigvals, _ = qml.eigs(M)
    assert np.iscomplexobj(eigvals)
    assert np.allclose(sorted(eigvals.imag), [-1.0, 1.0])


def test_expQt_matches_the_analytic_two_state_result():
    """CO: expQt[0,0](t) = 2/7 + 5/7 exp(-70t). Guards the real cast."""
    Q = np.array([[-50.0, 50.0], [20.0, -20.0]])
    t = 0.01
    expM = qml.expQt(Q, t)
    assert expM[0, 0] == pytest.approx(2 / 7 + 5 / 7 * math.exp(-70 * t), rel=1e-12)
    assert expM[0, 1] == pytest.approx(5 / 7 * (1 - math.exp(-70 * t)), rel=1e-12)


# --- the symptoms that were reported ----------------------------------------

def test_exact_mean_open_shut_time_is_real(mec):
    hmopen, hmshut = scl.exact_mean_open_shut_time(mec, TRES)
    assert not isinstance(hmopen, complex) and not np.iscomplexobj(hmopen)
    assert not isinstance(hmshut, complex) and not np.iscomplexobj(hmshut)


def test_Popen_with_dead_time_is_real(mec):
    p = popenlib.Popen(mec, TRES, 1e-7)
    assert not np.iscomplexobj(p), 'Popen returned %r' % (p,)
    assert 0.0 <= float(p) <= 1.0


def test_maxPopen_with_dead_time_is_real_and_silent():
    m = samples.CH82()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        maxP, cmax = popenlib.maxPopen(m, TRES)
    complex_warnings = [w for w in caught
                        if issubclass(w.category, np.exceptions.ComplexWarning)]
    assert not np.iscomplexobj(maxP), 'maxPopen returned %r' % (maxP,)
    assert not complex_warnings, '%d ComplexWarning(s)' % len(complex_warnings)
    assert float(maxP) == pytest.approx(0.953453875851444, rel=1e-9)


def test_EC50_with_dead_time_is_real():
    m = samples.CH82()
    ec50 = popenlib.EC50(m, TRES)
    assert not np.iscomplexobj(ec50)
    assert float(ec50) == pytest.approx(2.3560804576936233e-06, rel=1e-9)


def test_dead_time_results_are_unchanged_by_the_real_cast():
    """The cast must not move any number: compare tres=0 against known values."""
    m = samples.CH82()
    assert float(popenlib.maxPopen(m, 0)[0]) == pytest.approx(0.9677411458194296,
                                                              rel=1e-9)
    assert float(popenlib.EC50(m, 0)) == pytest.approx(2.4032020668474956e-06,
                                                       rel=1e-9)
