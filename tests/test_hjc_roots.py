"""Tests for AsymptoticPDFCalculator root finding (scalcs.hjclib).

Two defects are covered:

1. Overflow at large tres. The lower search bound was fixed at -1e6, so
   |s| * tres exceeded ln(float64 max) ~ 709 once tres was above about
   0.7 ms. exp((QFF - s*I)*tres) then overflowed and inf/nan reached W(s),
   raising "LinAlgError: Array must not contain infs or NaNs".

2. Infinite loop in _bisect_intervals. Subintervals containing no roots were
   pushed back onto the work list, so they split into two more empty halves
   for ever. Mechanisms with a single open state (CO, CCO) never returned.

The loop tests run in a subprocess with a timeout: a regression must show up
as a test failure, not as a hung test run.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from samples import samples
from scalcs import hjclib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runs asymptotic_roots in a fresh interpreter so a hang can be timed out.
_SNIPPET = """
import sys, numpy as np
sys.path.insert(0, r"{root}")
from samples import samples
from scalcs import hjclib
mec = samples.{mech}()
mec.set_eff("c", 1e-7)
calc = hjclib.AsymptoticPDFCalculator(mec, tres={tres})
roots = np.asarray(calc.asymptotic_roots(open={open_}))
assert np.all(np.isfinite(roots)), "non-finite roots"
print("ROOTS", " ".join("%.10g" % v for v in np.atleast_1d(roots)))
"""


def roots_within(mech, tres, open_=True, timeout=60):
    """Return the roots, or fail the test if the call does not terminate."""
    src = _SNIPPET.format(root=ROOT, mech=mech, tres=tres, open_=open_)
    try:
        r = subprocess.run([sys.executable, '-c', src], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail('%s at tres=%g did not terminate within %d s - the '
                    'bisection is looping' % (mech, tres, timeout))
    if r.returncode != 0:
        tail = [l for l in r.stderr.strip().splitlines() if l.strip()]
        pytest.fail('%s at tres=%g raised: %s'
                    % (mech, tres, tail[-1] if tail else '?'))
    line = [l for l in r.stdout.splitlines() if l.startswith('ROOTS')]
    assert line, 'no roots reported'
    return np.array([float(x) for x in line[-1].split()[1:]])


class TestLargeTresDoesNotOverflow:
    """|sas| * tres must be kept below the float64 exp limit."""

    @pytest.mark.parametrize('tres', [1e-4, 5e-4, 7e-4, 1e-3, 2e-3])
    def test_roots_are_finite(self, tres):
        roots = roots_within('CH82', tres)
        assert roots.size == 2
        assert np.all(np.isfinite(roots))
        assert np.all(roots > 0)

    def test_one_millisecond_used_to_raise(self):
        """tres = 1 ms is past the old overflow threshold (|s|*tres = 1000)."""
        mec = samples.CH82()
        mec.set_eff('c', 1e-7)
        roots = np.asarray(
            hjclib.AsymptoticPDFCalculator(mec, tres=1e-3).asymptotic_roots())
        assert np.all(np.isfinite(roots))

    def test_roots_vary_smoothly_with_tres(self):
        """No discontinuity where the bound starts being clamped (~0.7 ms)."""
        r1 = np.sort(roots_within('CH82', 6e-4))
        r2 = np.sort(roots_within('CH82', 8e-4))
        assert np.all(np.abs(r2 - r1) / r1 < 0.5)


class TestSingleOpenStateTerminates:
    """Mechanisms with kA = 1 used to loop for ever in _bisect_intervals."""

    @pytest.mark.parametrize('mech,tres', [('CO', 0.0), ('CO', 1e-4),
                                           ('CCO', 0.0), ('CCO', 1e-4)])
    def test_terminates(self, mech, tres):
        roots = roots_within(mech, tres)
        assert roots.size >= 1
        assert np.all(np.isfinite(roots))

    def test_co_ideal_root_is_the_shutting_rate(self):
        """CO at tres=0: the open-time constant is alpha, so the root is alpha."""
        roots = roots_within('CO', 0.0)
        mec = samples.CO()
        alpha = [r.unit_rate() for r in mec.Rates
                 if r.State1.name.endswith('*') or r.name.strip().startswith('alpha')]
        assert np.all(roots > 0)
        assert roots[0] == pytest.approx(min(alpha), rel=1e-6)


class TestMultiOpenStateUnaffected:
    """CH82 (kA=2, kF=3) worked before and must be unchanged."""

    def test_open_roots(self):
        roots = roots_within('CH82', 0.0, open_=True)
        assert roots.size == 2

    def test_shut_roots(self):
        roots = roots_within('CH82', 0.0, open_=False)
        assert roots.size == 3
