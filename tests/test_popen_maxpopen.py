"""Tests for the non-monotonic branch of scalcs.popen.maxPopen.

A plain agonist-gated mechanism gives a monotonically rising Popen curve, so
the bisection branch of maxPopen is never entered and its bugs stay hidden.
Adding a fast pore blocker makes the curve rise and then fall
(Popen_corrected = Popen / (1 + c/KB)), which forces maxPopen down the
bisection path and exposes it.
"""
import math

import numpy as np
import pytest

from samples import samples
from scalcs import popen as pop


@pytest.fixture(scope='module')
def ch82_fastblk():
    """CH82 plus a fast channel blocker with KB = 1 mM.

    Setting fastKB is enough - Mechanism._set_fastKB turns fastblock on.
    """
    mec = samples.CH82()
    mec.fastKB = 1e-3
    assert mec.fastblock, 'fixture failed to enable the fast block'
    return mec


class TestMaxPopenNonMonotonic:

    def test_curve_is_non_monotonic(self, ch82_fastblk):
        """Popen must rise then fall, otherwise the bisection is never reached."""
        p_low = pop.Popen(ch82_fastblk, tres=0, conc=1e-9)
        p_mid = pop.Popen(ch82_fastblk, tres=0, conc=1e-3)
        p_high = pop.Popen(ch82_fastblk, tres=0, conc=1.0)
        assert p_mid > p_low
        assert p_mid > p_high

    def test_returns_two_values(self, ch82_fastblk):
        assert len(pop.maxPopen(ch82_fastblk, tres=0)) == 2

    def test_max_popen_in_unit_interval(self, ch82_fastblk):
        maxP, _ = pop.maxPopen(ch82_fastblk, tres=0)
        assert 0.0 < maxP <= 1.0

    def test_cmax_positive(self, ch82_fastblk):
        _, cmax = pop.maxPopen(ch82_fastblk, tres=0)
        assert cmax > 0.0

    def test_popen_at_cmax_matches_maxPopen(self, ch82_fastblk):
        """Popen(cmax) must equal the returned maxP.

        These are returned from the same call, so any disagreement means the
        bracket and the reported value have come apart.
        """
        maxP, cmax = pop.maxPopen(ch82_fastblk, tres=0)
        assert pop.Popen(ch82_fastblk, tres=0, conc=cmax) == pytest.approx(maxP, rel=1e-3)

    def test_maxP_exceeds_popen_at_nearby_concentrations(self, ch82_fastblk):
        """maxP must be >= Popen on either side of cmax - it is a maximum."""
        maxP, cmax = pop.maxPopen(ch82_fastblk, tres=0)
        for factor in (0.1, 0.5, 2.0, 10.0):
            p = pop.Popen(ch82_fastblk, tres=0, conc=cmax * factor)
            assert maxP >= p - 1e-6, (
                'maxP=%.6g < Popen(%.3g M)=%.6g' % (maxP, cmax * factor, p))

    def test_cmax_in_reasonable_range(self, ch82_fastblk):
        _, cmax = pop.maxPopen(ch82_fastblk, tres=0)
        assert 1e-6 < cmax < 100e-3

    def test_bisection_finds_the_true_maximum(self, ch82_fastblk):
        """Dense scan around cmax must not beat the reported maximum.

        This is the test that actually pins the bug: with the broken loop the
        returned cmax is the first midpoint rather than a converged bracket,
        so a scan finds a noticeably higher Popen nearby.
        """
        maxP, cmax = pop.maxPopen(ch82_fastblk, tres=0)
        c_scan = np.logspace(math.log10(cmax / 10), math.log10(cmax * 10), 200)
        p_scan = np.array([pop.Popen(ch82_fastblk, tres=0, conc=c) for c in c_scan])
        scan_max = float(p_scan.max())
        assert maxP == pytest.approx(scan_max, rel=0.02), (
            'maxPopen returned %.5g but a scan found %.5g at c = %.3g M'
            % (maxP, scan_max, float(c_scan[p_scan.argmax()])))
