"""Tests for scalcs.scburst — single-channel burst calculations.

Reference mechanism: CH82 (Colquhoun & Hawkes 1982) at conc=100 nM.

Test strategy
-------------
* Property tests — shape, sign, ordering, and self-consistency relations
  that must hold for any correct mechanism.
* Regression tests — specific numerical outputs pinned from a verified run
  (CH82 at 100 nM, no dead-time correction).

Key self-consistency relations used
-------------------------------------
* phiBurst sums to 1
* endBurst entries in (0, 1]
* mean_burst_length = mean_open_time_per_burst + mean_shut_time_per_burst
* Popen_within_burst = open_time / burst_length
* P(r) for openings per burst: sum over r=1..∞ → 1
"""

import numpy as np
import pytest

from scalcs import qmatlib as qml
from scalcs import scburst
from scalcs.mechanism import Mechanism, Rate, State
from scalcs.samples.samples import CH82


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ch82():
    """CH82 5-state mechanism at conc=100 nM."""
    mec = CH82()
    mec.set_eff('c', 100e-9)
    return mec


# ---------------------------------------------------------------------------
# Regression reference values (CH82, conc=100 nM)
# Captured from scalcs source on modernise branch.
# ---------------------------------------------------------------------------
_PHIB  = [0.27536232, 0.72463768]
_ENDB  = [0.96090280, 0.20595089]   # column vector, stored as flat list
_MU    = 3.81864122                  # mean openings per burst
_MBL   = 7.32810346e-3               # mean burst length (s)
_MOP   = 7.16584520e-3               # mean total open time per burst (s)
_MSH   = 0.16225826e-3               # mean total shut time per burst (s)
_MSHB  = 3.79042853                  # mean shut time between bursts (s)
_PBST  = 0.97785808                  # Popen within burst = MOP / MBL


# ---------------------------------------------------------------------------
# phiBurst
# ---------------------------------------------------------------------------

class TestPhiBurst:

    def test_shape(self, ch82):
        phiB = scburst.phiBurst(ch82)
        assert phiB.shape == (ch82.kA,)

    def test_sums_to_one(self, ch82):
        phiB = scburst.phiBurst(ch82)
        assert phiB.sum() == pytest.approx(1.0, abs=1e-10)

    def test_non_negative(self, ch82):
        phiB = scburst.phiBurst(ch82)
        assert np.all(phiB >= 0.0)

    def test_regression(self, ch82):
        phiB = scburst.phiBurst(ch82)
        np.testing.assert_allclose(phiB, _PHIB, rtol=1e-6)


# ---------------------------------------------------------------------------
# endBurst
# ---------------------------------------------------------------------------

class TestEndBurst:

    def test_shape(self, ch82):
        endB = scburst.endBurst(ch82)
        assert endB.shape == (ch82.kA, 1)

    def test_entries_in_unit_interval(self, ch82):
        """Each entry is a probability: ∈ [0, 1]."""
        endB = scburst.endBurst(ch82)
        assert np.all(endB >= 0.0) and np.all(endB <= 1.0)

    def test_regression(self, ch82):
        endB = scburst.endBurst(ch82)
        np.testing.assert_allclose(endB.flatten(), _ENDB, rtol=1e-6)

    def test_phib_dot_endb_less_than_one(self, ch82):
        """P(burst ends after first opening) must be < 1 (multi-opening bursts exist)."""
        prob_single = np.dot(scburst.phiBurst(ch82),
                             scburst.endBurst(ch82))[0]
        assert 0.0 < prob_single < 1.0


# ---------------------------------------------------------------------------
# openings_mean
# ---------------------------------------------------------------------------

class TestOpeningsMean:

    def test_greater_than_one(self, ch82):
        """Mean openings per burst must be > 1 for CH82 (bursts have gaps)."""
        mu = scburst.openings_mean(ch82)
        assert mu > 1.0

    def test_regression(self, ch82):
        mu = scburst.openings_mean(ch82)
        assert mu == pytest.approx(_MU, rel=1e-6)

    def test_openings_distr_sums_to_one(self, ch82):
        """Sum of P(r) for r=1..20 should be very close to 1 for CH82."""
        total = sum(float(scburst.openings_distr(ch82, r)) for r in range(1, 21))
        assert total == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# length_mean / open_time_mean / shut_time_total_mean
# ---------------------------------------------------------------------------

class TestBurstLengths:

    def test_length_mean_regression(self, ch82):
        mbl = scburst.length_mean(ch82)
        assert mbl == pytest.approx(_MBL, rel=1e-6)

    def test_open_time_mean_regression(self, ch82):
        mop = scburst.open_time_mean(ch82)
        assert mop == pytest.approx(_MOP, rel=1e-6)

    def test_shut_time_total_mean_regression(self, ch82):
        msh = scburst.shut_time_total_mean(ch82)
        assert msh == pytest.approx(_MSH, rel=1e-5)

    def test_burst_length_equals_open_plus_shut(self, ch82):
        """Mean burst length = mean open time + mean shut time (Eq. 3.19, CH82)."""
        mbl = scburst.length_mean(ch82)
        mop = scburst.open_time_mean(ch82)
        msh = scburst.shut_time_total_mean(ch82)
        assert mop + msh == pytest.approx(mbl, rel=1e-6)

    def test_open_time_less_than_burst_length(self, ch82):
        assert scburst.open_time_mean(ch82) < scburst.length_mean(ch82)

    def test_popen_within_burst_regression(self, ch82):
        mbl = scburst.length_mean(ch82)
        mop = scburst.open_time_mean(ch82)
        assert mop / mbl == pytest.approx(_PBST, rel=1e-6)

    def test_popen_within_burst_less_than_one(self, ch82):
        mbl = scburst.length_mean(ch82)
        mop = scburst.open_time_mean(ch82)
        assert mop / mbl < 1.0


# ---------------------------------------------------------------------------
# shut_times_between_burst_mean
# ---------------------------------------------------------------------------

class TestShutBetweenBurst:

    def test_positive(self, ch82):
        mshb = scburst.shut_times_between_burst_mean(ch82)
        assert mshb > 0.0

    def test_greater_than_burst_length(self, ch82):
        """At 100 nM (sub-saturating), inter-burst gaps dominate."""
        mshb = scburst.shut_times_between_burst_mean(ch82)
        mbl  = scburst.length_mean(ch82)
        assert mshb > mbl

    def test_regression(self, ch82):
        mshb = scburst.shut_times_between_burst_mean(ch82)
        assert mshb == pytest.approx(_MSHB, rel=1e-5)


# ---------------------------------------------------------------------------
# length_pdf_components / open_time_total_pdf_components
# ---------------------------------------------------------------------------

class TestBurstPDFComponents:

    def test_length_pdf_components_count(self, ch82):
        """Burst length PDF has kE = kA + kB components."""
        eigs, w = scburst.length_pdf_components(ch82)
        assert len(eigs) == ch82.kE

    def test_length_pdf_components_eigs_positive(self, ch82):
        eigs, _ = scburst.length_pdf_components(ch82)
        assert np.all(eigs > 0)

    def test_open_time_total_pdf_components_count(self, ch82):
        """Total open time PDF has kA components."""
        eigs, w = scburst.open_time_total_pdf_components(ch82)
        assert len(eigs) == ch82.kA

    def test_shut_times_inside_pdf_components_count(self, ch82):
        """Shut times inside burst PDF has kB components."""
        eigs, w = scburst.shut_times_inside_burst_pdf_components(ch82)
        assert len(eigs) == ch82.kB
