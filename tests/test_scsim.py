"""Tests for scalcs.scsim -- single-channel Monte-Carlo simulation.

The module is a three-stage pipeline:

    simulate_intervals  ->  impose_resolution  ->  extract_bursts

Reference mechanisms
--------------------
CO  : 2-state (O<->C), alpha=50 s^-1, beta=20 s^-1.  kA=1, so open/shut
      intervals strictly alternate and the mean open/shut times are known
      analytically (1/alpha = 20 ms, 1/beta = 50 ms).
CH82: 5-state Colquhoun & Hawkes 1982 example at 100 nM; used for the burst
      statistics cross-check against scburst.

Strategy
--------
* deterministic property tests on the embedded-chain machinery and on the
  resolution / burst-extraction logic (hand-built records),
* seeded reproducibility,
* statistical convergence tests (marked ``slow``) against analytic means.
"""

import math

import numpy as np
import pytest

from scalcs import scsim
from scalcs import scburst
from scalcs.mechanism import Mechanism, Rate, State
from scalcs.samples.samples import CH82


ALPHA = 50.0        # CO  O->C rate (s^-1)  -> mean open  = 20 ms
BETA = 20.0         # CO  C->O rate (s^-1)  -> mean shut  = 50 ms


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def co():
    """2-state open<->closed mechanism with known analytic dwell-time means."""
    sO = State('A', 'O', 50e-12)
    sC = State('C', 'C', 0.0)
    rates = [
        Rate(ALPHA, sO, sC, name='alpha'),
        Rate(BETA,  sC, sO, name='beta'),
    ]
    return Mechanism(rates, mtitle='CO')


@pytest.fixture(scope="module")
def ch82():
    """CH82 5-state mechanism at 100 nM."""
    mec = CH82()
    mec.set_eff('c', 100e-9)
    return mec


# --------------------------------------------------------------------------- #
# transition_probability
# --------------------------------------------------------------------------- #
class TestTransitionProbability:

    def test_diagonal_zero(self, ch82):
        pi = scsim.transition_probability(ch82.Q)
        np.testing.assert_allclose(pi.diagonal(), 0.0, atol=1e-15)

    def test_row_sums_one(self, ch82):
        pi = scsim.transition_probability(ch82.Q)
        np.testing.assert_allclose(pi.sum(axis=1), 1.0, atol=1e-12)

    def test_off_diagonal_non_negative(self, ch82):
        pi = scsim.transition_probability(ch82.Q)
        assert (pi >= 0.0).all()

    def test_co_off_diagonal_is_one(self, co):
        """2-state: each state has a single exit, so pi_off = 1."""
        pi = scsim.transition_probability(co.Q)
        assert pi[0, 1] == pytest.approx(1.0)
        assert pi[1, 0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# next_state
# --------------------------------------------------------------------------- #
class TestNextState:

    def test_returns_valid_neighbour(self, ch82):
        picum = np.cumsum(scsim.transition_probability(ch82.Q), axis=1)
        tmean = -1.0 / ch82.Q.diagonal()
        import random
        random.seed(1)
        for present in range(ch82.k):
            nxt, t, a = scsim.next_state(present, picum, tmean, ch82.kA, 5)
            assert nxt != present
            assert 0 <= nxt < ch82.k
            assert t > 0.0
            assert a == (5 if nxt < ch82.kA else 0)


# --------------------------------------------------------------------------- #
# simulate_intervals  (ideal / full-resolution record)
# --------------------------------------------------------------------------- #
class TestSimulateIntervals:

    @pytest.fixture(scope="class")
    def rec(self, co):
        t, a, ns = scsim.simulate_intervals(co, nintmax=2000, seed=42)
        return t, a, ns

    def test_returns_three_values(self, rec):
        t, a, ns = rec
        assert isinstance(ns, (int, np.integer))

    def test_length_matches_nintmax(self, rec):
        t, a, ns = rec
        assert len(t) == 2000
        assert len(a) == 2000

    def test_amplitudes_strictly_alternate(self, rec):
        """No two consecutive intervals share a conductance class."""
        t, a, ns = rec
        assert np.all(a[:-1] != a[1:])

    def test_amplitudes_two_valued(self, co):
        t, a, ns = scsim.simulate_intervals(co, opamp=5, nintmax=500, seed=7)
        assert set(np.unique(a)).issubset({0.0, 5.0})

    def test_all_interval_times_positive(self, rec):
        t, a, ns = rec
        assert (t > 0.0).all()

    def test_nsojourns_at_least_nintervals(self, rec):
        t, a, ns = rec
        assert ns >= len(t)

    def test_seed_reproducible(self, co):
        t1, a1, _ = scsim.simulate_intervals(co, nintmax=300, seed=123)
        t2, a2, _ = scsim.simulate_intervals(co, nintmax=300, seed=123)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(a1, a2)

    def test_co_nsojourns_equals_nintervals(self, rec):
        """CO has one open + one shut state, so every sojourn is its own
        interval (no same-class merging)."""
        t, a, ns = rec
        assert ns == len(t)

    @pytest.mark.slow
    def test_co_mean_dwell_times(self, co):
        """Mean open ~ 1/alpha = 20 ms, mean shut ~ 1/beta = 50 ms."""
        t, a, ns = scsim.simulate_intervals(co, opamp=5, nintmax=60000, seed=2026)
        open_mean = t[a != 0].mean()
        shut_mean = t[a == 0].mean()
        assert open_mean == pytest.approx(1.0 / ALPHA, rel=0.05)
        assert shut_mean == pytest.approx(1.0 / BETA, rel=0.05)


# --------------------------------------------------------------------------- #
# impose_resolution  (Colquhoun-Sigworth dead time)
# --------------------------------------------------------------------------- #
class TestImposeResolution:

    def test_zero_tres_returns_unchanged(self):
        t = np.array([1.0, 2.0, 3.0])
        a = np.array([5.0, 0.0, 5.0])
        rt, ra = scsim.impose_resolution(t, a, 0.0)
        np.testing.assert_array_equal(rt, t)
        np.testing.assert_array_equal(ra, a)

    def test_brief_shut_merges_two_opens(self):
        """open, brief shut, open, shut, open -> the brief shut is missed and
        the two flanking opens fuse into one long opening."""
        t = np.array([1.0, 0.001, 1.0, 1.0, 1.0])
        a = np.array([5.0, 0.0, 5.0, 0.0, 5.0])
        rt, ra = scsim.impose_resolution(t, a, 0.01)
        np.testing.assert_allclose(rt, [2.001, 1.0, 1.0])
        np.testing.assert_array_equal(ra, [5.0, 0.0, 5.0])

    def test_brief_open_merges_two_shuts(self):
        t = np.array([1.0, 1.0, 0.001, 1.0, 1.0])
        a = np.array([5.0, 0.0, 5.0, 0.0, 5.0])
        rt, ra = scsim.impose_resolution(t, a, 0.01)
        np.testing.assert_allclose(rt, [1.0, 2.001, 1.0])
        np.testing.assert_array_equal(ra, [5.0, 0.0, 5.0])

    def test_resolved_intervals_alternate(self, co):
        t, a, _ = scsim.simulate_intervals(co, nintmax=4000, seed=11)
        rt, ra = scsim.impose_resolution(t, a, 5e-3)
        assert np.all(ra[:-1] != ra[1:])

    def test_resolved_intervals_meet_dead_time(self, co):
        """Every interval after the first is at least one dead time long."""
        tres = 5e-3
        t, a, _ = scsim.simulate_intervals(co, nintmax=4000, seed=12)
        rt, ra = scsim.impose_resolution(t, a, tres)
        assert (rt[1:] >= tres - 1e-15).all()


# --------------------------------------------------------------------------- #
# extract_bursts
# --------------------------------------------------------------------------- #
class TestExtractBursts:

    def test_handbuilt_three_bursts(self):
        """This record opens and closes with a long shut, so all three runs
        are bounded by separators and all three are complete bursts.

        These ends used to be dropped unconditionally. Against the Burzomato
        2004 records that lost two bursts from every file -- at 30 uM, two of
        six."""
        #          sep   A.o  A.s  A.o   sep   B.o  B.s  B.o  B.s  B.o   sep  C.o  sep
        t = np.array([1, 0.1, 0.01, 0.1, 1, 0.2, 0.02, 0.3, 0.01, 0.1, 1, 0.5, 1.0])
        a = np.array([0, 5,   0,    5,   0, 5,   0,    5,   0,    5,   0, 5,   0.0])
        lengths, nops = scsim.extract_bursts(t, a, tcrit=0.5)
        assert len(lengths) == 3
        assert lengths[0] == pytest.approx(0.21)     # 0.1+0.01+0.1
        assert lengths[1] == pytest.approx(0.63)     # 0.2+0.02+0.3+0.01+0.1
        assert lengths[2] == pytest.approx(0.5)
        assert list(nops) == [2, 3, 1]

    def test_keeps_ends_bounded_by_separators(self):
        """Both runs are bounded by long shuts, so both are complete."""
        t = np.array([1.0, 0.1, 1.0, 0.2, 1.0])
        a = np.array([0.0, 5.0, 0.0, 5.0, 0.0])
        lengths, nops = scsim.extract_bursts(t, a, tcrit=0.5)
        assert len(lengths) == 2

    def test_drops_genuinely_partial_ends(self):
        """A record that begins and ends on an opening did not show the start
        of its first burst or the end of its last, so both are dropped."""
        t = np.array([0.1, 1.0, 0.2, 1.0, 0.3])
        a = np.array([5.0, 0.0, 5.0, 0.0, 5.0])
        lengths, nops = scsim.extract_bursts(t, a, tcrit=0.5)
        assert len(lengths) == 1
        assert lengths[0] == pytest.approx(0.2)

    def test_no_separators_at_all(self):
        """With nothing to cut on, the whole record is one partial burst."""
        t = np.array([0.1, 0.01, 0.1])
        a = np.array([5.0, 0.0, 5.0])
        lengths, nops = scsim.extract_bursts(t, a, tcrit=0.5)
        assert len(lengths) == 0

    @pytest.mark.slow
    def test_ch82_mean_openings_per_burst(self, ch82):
        """At full resolution the simulated mean openings/burst must match
        scburst.openings_mean."""
        t, a, _ = scsim.simulate_intervals(ch82, nintmax=400000, seed=99)
        # critical time well separating within- and between-burst gaps
        tcrit = 1e-3
        lengths, nops = scsim.extract_bursts(t, a, tcrit)
        expected = float(np.ravel(scburst.openings_mean(ch82))[0])
        assert nops.mean() == pytest.approx(expected, rel=0.05)

    @pytest.mark.slow
    def test_ch82_mean_burst_length(self, ch82):
        t, a, _ = scsim.simulate_intervals(ch82, nintmax=400000, seed=98)
        tcrit = 1e-3
        lengths, nops = scsim.extract_bursts(t, a, tcrit)
        expected = float(np.ravel(scburst.length_mean(ch82))[0])
        assert lengths.mean() == pytest.approx(expected, rel=0.05)


# --------------------------------------------------------------------------- #
# extract_burst_intervals  (same segmentation as extract_bursts, but keeping
# the interval sequences that a missed-events likelihood needs)
# --------------------------------------------------------------------------- #
class TestExtractBurstIntervals:

    def test_handbuilt_intervals_in_order(self):
        """All three runs are bounded by separators, so all three are bursts,
        each reported as its interval sequence in order."""
        t = np.array([1, 0.1, 0.01, 0.1, 1, 0.2, 0.02, 0.3, 0.01, 0.1, 1, 0.5, 1.0])
        a = np.array([0, 5,   0,    5,   0, 5,   0,    5,   0,    5,   0, 5,   0.0])
        bursts = scsim.extract_burst_intervals(t, a, tcrit=0.5)
        assert len(bursts) == 3
        assert bursts[0] == pytest.approx([0.1, 0.01, 0.1])
        assert bursts[1] == pytest.approx([0.2, 0.02, 0.3, 0.01, 0.1])
        assert bursts[2] == pytest.approx([0.5])

    def test_drops_genuinely_partial_ends(self):
        """Begins and ends on an opening: neither end burst was seen whole."""
        t = np.array([0.1, 1.0, 0.2, 1.0, 0.3])
        a = np.array([5.0, 0.0, 5.0, 0.0, 5.0])
        bursts = scsim.extract_burst_intervals(t, a, tcrit=0.5)
        assert len(bursts) == 1
        assert bursts[0] == pytest.approx([0.2])

    def test_odd_length(self, co):
        """A burst starts and ends on an opening, so it has an odd number of
        intervals. The missed-events likelihood rejects even-length bursts."""
        t, a, _ = scsim.simulate_intervals(co, nintmax=20000, seed=7)
        bursts = scsim.extract_burst_intervals(t, a, tcrit=0.1)
        assert bursts
        assert all(len(b) % 2 == 1 for b in bursts)

    def test_agrees_with_extract_bursts(self, co):
        """Both functions report the same segmentation: one entry per burst,
        summing to the same length, with the same number of openings."""
        t, a, _ = scsim.simulate_intervals(co, nintmax=20000, seed=11)
        for tcrit in (0.05, 0.1, 0.5):
            lengths, nops = scsim.extract_bursts(t, a, tcrit)
            bursts = scsim.extract_burst_intervals(t, a, tcrit)
            assert len(bursts) == len(lengths)
            for b, length, n in zip(bursts, lengths, nops):
                assert b.sum() == pytest.approx(length)
                assert (len(b) + 1) // 2 == n


# --------------------------------------------------------------------------- #
# extract_subresolution_bursts  (recover all-sub-tres flicker bursts that
# impose_resolution merges into the bracketing long shut and so loses)
# --------------------------------------------------------------------------- #
class TestExtractSubresolutionBursts:

    TRES = 0.01

    def test_basic_flicker_burst_recovered(self):
        """long shut, [o s o s o] all < tres, long shut: a pure-flicker burst
        whose span (0.018) exceeds tres is recovered with its true opening count."""
        #             sep   o      s      o      s      o     sep
        t = np.array([1.0, 0.004, 0.003, 0.004, 0.003, 0.004, 1.0])
        a = np.array([0.0, 5.0,   0.0,   5.0,   0.0,   5.0,   0.0])
        lengths, nops = scsim.extract_subresolution_bursts(t, a, self.TRES)
        assert len(lengths) == 1
        assert lengths[0] == pytest.approx(0.018)
        assert nops[0] == 3

    def test_span_below_tres_not_counted(self):
        """A single brief opening between long shuts spans < tres -> no blip."""
        t = np.array([1.0, 0.004, 1.0])
        a = np.array([0.0, 5.0,   0.0])
        lengths, nops = scsim.extract_subresolution_bursts(t, a, self.TRES)
        assert len(lengths) == 0

    def test_run_ending_in_resolved_opening_skipped(self):
        """If the flicker is followed by a resolved (>= tres) opening the burst
        is detectable by the normal pipeline, so this recovery ignores it."""
        #             sep   o      s      O(resolved) sep
        t = np.array([1.0, 0.004, 0.003, 1.0,        1.0])
        a = np.array([0.0, 5.0,   0.0,   5.0,        0.0])
        lengths, nops = scsim.extract_subresolution_bursts(t, a, self.TRES)
        assert len(lengths) == 0

    def test_requires_preceding_long_shut(self):
        """A flicker run at the very start (no preceding long shut) is an
        incomplete burst and is not recovered."""
        #             o      s      o     sep  (leading flicker has no long shut before it)
        t = np.array([0.004, 0.003, 0.004, 1.0])
        a = np.array([5.0,   0.0,   5.0,   0.0])
        lengths, nops = scsim.extract_subresolution_bursts(t, a, self.TRES)
        assert len(lengths) == 0

    def test_two_consecutive_flicker_bursts(self):
        """Two pure-flicker bursts separated by a long shut are both recovered;
        the shared long shut is a boundary, not double counted."""
        t = np.array([1.0, 0.004, 0.003, 0.004, 1.0, 0.005, 0.004, 0.005, 1.0])
        a = np.array([0.0, 5.0,   0.0,   5.0,   0.0, 5.0,   0.0,   5.0,   0.0])
        lengths, nops = scsim.extract_subresolution_bursts(t, a, self.TRES)
        assert len(lengths) == 2
        assert lengths[0] == pytest.approx(0.011)
        assert lengths[1] == pytest.approx(0.014)
        assert list(nops) == [2, 2]

    def test_lengths_exceed_tres_and_have_openings(self):
        t = np.array([1.0, 0.004, 0.003, 0.004, 0.003, 0.004, 1.0])
        a = np.array([0.0, 5.0,   0.0,   5.0,   0.0,   5.0,   0.0])
        lengths, nops = scsim.extract_subresolution_bursts(t, a, self.TRES)
        assert (lengths > self.TRES).all()
        assert (nops >= 1).all()

    def test_recovers_what_impose_resolution_loses(self):
        """Integration: the standard pipeline loses the pure-flicker burst (it is
        merged into the long shut), while this recovers it."""
        t = np.array([1.0, 0.004, 0.003, 0.004, 0.003, 0.004, 1.0])
        a = np.array([0.0, 5.0,   0.0,   5.0,   0.0,   5.0,   0.0])
        rt, ra = scsim.impose_resolution(t, a, self.TRES)
        std_lengths, _ = scsim.extract_bursts(rt, ra, tcrit=0.5)
        assert len(std_lengths) == 0                      # lost by Colquhoun-Sigworth
        rec_lengths, _ = scsim.extract_subresolution_bursts(t, a, self.TRES)
        assert len(rec_lengths) == 1                      # recovered here


# --------------------------------------------------------------------------- #
# extract_bursts_recovered  (standard apparent bursts + recovered flicker)
# --------------------------------------------------------------------------- #
class TestExtractBurstsRecovered:

    TRES, TCRIT = 0.01, 0.5

    def _record(self):
        """Three resolved bursts (so the middle survive end-trimming) plus a
        pure-flicker burst that the standard route loses."""
        #              sep  O    sep  O    sep  o      s      o      s      o     sep  O    sep
        t = np.array([1.0, 0.2, 1.0, 0.2, 1.0, 0.004, 0.003, 0.004, 0.003, 0.004, 1.0, 0.2, 1.0])
        a = np.array([0.0, 5.0, 0.0, 5.0, 0.0, 5.0,   0.0,   5.0,   0.0,   5.0,   0.0, 5.0, 0.0])
        return t, a

    def test_equals_concatenation_of_both_routes(self):
        t, a = self._record()
        rt, ra = scsim.impose_resolution(t, a, self.TRES)
        Ls, Ns = scsim.extract_bursts(rt, ra, self.TCRIT)
        Lr, Nr = scsim.extract_subresolution_bursts(t, a, self.TRES)
        L, N = scsim.extract_bursts_recovered(t, a, self.TRES, self.TCRIT)
        assert len(L) == len(Ls) + len(Lr)
        np.testing.assert_allclose(np.sort(L), np.sort(np.concatenate([Ls, Lr])))

    def test_recovers_more_bursts_than_standard(self):
        t, a = self._record()
        rt, ra = scsim.impose_resolution(t, a, self.TRES)
        std, _ = scsim.extract_bursts(rt, ra, self.TCRIT)
        rec, _ = scsim.extract_bursts_recovered(t, a, self.TRES, self.TCRIT)
        assert len(rec) > len(std)
