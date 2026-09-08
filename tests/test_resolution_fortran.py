"""``impose_resolution`` against the rules transcribed from the Fortran.

``scsim.impose_resolution`` is checked here against an independent
implementation of the same definition, transcribed from ``RESHJC2.FOR`` in
``DCPROGS/DCFORTRAN`` (``Fort90/HJCFIT/fixes/``) -- the routine that imposed
the dead time for every published HJCFIT result, Colquhoun, Hatton & Hawkes
(2003) among them.

The point is that the two implementations owe each other nothing. Testing
``impose_resolution`` against values it produced itself pins whatever it does;
testing it against a separate reading of the original pins whether what it does
is *right*. The rules are short enough to transcribe faithfully:

``RESOLV`` (lines 1045-1061)
    an interval is resolvable when its duration is at least ``treso`` if open
    or ``tresg`` if shut. HJCFIT's simulate-and-fit path sets ``tresg = treso``
    (``Hjcfit1-09122003.for`` line 2670), so there is one dead time.
the shut branch (line 419)
    a concatenated shut group ends **only** when the next interval is open
    *and* resolvable.
the open branch (line 448)
    an open group ends **only** when the next interval has a different
    amplitude, or is shut, *and* is resolvable.

Both are the Colquhoun & Sigworth definition, in which an unresolvable event is
absorbed into the interval in progress rather than split between neighbours.

One difference is real and is asserted rather than hidden: the first interval.
``RESHJC2`` lines 364-375 concatenate an unresolvable first interval with the
second and give the result the second's conductance class, where
``impose_resolution`` keeps it as its own interval. It can affect only the
first interval of a record, and burst segmentation discards a leading shut
interval anyway, so it has never mattered in practice -- but it is a
difference, and :func:`test_first_interval_is_the_one_known_difference` says so.
"""

import numpy as np
import numpy.testing as npt
import pytest

from scalcs import scsim
from scalcs.samples import samples

TRES = 25e-6


def fortran_resolution(tints, ampls, tres, first_interval="scalcs"):
    """``RESHJC2.FOR``'s rules, transcribed.

    ``first_interval`` chooses what happens when the record's first interval is
    shorter than the dead time: ``"fortran"`` concatenates it with the second
    and takes the second's conductance class (lines 364-375), ``"scalcs"``
    keeps it as its own interval, which is what ``impose_resolution`` does.
    """
    t = np.asarray(tints, float)
    a = np.asarray(ampls, float)
    n = t.size
    if n == 0:
        return np.array([]), np.array([])

    def resolvable(i):
        return t[i] >= tres

    out_t, out_a = [], []
    if first_interval == "fortran" and n > 1 and not resolvable(0):
        out_t.append(t[0] + t[1])           # fbad, lines 364-375
        out_a.append(a[1])
        i = 2
    else:
        out_t.append(t[0])
        out_a.append(a[0])
        i = 1

    while i < n:
        if out_a[-1] == 0.0:                # a shut group (line 421)
            ends = a[i] != 0.0 and resolvable(i)
        else:                               # an open group (lines 449-451)
            different = (a[i] == 0.0) or (a[i] != out_a[-1])
            ends = different and resolvable(i)
        if ends:
            out_t.append(t[i])
            out_a.append(a[i])
        else:
            out_t[-1] += t[i]
        i += 1
    return np.array(out_t), np.array(out_a)


def simulated(nintervals=20000, seed=0, conc=30e-9):
    mec = samples.AChR_diamond()
    mec.set_eff('c', conc)
    return scsim.simulate_intervals(mec, nintmax=nintervals, seed=seed)[:2]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_agrees_with_the_fortran_rules_on_a_simulated_record(seed):
    """Interval for interval, on records long enough to exercise the rules."""
    ti, ai = simulated(seed=seed)
    got_t, got_a = scsim.impose_resolution(ti, ai, TRES)
    want_t, want_a = fortran_resolution(ti, ai, TRES)

    assert got_t.size == want_t.size
    npt.assert_array_equal(got_a, want_a)
    npt.assert_array_equal(got_t, want_t)


@pytest.mark.parametrize("tres", [1e-6, 10e-6, 25e-6, 100e-6, 500e-6])
def test_agrees_at_every_dead_time(tres):
    """From a dead time that resolves almost everything to one that loses most."""
    ti, ai = simulated(nintervals=4000, seed=7)
    got_t, got_a = scsim.impose_resolution(ti, ai, tres)
    want_t, want_a = fortran_resolution(ti, ai, tres)
    npt.assert_array_equal(got_a, want_a)
    npt.assert_array_equal(got_t, want_t)
    # and the rules really are being exercised
    assert got_t.size < ti.size


def test_first_interval_is_the_one_known_difference():
    """RESHJC2 absorbs an unresolvable first interval; impose_resolution keeps it.

    Asserted rather than hidden. If this ever starts failing, one of the two
    has changed its mind about the start of a record.
    """
    t = np.array([5e-6, 1e-3, 2e-4, 1e-3])       # first interval sub-resolution
    a = np.array([5.0, 0.0, 5.0, 0.0])

    ours_t, ours_a = scsim.impose_resolution(t, a, TRES)
    same_t, _ = fortran_resolution(t, a, TRES, first_interval="scalcs")
    fort_t, fort_a = fortran_resolution(t, a, TRES, first_interval="fortran")

    npt.assert_array_equal(ours_t, same_t)       # identical under the same rule
    assert ours_t.size == fort_t.size + 1        # the Fortran emits one fewer
    assert ours_a[0] == 5.0 and fort_a[0] == 0.0
    npt.assert_allclose(fort_t[0], t[0] + t[1])


def test_every_interval_after_the_first_is_resolvable():
    """The property both rules exist to guarantee."""
    ti, ai = simulated(nintervals=4000, seed=11)
    got_t, _ = scsim.impose_resolution(ti, ai, TRES)
    assert np.all(got_t[1:] >= TRES)


def test_the_record_still_alternates_and_conserves_time():
    ti, ai = simulated(nintervals=4000, seed=13)
    got_t, got_a = scsim.impose_resolution(ti, ai, TRES)
    assert np.all(np.diff(got_a != 0) != 0), "open and shut must alternate"
    npt.assert_allclose(got_t.sum(), ti.sum(), rtol=1e-12)


def test_a_dead_time_of_zero_changes_nothing():
    ti, ai = simulated(nintervals=200, seed=17)
    got_t, got_a = scsim.impose_resolution(ti, ai, 0.0)
    npt.assert_array_equal(got_t, ti)
    npt.assert_array_equal(got_a, ai)
