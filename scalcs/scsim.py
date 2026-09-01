r"""Monte-Carlo simulation of single-channel records from a Q matrix.

The module is a three-stage pipeline:

    1. :func:`simulate_intervals`  generate an *ideal* (full-resolution) record
       of alternating open/shut intervals directly from the Q matrix
       (embedded-chain / Gillespie algorithm).
    2. :func:`impose_resolution`   apply a dead time ``tres`` to obtain the
       *apparent* (experimentally observable) record, following the
       Colquhoun-Sigworth rule for missed events.
    3. :func:`extract_bursts` / :func:`extract_burst_intervals`   group the
       apparent intervals into bursts
       using a critical shut time ``tcrit``.

Stages 1 and 2 are deliberately kept separate. The channel kinetics do not
depend on the recording resolution, so a single ideal record can be re-used to
study several dead times in post-processing -- without re-simulating and
without introducing extra Monte-Carlo noise between the conditions compared.

Conventions
-----------
* States are ordered open first: indices ``0 .. kA-1`` are open (conducting),
  ``kA .. k-1`` are shut. An interval's amplitude is ``opamp`` if open, else 0.
* Embedded-chain jump probability ``pi_ij = q_ij / -q_ii`` with ``pi_ii = 0``;
  the sojourn time in a state is exponential with mean ``-1 / q_ii``.
* An *interval* is a maximal run of same-conductance sojourns (e.g. an
  open->open transition between two open states is a single open interval that
  spans two sojourns). Hence ``n_sojourns >= n_intervals``.
"""

import math
import random

import numpy as np

#: SCN property flag marking an interval of unknown length
#: (``dcio.formats.scn.FLAG_UNUSABLE``).
FLAG_UNUSABLE = 8


def transition_probability(Q):
    r"""Embedded-chain (jump) probability matrix of a Q matrix.

    ``pi_ij = q_ij / -q_ii`` for ``i != j`` and ``pi_ii = 0``: given that the
    channel leaves state ``i``, ``pi_ij`` is the probability the jump is to
    ``j``. Each off-diagonal row therefore sums to 1.
    """
    k = Q.shape[0]
    pi = Q.copy().astype(float)
    for i in range(k):
        pi[i] = pi[i] / -Q[i, i]
        pi[i, i] = 0.0
    return pi


def next_state(present, picum, tmean, kA, opamp):
    r"""Draw the next state, its sojourn time and amplitude.

    Parameters
    ----------
    present : int
        Current state index.
    picum : ndarray
        Row-cumulative embedded-chain probabilities
        (``np.cumsum(transition_probability(Q), axis=1)``).
    tmean : ndarray
        Mean sojourn time of each state, ``-1 / Q.diagonal()`` [s].
    kA : int
        Number of open states.
    opamp : float
        Open-channel amplitude.

    Returns
    -------
    nxt : int
        Next state index (never equal to ``present``).
    t : float
        Sojourn time drawn from an exponential with mean ``tmean[nxt]`` [s].
    a : float
        Amplitude of ``nxt`` (``opamp`` if open, else 0).
    """
    r = random.random()
    possible = np.nonzero(picum[present] >= r)[0]
    # pi_ii = 0 makes the CDF flat at `present`; guard the r == 0 edge anyway.
    nxt = int(possible[possible != present][0])
    t = random.expovariate(1.0 / tmean[nxt])
    a = opamp if nxt < kA else 0.0
    return nxt, t, a


def simulate_intervals(mec, state=None, opamp=5, nintmax=5000, seed=None):
    r"""Simulate an ideal (full-resolution) single-channel record.

    Generates a sequence of alternating open and shut intervals by walking the
    embedded Markov chain of ``mec.Q`` and drawing an exponential sojourn time
    for each state visited. Consecutive sojourns in the same conductance class
    (e.g. two open states) are accumulated into a single interval, so the
    returned record strictly alternates open/shut.

    No dead time is imposed here -- this is the ground-truth record. Use
    :func:`impose_resolution` for the apparent record.

    Parameters
    ----------
    mec : Mechanism
        Q matrix must already be set for the desired concentration
        (``mec.set_eff('c', conc)``).
    state : int, optional
        Initial state. Defaults to the last state (``mec.k - 1``), the
        unliganded shut state in the sample mechanisms.
    opamp : float
        Open-channel amplitude (arbitrary units; only zero/non-zero matters).
    nintmax : int
        Number of intervals to generate.
    seed : int, optional
        Seed for the ``random`` module (for reproducibility).

    Returns
    -------
    tints : ndarray, shape (nintmax,)
        Interval durations [s], open and shut strictly alternating.
    ampls : ndarray, shape (nintmax,)
        Interval amplitudes (``opamp`` for open, 0 for shut).
    nsojourns : int
        Total number of individual state sojourns generated
        (``>= nintmax`` because same-conductance sojourns merge).
    """
    if seed is not None:
        random.seed(seed)
    if state is None:
        state = mec.k - 1

    picum = np.cumsum(transition_probability(mec.Q), axis=1)
    tmean = -1.0 / mec.Q.diagonal()                 # mean sojourn time [s]

    a = opamp if state < mec.kA else 0.0
    t = random.expovariate(1.0 / tmean[state])
    tints, ampls = [t], [a]
    nsojourns = 1

    while len(tints) < nintmax:
        state, t, a = next_state(state, picum, tmean, mec.kA, opamp)
        nsojourns += 1
        if a == ampls[-1]:                          # same conductance -> merge
            tints[-1] += t
        else:
            tints.append(t)
            ampls.append(a)

    return np.array(tints), np.array(ampls), nsojourns


def impose_resolution(tints, ampls, tres):
    r"""Impose a dead time on an alternating record (Colquhoun-Sigworth).

    Any sojourn shorter than ``tres`` is unresolved: its duration is added to
    the current resolved interval, and a following interval of the *same*
    conductance then merges into it. The result again strictly alternates
    open/shut, and every interval after the first is at least ``tres`` long.

    ``tres <= 0`` returns an unchanged copy (the ideal record).

    Parameters
    ----------
    tints, ampls : array_like
        Alternating interval record (e.g. from :func:`simulate_intervals`).
    tres : float
        Dead time [s].

    Returns
    -------
    tints, ampls : ndarray
        The apparent (resolution-imposed) record.
    """
    tints = np.asarray(tints, float)
    ampls = np.asarray(ampls, float)
    if tres <= 0.0:
        return tints.copy(), ampls.copy()

    out_t = [float(tints[0])]
    out_a = [float(ampls[0])]
    for t, a in zip(tints[1:], ampls[1:]):
        if t < tres:                                # unresolved sojourn
            out_t[-1] += t
        elif (a != 0.0) == (out_a[-1] != 0.0):      # same conductance -> merge
            out_t[-1] += t
        else:
            out_t.append(float(t))
            out_a.append(float(a))
    return np.array(out_t), np.array(out_a)


def _burst_segments(tints, ampls, tcrit, flags=None):
    r"""Split a record into bursts and return them.

    Shared by :func:`extract_bursts` and :func:`extract_burst_intervals`, which
    differ only in what they report about one and the same segmentation.

    The convention is the one EKDIST states in ``Bursts.slice_bursts`` and
    dcpyps followed:

      1. no gap longer than ``tcrit`` is required before the first burst of a
         record -- the first defined opening is a valid burst start;
      2. an unusable interval is a valid end of burst.

    Both matter. Time-course fitting in SCAN leaves the last interval of a
    record with no defined length, flagged unusable; it still ends the burst
    before it. A leading interval may likewise be bad, and is discarded, but
    the first defined opening after it starts a real burst. Dropping the runs
    at both ends -- as this once did -- loses two bursts from every record,
    which at 30 uM in the Burzomato 2004 set is a third of the data.

    Parameters
    ----------
    tints, ampls : array_like
        Alternating interval record (ideal or apparent).
    tcrit : float
        Critical shut time separating within- from between-burst gaps [s].
    flags : array_like of int, optional
        Per-interval SCN property flags. An interval is unusable when
        ``flags & 8`` is set (``dcio.formats.scn.FLAG_UNUSABLE``). Without
        them no interval is treated as unusable, which is right for a
        simulated record and wrong for an experimental one.

    Returns
    -------
    list of list of (float, float)
        One list of ``(interval, amplitude)`` pairs per burst, each starting
        and ending on an opening.
    """
    tints = np.asarray(tints, float)
    ampls = np.asarray(ampls, float)
    if tints.size == 0:
        return []

    if flags is None:
        unusable = np.zeros(tints.shape, dtype=bool)
    else:
        unusable = (np.asarray(flags, int) & FLAG_UNUSABLE) != 0

    # A burst ends at a between-burst gap or at an interval of unknown length.
    # An unusable interval has no measured duration, so it is never compared
    # with tcrit.
    separator = ((ampls == 0.0) & (tints >= tcrit) & ~unusable) | unusable

    # Trim to the first and last defined opening, so the record begins and
    # ends on one. What lies outside is a shut interval or an unusable one,
    # and neither belongs to a burst.
    opening = (ampls != 0.0) & ~unusable
    if not opening.any():
        return []
    lo, hi = int(np.argmax(opening)), int(len(opening) - np.argmax(opening[::-1]))

    raw, seg = [], []
    for t, a, sep in zip(tints[lo:hi], ampls[lo:hi], separator[lo:hi]):
        if sep:
            if seg:
                raw.append(seg)
            seg = []
        else:
            seg.append((t, a))
    if seg:
        raw.append(seg)

    bursts = []
    for seg in raw:
        while seg and seg[0][1] == 0.0:             # trim leading shut
            seg = seg[1:]
        while seg and seg[-1][1] == 0.0:            # trim trailing shut
            seg = seg[:-1]
        if seg:
            bursts.append(seg)
    return bursts


def extract_bursts(tints, ampls, tcrit, flags=None):
    r"""Split a record into bursts at shut intervals >= ``tcrit``.

    A burst is a run of intervals delimited by shut (closed) intervals at least
    ``tcrit`` long. Each burst is trimmed to start and end on an opening; the
    first and last (necessarily partial) bursts are discarded.

    Parameters
    ----------
    tints, ampls : array_like
        Alternating interval record (ideal or apparent).
    tcrit : float
        Critical shut time separating within- from between-burst gaps [s].

    Returns
    -------
    lengths : ndarray
        Burst lengths [s] (first opening start to last opening end).
    n_openings : ndarray of int
        Number of (apparent) openings in each burst.

    See Also
    --------
    extract_burst_intervals : the same bursts, as interval sequences.
    """
    bursts = _burst_segments(tints, ampls, tcrit, flags)
    lengths = [sum(t for t, _ in seg) for seg in bursts]
    nops = [sum(1 for _, a in seg if a != 0.0) for seg in bursts]
    return np.array(lengths), np.array(nops, dtype=int)


def extract_burst_intervals(tints, ampls, tcrit, flags=None):
    r"""Split a record into bursts and return the intervals of each.

    The segmentation is that of :func:`extract_bursts`, which reduces each
    burst to its length and its number of openings. Maximum-likelihood fitting
    of missed-events mechanisms needs the interval sequences themselves: the
    HJC likelihood is a product of matrices, one per interval, so the order and
    the individual durations both matter.

    Parameters
    ----------
    tints, ampls : array_like
        Alternating interval record (ideal or apparent).
    tcrit : float
        Critical shut time separating within- from between-burst gaps [s].

    Returns
    -------
    list of ndarray
        One array of interval durations [s] per burst, alternating open and
        shut and both starting and ending with an opening -- so every array has
        odd length, which is what the missed-events likelihood requires.

    See Also
    --------
    extract_bursts : the same bursts, as lengths and opening counts.
    """
    return [np.array([t for t, _ in seg], dtype=float)
            for seg in _burst_segments(tints, ampls, tcrit, flags)]


def extract_subresolution_bursts(tints, ampls, tres, tcrit=None):
    r"""Recover sub-resolution flicker bursts lost by the dead time.

    A burst is invisible to :func:`impose_resolution` (Colquhoun-Sigworth)
    precisely when **every one of its openings is shorter than ``tres``**: with
    no resolvable opening the whole burst stays at shut level and is merged into
    the long shut that precedes it, then discarded by :func:`extract_bursts` as
    part of a between-burst gap. Such a burst is nevertheless a real cluster of
    fast flicker and, if its total span exceeds ``tres``, would show up
    experimentally as a brief partially-resolved (reduced-amplitude) blip.

    This function scans the **ideal** (full-resolution) record -- e.g. straight
    from :func:`simulate_intervals`, *before* :func:`impose_resolution`. It
    segments the record at between-burst gaps (shut intervals ``>= tcrit``) and
    returns each segment that is

    * bracketed by gaps on both sides (a complete burst, not a record end);
    * started by an opening shorter than ``tres``;
    * **all of whose openings are shorter than ``tres``** (no resolvable opening,
      hence lost by the dead time -- within-burst *shuts* may be any length
      ``< tcrit``);

    reported only if its span (first-opening-start to last-opening-end) exceeds
    ``tres``. A segment containing a *resolved* opening (``>= tres``) is an
    ordinary burst already captured by :func:`extract_bursts` and is skipped, so
    the two functions partition the bursts without overlap (combine their outputs
    for a corrected count).

    Parameters
    ----------
    tints, ampls : array_like
        The *ideal* alternating interval record (open amplitude non-zero, shut
        zero), as returned by :func:`simulate_intervals`.
    tres : float
        Dead time [s]; openings shorter than this are unresolved.
    tcrit : float, optional
        Critical shut time delimiting bursts [s]. Defaults to ``tres`` (the
        literal all-sub-``tres`` flicker model, in which every within-burst shut
        is also ``< tres``). Pass the same ``tcrit`` used for
        :func:`extract_bursts` to recover, as single objects, lost bursts that
        contain *resolved* within-burst shuts (``tres <= shut < tcrit``) --
        otherwise such a burst is split into sub-``tres`` fragments and under-
        recovered.

    Returns
    -------
    lengths : ndarray
        Sub-resolution burst spans [s] (each ``> tres``).
    n_openings : ndarray of int
        True number of (sub-``tres``) openings in each recovered burst.
    """
    tints = np.asarray(tints, float)
    ampls = np.asarray(ampls, float)
    is_open = ampls != 0.0
    n = len(tints)
    if tcrit is None:
        tcrit = tres

    lengths, nops = [], []
    i = 0
    while i < n:
        # find a between-burst gap (long shut); the burst starts right after it
        if not (not is_open[i] and tints[i] >= tcrit):
            i += 1
            continue
        j = i + 1
        # accumulate the segment up to the next between-burst gap
        k = j
        has_resolved_open = False
        while k < n and not (not is_open[k] and tints[k] >= tcrit):
            if is_open[k] and tints[k] >= tres:
                has_resolved_open = True
            k += 1
        if k >= n:                                   # no terminating gap -> partial
            break
        seg_open = is_open[j:k]
        if seg_open.any() and not has_resolved_open:
            # span = first-opening-start to last-opening-end (trim trailing shut)
            last_open = j + int(np.nonzero(seg_open)[0][-1])
            span = float(tints[j:last_open + 1].sum())
            if span > tres:
                lengths.append(span)
                nops.append(int(seg_open.sum()))
        i = k                                        # resume from the terminating gap
    return np.array(lengths), np.array(nops, dtype=int)


def extract_bursts_recovered(tints, ampls, tres, tcrit):
    r"""Burst extraction that also recovers the lost sub-resolution flicker bursts.

    Two resolution-imposition routes are kept side by side:

    * **standard** -- :func:`extract_bursts` applied to the Colquhoun-Sigworth
      apparent record :func:`impose_resolution`. This loses every burst whose
      openings are all shorter than ``tres`` (they merge into the bracketing long
      shut);
    * **recovered** -- this function additionally adds the pure-flicker bursts
      found by :func:`extract_subresolution_bursts` on the *ideal* record (those
      with span ``> tres``), restoring the lost short bursts.

    It returns the **merged** burst list (standard apparent bursts followed by
    recovered flicker bursts), so a length histogram built from it fills back the
    near-``tres`` deficit left by the standard route. Recovered flicker bursts
    carry their length as the true span (first to last sub-``tres`` opening) and
    their ``n_openings`` as the true sub-``tres`` opening count (experimentally
    they would appear as a single reduced-amplitude blip).

    Parameters
    ----------
    tints, ampls : array_like
        The *ideal* (full-resolution) alternating record (from
        :func:`simulate_intervals`).
    tres : float
        Dead time [s].
    tcrit : float
        Critical shut time for the standard burst split [s].

    Returns
    -------
    lengths : ndarray
        Burst lengths [s]: standard apparent bursts then recovered flicker bursts.
    n_openings : ndarray of int
        Matching opening counts.
    """
    rt, ra = impose_resolution(tints, ampls, tres)
    Ls, Ns = extract_bursts(rt, ra, tcrit)
    Lr, Nr = extract_subresolution_bursts(tints, ampls, tres, tcrit)
    lengths = np.concatenate([Ls, Lr])
    nops = np.concatenate([Ns, Nr]).astype(int)
    return lengths, nops
