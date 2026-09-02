"""Root finding for the asymptotic (missed-events) pdf.

det[W(s)] = 0 has kA roots. Locating them used to fail for most resolutions on
the shipped CH82 mechanism, in two different ways and with the set of failing
resolutions depending on the scipy version -- see #14.
"""

import numpy as np
import numpy.linalg as nplin
import pytest

from scalcs import qmatlib as qml
from scalcs import scalcslib as scl
from scalcs.samples import samples

RESOLUTIONS = [1e-5, 2e-5, 4e-5, 5e-5, 6e-5, 8e-5, 1e-4, 2e-4, 5e-4]


def configured(name, conc=100e-9):
    mec = getattr(samples, name)()
    mec.set_eff("c", conc)
    return mec


def residual(s, tres, mec):
    """|eig(H(s)) - s|: zero exactly at a root, by definition."""
    h = qml.H(s, tres, mec.QAA, mec.QII, mec.QAI, mec.QIA, mec.kI)
    return float(np.min(np.abs(nplin.eigvals(h).real - s)))


def roots_of(mec, tres):
    return scl.asymptotic_roots(tres, mec.QAA, mec.QII, mec.QAI, mec.QIA,
                                mec.kA, mec.kI)


@pytest.mark.parametrize("name", ["CO", "CCO", "CH82"])
@pytest.mark.parametrize("tres", RESOLUTIONS)
class TestRootsAreFound:

    def test_all_roots_located(self, name, tres):
        """One root per open state, at every resolution.

        CH82 used to raise IndexError here -- bisect_intervals returned fewer
        brackets than kA and asymptotic_roots indexed past the end."""
        mec = configured(name)
        assert len(roots_of(mec, tres)) == mec.kA

    def test_roots_satisfy_the_defining_equation(self, name, tres):
        """A root is an s that is its own eigenvalue of H(s). Returning
        without raising is not the same as returning a root."""
        mec = configured(name)
        for s in roots_of(mec, tres):
            assert residual(s, tres, mec) <= 1e-6 * max(abs(s), 1.0)

    def test_roots_are_negative_and_distinct(self, name, tres):
        mec = configured(name)
        r = roots_of(mec, tres)
        assert (r < 0).all()
        assert len(np.unique(r)) == len(r)


class TestBracketing:

    @pytest.mark.parametrize("tres", RESOLUTIONS)
    def test_counts_are_recomputed_when_a_bound_moves(self, tres):
        """bisect_intervals widens its bounds until they enclose every root.

        It used to widen once, by a factor of four, without recounting -- so
        the interval was entered carrying the count from the *old* bound and
        one root of two went missing."""
        mec = configured("CH82")
        sro = scl.bisect_intervals(-1e6, -1e-7, tres, mec.QAA, mec.QII,
                                   mec.QAI, mec.QIA, mec.kA, mec.kI)
        assert len(sro) == mec.kA

    @pytest.mark.parametrize("tres", RESOLUTIONS)
    def test_each_bracket_contains_its_root(self, tres):
        mec = configured("CH82")
        sro = scl.bisect_intervals(-1e6, -1e-7, tres, mec.QAA, mec.QII,
                                   mec.QAI, mec.QIA, mec.kA, mec.kI)
        roots = sorted(roots_of(mec, tres))
        for (lo, hi), s in zip(sorted(sro.tolist()), roots):
            assert lo <= s <= hi

    def test_starting_from_a_bound_inside_the_roots_still_works(self):
        """A caller may hand in a range that does not enclose the roots. The
        widening exists for exactly that, and must converge rather than give
        up after one step."""
        mec = configured("CH82")
        sro = scl.bisect_intervals(-10.0, -1e-7, 5e-5, mec.QAA, mec.QII,
                                   mec.QAI, mec.QIA, mec.kA, mec.kI)
        assert len(sro) == mec.kA


class TestUnsetConcentration:
    """A mechanism straight from scalcs.samples has never had set_eff called,
    so its association rates are unscaled and its Q eigenvalues sit near -5e8
    rather than -3e3. No root in that range can be located in double
    precision, and the old code returned whatever the bisection produced."""

    def test_raises_rather_than_returning_a_bad_root(self):
        mec = samples.CH82()
        with pytest.raises(RuntimeError, match="did not converge"):
            roots_of(mec, 1e-5)

    def test_the_message_names_the_likely_cause(self):
        mec = samples.CH82()
        with pytest.raises(RuntimeError, match="concentration"):
            roots_of(mec, 1e-5)

    def test_setting_a_concentration_fixes_it(self):
        mec = samples.CH82()
        mec.set_eff("c", 100e-9)
        assert len(roots_of(mec, 1e-5)) == mec.kA
