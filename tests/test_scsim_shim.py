"""scsim's burst functions are dcio's.

M2 of the record-layer plan. Burst segmentation was written here and moved to
dcio.analysis.bursts so EKDIST and HJCFIT could share it. These names stay,
because scripts, notebooks and HJCFIT's documented entry point import them
from here -- but they must be the same functions, not a second copy that can
drift.
"""

import numpy as np
import pytest

from dcio.analysis import bursts as dcio_bursts
from dcio.formats import scn as dcio_scn
from scalcs import scsim


class TestReExport:

    @pytest.mark.parametrize("name", [
        "_burst_segments", "extract_bursts", "extract_burst_intervals"])
    def test_is_literally_dcio_s_function(self, name):
        assert getattr(scsim, name) is getattr(dcio_bursts, name)

    def test_flag_constant_comes_from_dcio(self):
        assert scsim.FLAG_UNUSABLE == dcio_scn.FLAG_UNUSABLE

    def test_still_importable_under_the_old_names(self):
        """A script that did `from scalcs.scsim import extract_bursts` keeps
        working."""
        from scalcs.scsim import (  # noqa: F401
            FLAG_UNUSABLE, extract_burst_intervals, extract_bursts,
        )


class TestSegmentationUnchanged:
    """The convention SCALCS pinned, still pinned, now through dcio."""

    def test_handbuilt_three_bursts(self):
        t = np.array([1, 0.1, 0.01, 0.1, 1, 0.2, 0.02, 0.3, 0.01, 0.1, 1, 0.5, 1.0])
        a = np.array([0, 5,   0,    5,   0, 5,   0,    5,   0,    5,   0, 5,   0.0])
        lengths, nops = scsim.extract_bursts(t, a, tcrit=0.5)
        assert len(lengths) == 3
        assert list(nops) == [2, 3, 1]

    def test_shut_exactly_tcrit_is_within_burst(self):
        t = np.array([0.1, 0.5, 0.2])
        a = np.array([5.0, 0.0, 5.0])
        assert len(scsim.extract_bursts(t, a, tcrit=0.5)[0]) == 1

    def test_unusable_interval_ends_a_burst(self):
        t = np.array([0.1, 0.01, 0.1, 0.00005])
        a = np.array([5.0, 0.0,  5.0, 0.0])
        flags = np.array([0, 0, 0, scsim.FLAG_UNUSABLE])
        lengths, _ = scsim.extract_bursts(t, a, tcrit=0.5, flags=flags)
        assert len(lengths) == 1

    def test_simulation_pipeline_still_runs_end_to_end(self):
        """scsim keeps its own simulator and its simulated-record dead time;
        only the segmentation moved."""
        from scalcs.samples import samples
        mec = samples.CH82()
        mec.set_eff("c", 100e-9)
        t, a, _ = scsim.simulate_intervals(mec, nintmax=5000, seed=11)
        rt, ra = scsim.impose_resolution(t, a, 1e-4)
        lengths, nops = scsim.extract_bursts(rt, ra, tcrit=1e-3)
        assert len(lengths) > 0
        assert (nops >= 1).all()
