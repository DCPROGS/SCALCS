"""The scplotlib -> sccurves rename.

The module never plotted anything; the name did the damage, by making it read
as a duplicate of EKDIST's empirical histogram module. These tests pin the new
name, the compatibility of the old one, and the property that motivated the
rename in the first place.
"""

import warnings

import numpy as np
import pytest

from scalcs import sccurves
from scalcs.samples import samples


@pytest.fixture()
def ch82():
    return samples.CH82()


class TestRename:

    def test_new_module_exposes_the_api(self):
        for name in ["Popen", "burst_length_pdf", "burst_openings_pdf",
                     "open_time_pdf", "shut_time_pdf", "scaled_pdf",
                     "corr_open_shut", "dependency_plot"]:
            assert hasattr(sccurves, name), name

    def test_old_name_still_imports(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from scalcs import scplotlib
        assert scplotlib.burst_length_pdf is sccurves.burst_length_pdf

    def test_old_name_warns(self):
        import importlib

        import scalcs.scplotlib
        with pytest.warns(DeprecationWarning, match="renamed to scalcs.sccurves"):
            importlib.reload(scalcs.scplotlib)

    def test_old_name_forwards_unknown_attributes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from scalcs import scplotlib
        assert scplotlib.np is sccurves.np


class TestReturnsArraysNotFigures:
    """The property the old name denied.  If any of these start drawing, the
    rename was wrong and the module belongs with the GUI instead."""

    def test_popen_returns_arrays(self, ch82):
        out = sccurves.Popen(ch82, tres=0.0)
        assert all(isinstance(x, np.ndarray) for x in out[:2])

    def test_burst_length_pdf_returns_two_arrays(self, ch82):
        t, f = sccurves.burst_length_pdf(ch82)
        assert isinstance(t, np.ndarray) and isinstance(f, np.ndarray)
        assert t.shape == f.shape
        assert (t > 0).all()

    def test_open_time_pdf_returns_four_arrays(self, ch82):
        """t, ideal, asymptotic and exact -- four, not two.

        tres is 80 us because CH82's missed-events root finding fails at most
        smaller values (IndexError from bisectHJC locating 1 root of 2, or
        brentq on an interval with no sign change). That fragility predates
        the rename and is not its business, but it is why this number is not
        the 50 us the rest of the suite uses."""
        out = sccurves.open_time_pdf(ch82, tres=8e-5)
        assert len(out) == 4
        assert all(isinstance(x, np.ndarray) for x in out)
        assert all(x.shape == out[0].shape for x in out)

    def test_no_pyplot_figure_is_created(self, ch82):
        """Calling the curve functions must not open a figure.  Only
        png_save_pdf_fig is allowed to touch matplotlib."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.close("all")
        before = len(plt.get_fignums())
        sccurves.burst_length_pdf(ch82)
        sccurves.open_time_pdf(ch82, tres=8e-5)
        assert len(plt.get_fignums()) == before


class TestImportableWithoutMatplotlib:
    """The module calculates curves and returns arrays; matplotlib is not a
    SCALCS dependency. It used to import pylab at module level, so every
    consumer needed matplotlib to get at functions that never draw -- and CI
    never noticed, because nothing imported the module outside the GUI tests.
    """

    def test_imports_with_matplotlib_unavailable(self):
        import importlib
        import sys

        blocked = ["matplotlib", "matplotlib.pyplot", "pylab"]
        saved = {k: sys.modules.get(k) for k in blocked}
        saved["scalcs.sccurves"] = sys.modules.pop("scalcs.sccurves", None)
        try:
            for k in blocked:
                sys.modules[k] = None          # makes `import k` raise ImportError
            module = importlib.import_module("scalcs.sccurves")
            assert hasattr(module, "burst_length_pdf")
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            importlib.import_module("scalcs.sccurves")

    def test_png_save_pdf_fig_still_exists(self):
        """It is allowed to need matplotlib -- it is the one that draws."""
        assert callable(sccurves.png_save_pdf_fig)
