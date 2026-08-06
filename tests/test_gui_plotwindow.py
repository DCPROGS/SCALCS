"""The Qt plotting window must at least import and register its custom scale.

Two things had gone stale against modern matplotlib:

  * the Qt4 Agg backend was imported by name. matplotlib deprecated it in 3.3
    and removed it in 3.5, so the module raised on import - and because the
    import sits under a bare `except`, the error reported was the misleading
    "matplotlib module is missing".

  * SquareRootScale passed no axis to ScaleBase.__init__, which has required
    one since matplotlib 3.1, so constructing the scale raised TypeError. The
    class already receives `axis` and was simply not passing it on.

These are skipped rather than failed where matplotlib has no usable Qt binding,
since the module cannot be imported at all without one.
"""
import pytest

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')

pytest.importorskip('matplotlib.backends.backend_qtagg',
                    reason='no Qt binding available for matplotlib')

from matplotlib import scale as mscale                       # noqa: E402

from scalcs.gui import plotwindow                            # noqa: E402


def test_plotwindow_imports():
    """Guards the backend name: this raised ImportError before."""
    assert hasattr(plotwindow, 'SquareRootScale')


def test_square_root_scale_constructs():
    """Guards ScaleBase.__init__(self, axis): this raised TypeError before."""
    scale = plotwindow.SquareRootScale(axis=None)
    assert scale.name == 'sqrtscale'


def test_square_root_scale_can_draw():
    """The scale is usable on a real axis, not merely constructible."""
    import matplotlib.pyplot as plt

    mscale.register_scale(plotwindow.SquareRootScale)
    fig, ax = plt.subplots()
    try:
        ax.set_yscale('sqrtscale')
        ax.plot([0, 1, 4, 9], [0, 1, 4, 9])
        fig.canvas.draw()
        assert ax.get_yscale() == 'sqrtscale'
    finally:
        plt.close(fig)
