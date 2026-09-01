"""Deprecated alias for :mod:`scalcs.sccurves`.

The module was renamed because the old name described the wrong thing: nothing
here plots.  Every function takes a mechanism and returns arrays, which is why
it kept reading as a duplicate of EKDIST's ``ekplot`` -- an empirical histogram
module it has no overlap with at all.

Importing this module still works and re-exports everything from
:mod:`scalcs.sccurves`, with a :class:`DeprecationWarning`.  Notebooks and
scripts written against ``scplotlib`` need no change.

    from scalcs import sccurves as scpl        # preferred
    from scalcs import scplotlib as scpl       # still works, warns
"""

import warnings

from scalcs.sccurves import *          # noqa: F401,F403
from scalcs import sccurves as _sccurves

__all__ = [n for n in dir(_sccurves) if not n.startswith("_")]

warnings.warn(
    "scalcs.scplotlib has been renamed to scalcs.sccurves -- nothing in it "
    "plots, it calculates theoretical distributions and returns arrays. "
    "The old name still works but will be removed; import scalcs.sccurves.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    """Forward anything the star-import did not carry over (e.g. dunders)."""
    return getattr(_sccurves, name)
