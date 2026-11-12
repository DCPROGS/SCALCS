"""SCALCS: Q-matrix calculations for single ion channel kinetics.

This file makes scalcs a regular package. Without it, Python treats scalcs as
a namespace package and merges every directory on the path that holds a scalcs
folder, so a module missing from the copy being tested is quietly imported from
another copy - an editable install, say. That once made a check report success
against one version of SCALCS while running modules from another.

Keep it light: importing scalcs should import nothing but the version.
"""
from scalcs.version import __version__  # noqa: F401
